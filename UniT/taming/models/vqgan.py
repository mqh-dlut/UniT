# 原始模型VQModel以及重写了添加了Transformer模块的VQModel
# 新增背景差分模型BackgroundDiffVQModel, 预测预测图与背景图之间的差分, 包括损失、优化器等
# 新增ContactWeightedVQModel模型, 只是多在原基础上多添加了一个接触加权损失

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse, os, sys, datetime, glob, importlib
from UniT.taming.logging_util import instantiate_from_config, get_obj_from_str
from UniT.taming.modules.diffusionmodules.model import Encoder, Decoder
from UniT.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from UniT.taming.modules.vqvae.quantize import GumbelQuantize
from UniT.taming.modules.vqvae.quantize import EMAVectorQuantizer

import torch.nn as nn
import zarr
import cv2
import numpy as np

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape) # 量化层 1024*3的码本
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1) # 1*1卷积

        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1) # quant_conv相反
        if ckpt_path is not None:
            print('init from ckpt')
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False) # 监测是否真的读取到了
        print(f"Restored from {path}")
        # print("\nMissing keys:")
        # for k in missing_keys:
        #     print("  ", k)
        # print("\nUnexpected keys:")
        # for k in unexpected_keys:
        #     print("  ", k)
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def to_latent(self, x): # 下游任务的时候用这种representation
        h1 = self.encoder(x)
        h2 = self.quant_conv(h1)
        return  h2
    

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx): # 总调用控制器
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x) # 等价于self.forward(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self): # 负责告诉Lightning哪些参数要被训练
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class TemporalTransformerFusion(torch.nn.Module):
    def __init__(
        self,
        in_dim=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_seq_len=16,
        dropout=0.0
    ):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, d_model) # 将三维扩展至64维

        # 时间位置编码
        self.temporal_pos_embed = torch.nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        # Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, # token维度
            nhead=nhead, # 注意力头数, 一般是d_model/4, d_model mod nhead一定为0
            dim_feedforward=d_model * 4, # FFN的隐藏维度(attention后的FFN)
            dropout=dropout,
            activation="gelu", # 激活函数
            batch_first=True # 这个参数规定Transformer输入格式是[B, Sequence, Feature]
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_proj = torch.nn.Linear(d_model, in_dim) # 64维变回3维通道

        # 初始时让 Temporal 分支输出接近0, 这样模型一开始近似原始UniT
        torch.nn.init.zeros_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, h_seq):
        B, T, C, H, W = h_seq.shape
        if T > self.temporal_pos_embed.shape[1]:
            raise ValueError(f"T={T} exceeds max_seq_len={self.temporal_pos_embed.shape[1]}")

        x = h_seq.permute(0, 3, 4, 1, 2).contiguous() # ->[B,H,W,T,C]
        x = x.view(B * H * W, T, C) # ->[B*H*W,T,C] 共有B*H*W个序列, 每个序列有T个时间步, 每个序列三维
        x = self.input_proj(x) # [B*H*W,T,64]

        # 加时间位置编码
        x = x + self.temporal_pos_embed[:, :T]

        # Temporal Transformer
        x = self.transformer(x) # [B*H*W,T,64]
        x = x[:, -1, :] # 只取最后一个时间位置 [B*H*W,64]

        delta = self.output_proj(x)
        delta = delta.view(B, H, W, C)
        delta = delta.permute(0, 3, 1, 2).contiguous()

        # Residual残差分支 不需要从头生成帧，只需要根据过去的三帧来学会修正多少
        current_feature = h_seq[:, -1]
        fused_feature = current_feature + delta

        return fused_feature

class TemporalVQModel(VQModel):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        temporal_dim=64,
        temporal_heads=4,
        temporal_layers=2,
        temporal_max_len=16,
        **kwargs
    ):
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            **kwargs
        )

        self.temporal_fusion = TemporalTransformerFusion(
            in_dim=ddconfig["z_channels"],
            d_model=temporal_dim,
            nhead=temporal_heads,
            num_layers=temporal_layers,
            max_seq_len=temporal_max_len,
            dropout=0.0
        )

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        # Temporal dataset:[B, T, H, W, C]
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3).contiguous() # ->[B, T, C, H, W]
        # 兼容原来的单帧输入
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2).contiguous() # ->[B, C, H, W]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return x.float()

    def encode(self, x):
        # 如果是普通单帧，保持原逻辑
        if x.ndim == 4:
            h = self.encoder(x)
            h = self.quant_conv(h)
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info

        # Temporal input
        # [B, T, C, H, W]
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            # 把 batch 和时间维合并, 方便输入Encoder中
            x_flat = x.reshape(B * T, C, H, W)
            h_flat = self.encoder(x_flat) # 预计:[B*T, 3, 16, 20]
            _, C_z, H_z, W_z = h_flat.shape

            # 恢复时间维
            h_seq = h_flat.reshape(B, T, C_z, H_z, W_z)

            #todo: 加入transformer
            # 只取每个batch最后一帧加入到transformer中, transformer会输出四帧的信息, 最后一帧已经融入了前三帧的信息
            h = self.temporal_fusion(h_seq)

            h = self.quant_conv(h)
            quant, emb_loss, info = self.quantize(h)

            return quant, emb_loss, info

        raise ValueError(
            f"Unexpected input shape: {x.shape}"
        )

    def to_latent(self, x):
        # 单帧输入：保持原始 UniT 的 downstream 行为
        if x.ndim == 4:
            h = self.encoder(x)
            h = self.quant_conv(h)
            return h

        # Temporal 输入：[B,T,C,H,W]
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            h_flat = self.encoder(x_flat)

            _, C_z, H_z, W_z = h_flat.shape
            h_seq = h_flat.reshape(B, T, C_z, H_z, W_z)

            h = self.temporal_fusion(h_seq)
            h = self.quant_conv(h)
            return h

        raise ValueError(f"Unexpected input shape: {x.shape}")

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_seq = self.get_input(batch, self.image_key) # [B,T,C,H,W]
        x_target = x_seq[:, -1] # reconstruction target是最后一帧
        xrec, qloss = self(x_seq) # [B,C,H,W]

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x_target, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x_target, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x_seq = self.get_input(batch, self.image_key)
        x_target = x_seq[:, -1]
        xrec, qloss = self(x_seq)

        aeloss, log_dict_ae = self.loss(qloss, x_target, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x_target, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    # def on_after_backward(self):
    #     if self.global_step < 5:
    #         print("\n====== Temporal gradients ======")
    #         for name, p in self.temporal_fusion.named_parameters():
    #             if p.grad is None:
    #                 print(name, "grad = None")
    #             else:
    #                 print(name, "grad mean =", p.grad.abs().mean().item())

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            [
                 {
                    "params": self.temporal_fusion.parameters(),
                    "lr": lr
                 },
                 {
                    "params":
                        list(self.encoder.parameters()) +
                        list(self.decoder.parameters()) +
                        list(self.quantize.parameters()) +
                        list(self.quant_conv.parameters()) +
                        list(self.post_quant_conv.parameters()),
                    "lr": lr, # lr * 0.1
                 }
            ],
            betas=(0.5, 0.9)
        )

        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, # lr * 0.1,
            betas=(0.5, 0.9)
        )

        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x_seq = self.get_input(batch, self.image_key)
        x_seq = x_seq.to(self.device)
        # 只显示最后一帧 ground truth
        x_target = x_seq[:, -1]
        xrec, _ = self(x_seq)

        if x_target.shape[1] > 3:
            assert xrec.shape[1] > 3
            x_target = self.to_rgb(x_target)
            xrec = self.to_rgb(xrec)

        log["inputs"] = x_target
        log["reconstructions"] = xrec

        return log
        

class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           
    


class WOVQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)



        if ckpt_path is not None:
            print('init from ckpt')
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        return h
    
    def to_latent(self, x):
        h = self.encoder(x)
        return  h
    
    def decode(self, h):
        dec = self.decoder(h)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        latent = self.encode(input)
        dec = self.decode(latent)
        return dec

    def get_input(self, batch, k):

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class BackgroundDiffVQModel(VQModel):
    """
    在原始 VQModel 上增加背景差分辅助分支。
    原始主干：
        image -> Encoder -> quant_conv -> VQ -> Decoder
    新增分支：
        quant_conv 输出 z -> diff_head -> 预测背景差分图
    """

    def __init__(self, background_zarr_path, background_frame_idx=5056, diff_weight=0.1, **kwargs):
        # 1. 创建原始 UniT 模型
        super().__init__(**kwargs)

        self.diff_weight = float(diff_weight)
        # 2. 新增背景差分预测分支, 输入是下游使用的连续特征 z
        self.diff_head = nn.Sequential(
            nn.Conv2d(kwargs["embed_dim"], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. 读取参考背景图片
        root = zarr.open(background_zarr_path, mode="r")
        background = np.asarray(
            root["data"]["tactile_image"][
                background_frame_idx
            ]
        ).astype(np.float32)

        # 4. 与 Dataset 使用相同的预处理
        background = background / 255.0
        background = cv2.resize(
            background,
            (160, 128),
            interpolation=cv2.INTER_LINEAR
        )

        # [H,W,3] -> [1,3,H,W]
        background = torch.from_numpy(
            np.ascontiguousarray(background)
        ).permute(2, 0, 1).unsqueeze(0)

        # 注册为 buffer：不参与梯度更新，但会随模型移动到 GPU
        self.register_buffer("background", background)

        print("Background loaded:", background_zarr_path, "frame:", background_frame_idx)

    def make_diff_target(self, x):
        """
        x: [B,3,128,160]
        返回真实背景差分图: [B,1,128,160]
        """
        if x.shape[1:] != self.background.shape[1:]:
            raise ValueError(
                f"Input shape {x.shape[1:]} "
                f"does not match background "
                f"{self.background.shape[1:]}"
            )

        # 每个像素 RGB 三通道的平均绝对差
        diff = torch.abs(x - self.background).mean(dim=1, keepdim=True)

        return diff

    def forward_with_aux(self, x):
        """
        同时执行原图重建和背景差分预测。
        """

        # 1. Encoder 提取特征
        h = self.encoder(x)

        # 2. 得到下游所用的连续特征
        z = self.quant_conv(h)

        # 3. 新增分支预测差分图
        pred_diff = self.diff_head(z)
        # 从特征分辨率放大至原图尺寸
        pred_diff = F.interpolate(
            pred_diff,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # 4. 原始 VQ 重建流程
        quant, qloss, _ = self.quantize(z)

        xrec = self.decode(quant)

        return xrec, qloss, pred_diff

    
    def training_step(self, batch, batch_idx, optimizer_idx):

        # 判别器：完全沿用原始 UniT
        if optimizer_idx == 1:
            return super().training_step(batch, batch_idx, optimizer_idx)

        # 自编码器：增加背景差分辅助监督
        if optimizer_idx == 0:

            x = self.get_input(batch, self.image_key)
            # 1. 同时获得重建图和预测差分图
            xrec, qloss, pred_diff = (self.forward_with_aux(x))

            # 2. 计算原始 UniT 损失
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # 3. 生成真实背景差分图
            with torch.no_grad():
                true_diff = self.make_diff_target(x)

            # 4. 计算差分辅助损失
            diff_loss = F.l1_loss(pred_diff, true_diff)

            # 5. 计算总损失
            total_loss = (aeloss + self.diff_weight * diff_loss)

            # 6. 记录训练日志
            self.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/diff_loss", diff_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/total_ae_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/diff_target_mean", true_diff.mean(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/diff_pred_mean", pred_diff.mean(), on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            return total_loss

        raise ValueError(
            f"Unexpected optimizer_idx={optimizer_idx}"
        )

    def configure_optimizers(self):
        # 继承原始 UniT 的两个优化器
        optimizers, schedulers = super().configure_optimizers()

        # 第一个优化器负责自编码器
        opt_ae = optimizers[0]

        # 把新增分支加入自编码器优化器
        opt_ae.add_param_group({
            "params": list(self.diff_head.parameters())
        })

        return optimizers, schedulers


class ContactWeightedVQModel(VQModel):
    """
    UniT + 接触变化加权重建

    保留原始 VQModel 全部主干结构。
    不增加任何预测分支。
    只在自编码器损失中增加接触变化加权项。
    """

    def __init__(self, background_zarr_path, background_frame_idx=5056, contact_lambda=1.0, diff_scale=0.12, **kwargs):
        super().__init__(**kwargs)

        self.contact_lambda = float(contact_lambda)
        self.diff_scale = float(diff_scale)

        if self.diff_scale <= 0:
            raise ValueError("diff_scale must be positive")

        # 读取参考背景
        root = zarr.open(background_zarr_path, mode="r")

        background = np.asarray(
            root["data"]["tactile_image"][
                background_frame_idx
            ]
        ).astype(np.float32) / 255.0

        # 与 Dataset 使用相同的预处理
        background = cv2.resize(
            background,
            (160, 128),
            interpolation=cv2.INTER_LINEAR
        )

        background = torch.from_numpy(
            np.ascontiguousarray(background)
        ).permute(2, 0, 1).unsqueeze(0)

        # 背景不参与训练，也不保存到 checkpoint
        self.register_buffer("background", background, persistent=False)

    @torch.no_grad()
    def make_contact_weight(self, x):
        """
        x: [B,3,128,160]

        return:
            weight: [B,1,128,160]
            diff:   [B,1,128,160]
        """
        if x.shape[1:] != self.background.shape[1:]:
            raise ValueError(
                "Input and background shape mismatch"
            )

        # 计算 RGB 平均绝对背景差异
        diff = torch.abs(x - self.background).mean(dim=1, keepdim=True)

        # 连续权重，限制到 0~1
        weight = torch.clamp(diff / self.diff_scale, min=0.0, max=1.0)

        return weight, diff

    def compute_contact_loss(self, x, xrec):
        """
        接触变化加权重建损失。
        """
        weight, diff = self.make_contact_weight(x)

        # 每个像素的平均 RGB 重建误差
        pixel_error = torch.abs(xrec - x).mean(dim=1, keepdim=True)

        # 变化明显区域贡献更大的额外损失
        contact_loss = (weight * pixel_error).mean()

        return contact_loss, weight, diff

    def training_step(self, batch, batch_idx, optimizer_idx):

        # 判别器：保持原始 UniT 训练逻辑
        if optimizer_idx == 1:
            return super().training_step(
                batch, batch_idx, optimizer_idx
            )

        if optimizer_idx == 0:

            x = self.get_input(batch, self.image_key)

            # 原始 UniT 前向传播
            xrec, qloss = self(x)

            # 原始 VQGAN 损失
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
            # 新增：接触变化加权重建损失
            contact_loss, weight, diff = (self.compute_contact_loss(x, xrec))
            total_loss = (aeloss + self.contact_lambda * contact_loss)

            # 日志记录
            self.log("train/aeloss", aeloss, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/contact_loss", contact_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/total_ae_loss", total_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/contact_weight_mean", weight.mean(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/background_diff_mean", diff.mean(), on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            return total_loss

        raise ValueError(
            f"Unexpected optimizer_idx={optimizer_idx}"
        )
