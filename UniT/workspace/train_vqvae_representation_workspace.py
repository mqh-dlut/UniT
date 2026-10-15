# 表征预训练代码
#! 这里注意模型的选取以及设置了ckpt接续训练

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import pathlib
import random
import tqdm
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torchvision
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from PIL import Image as PILImage
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import wandb
import copy
from UniT.dataset.tactile_representation_dataset import TactileRepresentationDataset
from UniT.dataset.tactile_representation_dataset_ycb import TactileRepresentationDatasetYcb
from UniT.taming.models.vqgan import VQModel, WOVQModel, TemporalVQModel, ContactWeightedVQModel
from UniT.dataset.representation_wrapper import RepresentationWrapper
from UniT.taming.logging_util import *

OmegaConf.register_new_resolver("eval", eval, replace=True)
class TrainVqvaeRepresentationWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        trainer_config = {}
        trainer_config["gpus"] = cfg.gpus
        trainer_config["strategy"] = "ddp"
        trainer_config["max_epochs"] = cfg.max_epochs
        trainer_config["accumulate_grad_batches"] = cfg.accumulate_grad_batches

        # configure model
        model = None
        if cfg.with_vq:
            if cfg.get("use_temporal_transformer", False) and cfg.baseline_ckpt:
                model = TemporalVQModel(**cfg.model)
                model.init_from_ckpt(cfg.baseline_ckpt)
                name = 'temporal_vq'
                print('temporal_vq model')
            elif cfg.get("use_temporal_transformer", False) and not cfg.baseline_ckpt:
                model = TemporalVQModel(**cfg.model)
                name = 'temporal_vq_fromhead'
                print('temporal_vq_fromhead')
            elif cfg.get("use_contact_weighted", False):
                background_zarr_path = os.path.join(cfg.dataset.dataset_path, "replay_buffer.zarr")
                model = ContactWeightedVQModel(
                    **cfg.model, 
                    background_zarr_path=background_zarr_path,
                    background_frame_idx=int(cfg.get("background_frame_idx", 5056)),
                    contact_lambda=float(cfg.get("contact_lambda", 1.0)),
                    diff_scale=float(cfg.get("diff_scale", 0.12))
                )
                name = 'contact_weighted_vq'
                print('contact_weighted_vq model')
            else:
                model = VQModel(**cfg.model)
                name = 'vq'
                print('vq model')
        else:
            model = WOVQModel(**cfg.model)
            name = 'wovq'
            print('without vq model')
        self.model = model
        # print model size
        print("Model size: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        sys.path.append(os.getcwd())
        parser = argparse.ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        parser = config_to_parser(cfg.trainer, parser)
        opt, unknown = parser.parse_known_args()

        nowname = now+name+opt.postfix
        logdir = os.path.join("representation_models", nowname)
        ckptdir = os.path.join(logdir, "checkpoints")

        configs = {}
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # trainer and callbacks
        trainer_kwargs = dict()
        # add callbacks
        default_callbacks_cfg = {}
        default_callbacks_cfg["periodic_checkpoint"] = {
            "target": "UniT.taming.logging_util.PeriodicCheckpointCallback",
            "params": {
                "ckptdir": ckptdir,
                "save_interval": 1,  
                "save_last_n": 3,   
                "verbose": True,
            }
        }
        
        callbacks_cfg = lightning_config.get('callbacks') or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        self.trainer = trainer
        self.lightning_config = lightning_config
        self.cpktdir = ckptdir
        self.opt = opt

    # ==================================================
    # 可视化重建图
    # ==================================================
    def visualize_reconstruction(self, data, ckpt_path, num_show=6):
        import matplotlib.pyplot as plt
        print("\n========== Reconstruction Visualization ==========")
        print("Loading checkpoint:")
        print(ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")

        # 只用一张 GPU 做推理
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        # 直接取训练时使用的 dataloader
        train_loader = data.train_dataloader()
        batch = next(iter(train_loader))
        # 只 forward，不训练
        with torch.no_grad():
            log = self.model.log_images(batch)

        x_real = log["inputs"].detach().cpu()
        x_rec = log["reconstructions"].detach().cpu()
        
        print("\nShape:")
        print("Real           :", x_real.shape)
        print("Reconstruction :", x_rec.shape)

        print("\nValue range:")
        print(
            "Real: min={:.6f}, max={:.6f}, mean={:.6f}".format(
                x_real.min().item(),
                x_real.max().item(),
                x_real.mean().item()
            )
        )
        print(
            "Rec : min={:.6f}, max={:.6f}, mean={:.6f}".format(
                x_rec.min().item(),
                x_rec.max().item(),
                x_rec.mean().item()
            )
        )

        # 真实图本来就是 [0,1], dataset 中明确写了: astype(np.float32) / 255.
        real_vis = x_real.clamp(0, 1)
        # reconstruction 是网络输出, 为了显示先限制在合法 RGB 范围
        rec_vis = x_rec.clamp(0, 1)

        num_show = min(num_show, x_real.shape[0])
        fig, axes = plt.subplots(num_show, 2, figsize=(8, num_show * 3))
        if num_show == 1:
            axes = axes.reshape(1, 2)
        for i in range(num_show):
            real_img = real_vis[i].permute(1, 2, 0).numpy()
            rec_img = rec_vis[i].permute(1, 2, 0).numpy()

            axes[i, 0].imshow(real_img)
            axes[i, 0].set_title(f"Sample {i} - Real x_t")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(rec_img)
            axes[i, 1].set_title(f"Sample {i} - Reconstruction")
            axes[i, 1].axis("off")
        plt.tight_layout()

        save_path = "real_vs_reconstruction_epoch29.png"

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("\nSaved visualization to:")
        print(save_path)
        plt.close()
    # ==================================================

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        lightning_config = self.lightning_config
        opt = self.opt
        if cfg.datatype == 'ycb':
            human_demonstrations_img = TactileRepresentationDatasetYcb(**cfg.dataset)
        else:
            human_demonstrations_img = TactileRepresentationDataset(**cfg.dataset)
        data = RepresentationWrapper(human_demonstrations_img,cfg)

        # =================================
        # 临时增加, 以后要注释掉
        # =================================
        # self.visualize_reconstruction(
        #     data=data, 
        #     ckpt_path="/mnt/nvme0n1/home/mmm/projects/UniT/representation_models/2026-09-18T09-33-05temporal_vq/checkpoints/checkpoint-epoch=29.ckpt",
        #     num_show=6
        # )
        # return
        # ==================================

        bs, base_lr = cfg.batch_size, cfg.base_learning_rate
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches', 1)
        self.model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            self.model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        if opt.train:
            try:
                self.trainer.fit(self.model, data) # 每一轮都会进行训练和验证, 但没传参, 并不进行验证
                # resume_ckpt = "/mnt/nvme0n1/home/mmm/projects/UniT/representation_models/2026-09-18T09-33-05temporal_vq/checkpoints/checkpoint-epoch=29.ckpt"
                # self.trainer.fit(
                #     self.model,
                #     data,
                #     ckpt_path=resume_ckpt
                # )
            except Exception:
                raise
        if not opt.no_test and not self.trainer.interrupted:
            self.trainer.test(self.model, data) # 测试

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainVqvaeRepresentationWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
