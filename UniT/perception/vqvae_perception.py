# 进行下游任务的模型
#! 注意模型的选取

from typing import Dict
import torch
import torch.nn as nn
from UniT.taming.models.vqgan import VQModel, TemporalVQModel
from UniT.model.tactile.utils import quaternion_angle_loss
from UniT.model.tactile.utils import ConvPoolingHead
from UniT.model.tactile.utils import MlpHead
import copy

class VqvaePerception(nn.Module):
    def __init__(self, 
            vq_model_config,
            latent_shape=[3,28,28],
            freeze_encoder=True,
            task_type='3Dpose',
            num_classes=None,
            use_temporal_transformer=False,
            use_temporal_fusion=False
        ):
        super().__init__()
        # self.vqgan = VQModel(**vq_model_config)
        vq_cfg = copy.deepcopy(vq_model_config)
        ckpt_path = vq_cfg.pop("ckpt_path", None)
        if use_temporal_transformer:
            self.vqgan = TemporalVQModel(**vq_cfg)
        else:
            self.vqgan = VQModel(**vq_cfg)
        if ckpt_path is not None:
            self.vqgan.init_from_ckpt(ckpt_path)

        self.task_type = task_type
        self.use_temporal_transformer=use_temporal_transformer # 还是通过yaml传递
        self.use_temporal_fusion=use_temporal_fusion

        if freeze_encoder:
            for param in self.vqgan.parameters():
                param.requires_grad = False

        self.conv_pooling_head = ConvPoolingHead(latent_shape[0])
        
        conv_output_dim = self.conv_pooling_head(
            torch.randn(1, latent_shape[0], latent_shape[1], latent_shape[2])
        ).shape[1]
        
        if task_type == '6Dpose':
            self.mlp_head = MlpHead(conv_output_dim, output_dim=7)
        elif task_type == 'classification':
            assert num_classes is not None, "num_classes must be specified for classification task"
            self.mlp_head = MlpHead(conv_output_dim, output_dim=num_classes)
        elif task_type == '3Dpose':
            self.mlp_head = MlpHead(conv_output_dim, output_dim=4)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def forward(self, image):
        # batch: B H W C
        if (self.use_temporal_transformer and not self.use_temporal_fusion and image.ndim == 5):
            # 消融实验：仍然使用同一个 T=4 数据集，
            # 但 representation 只看最后一帧
            image = image[:, -1]
        tactile_feature = self.vqgan.to_latent(self.vqgan.get_input({'image': image}, 'image'))
        tactile_feature = self.conv_pooling_head(tactile_feature)
        output = self.mlp_head(tactile_feature)
        return output

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        y_hat = self.forward(image)
        
        if self.task_type == '6Dpose':
            y = batch['6Dpose']
            position_loss = torch.nn.functional.mse_loss(y_hat[:, :3], y[:, :3])
            position_error = torch.nn.functional.l1_loss(y_hat[:, :3], y[:, :3])
            rotation_loss = quaternion_angle_loss(y_hat[:, 3:], y[:, 3:])
            
            total_loss = 1000 * position_loss + rotation_loss
            # training loss, position error, rotation error
            return total_loss, position_error, rotation_loss
            
        elif self.task_type == 'classification':
            y = batch['label']
            classification_loss = torch.nn.functional.cross_entropy(y_hat, y)
            accuracy = (y_hat.argmax(dim=1) == y).float().mean()
            # classification loss, accuracy, dummy loss
            return classification_loss, accuracy, torch.tensor(0.0).to(y_hat.device)
        elif self.task_type == '3Dpose':
            y = batch['3Dpose']
            loss = quaternion_angle_loss(y_hat, y)
            # angle loss/error, dummy loss, dummy loss
            return loss, torch.tensor(0.0).to(y_hat.device), torch.tensor(0.0).to(y_hat.device)