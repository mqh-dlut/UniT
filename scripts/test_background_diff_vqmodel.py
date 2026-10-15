"""
测试背景差分模型BackgroundDiffVQModel是否正确, 前向传播是否可通过
"""

import numpy as np
import cv2
import torch
import zarr
from pathlib import Path
from omegaconf import OmegaConf
import torch.nn.functional as F

from UniT.taming.models.vqgan import BackgroundDiffVQModel


# 1. 加载现有配置
cfg = OmegaConf.load(
    "UniT/config/vqvae_representation_key.yaml"
)

zarr_path = str(
    Path(cfg.dataset.dataset_path)
    / "replay_buffer.zarr"
)

# 2. 创建新的模型
model = BackgroundDiffVQModel(
    **cfg.model,
    background_zarr_path=zarr_path,
    background_frame_idx=5056
)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
model.eval()

root = zarr.open(zarr_path, mode="r")
images = root["data"]["tactile_image"]


# 3. 读取一张真实图片，使用相同预处理
def load_frame(idx):
    image = np.asarray(images[idx]).astype(
        np.float32
    ) / 255.0

    image = cv2.resize(
        image, (160, 128),
        interpolation=cv2.INTER_LINEAR
    )

    x = torch.from_numpy(
        np.ascontiguousarray(image)
    ).permute(2, 0, 1).unsqueeze(0)

    return x.to(device)


# 4. 测试背景帧和明显接触帧
with torch.no_grad():
    for idx in [5056, 781]:
        x = load_frame(idx)

        z = model.to_latent(x)
        xrec, qloss, pred_diff = (
            model.forward_with_aux(x)
        )

        true_diff = model.make_diff_target(x)

        print(f"\n===== Frame {idx} =====")
        print("Input:", tuple(x.shape))
        print("Latent:", tuple(z.shape))
        print("Reconstruction:", tuple(xrec.shape))
        print("Pred Diff:", tuple(pred_diff.shape))
        print("True Diff:", tuple(true_diff.shape))
        print("True Diff Mean:", true_diff.mean().item())
        print("Pred Diff Mean:", pred_diff.mean().item())

        assert xrec.shape == x.shape
        assert pred_diff.shape == true_diff.shape
        assert torch.isfinite(pred_diff).all()
        assert torch.isfinite(true_diff).all()

print("\n前向传播形状检查通过！")


# ================================
# 检查辅助损失的梯度传播
# ================================
print("\n========== 梯度检查 ==========")

model.train()
model.zero_grad()

# 使用真实接触帧
x = load_frame(781)

# 必须允许梯度计算
xrec, qloss, pred_diff = model.forward_with_aux(x)

with torch.no_grad():
    true_diff = model.make_diff_target(x)

diff_loss = F.l1_loss(pred_diff, true_diff)

# 这里只反向传播辅助损失
# 目的：单独验证它是否能够更新 Encoder
diff_loss.backward()


def check_grad(name, parameter):
    grad = parameter.grad

    if grad is None:
        print(name, ": grad = None")
        return False

    grad_sum = grad.abs().sum().item()

    print(name, ": grad abs sum =", grad_sum)

    return grad_sum > 0


head_ok = check_grad(
    "diff_head",
    model.diff_head[0].weight
)

quant_ok = check_grad(
    "quant_conv",
    model.quant_conv.weight
)

encoder_ok = check_grad(
    "encoder",
    next(model.encoder.parameters())
)

assert head_ok, "diff_head 没有收到有效梯度"
assert quant_ok, "quant_conv 没有收到有效梯度"
assert encoder_ok, "encoder 没有收到有效梯度"

print("\n辅助损失梯度传播检查通过！")


# ================================
# 检查优化器是否包含新增分支
# ================================

print("\n========== 优化器检查 ==========")

model.learning_rate = 1e-4

optimizers, _ = model.configure_optimizers()

ae_params = {
    id(p)
    for group in optimizers[0].param_groups
    for p in group["params"]
}

disc_params = {
    id(p)
    for group in optimizers[1].param_groups
    for p in group["params"]
}

head_params = {
    id(p)
    for p in model.diff_head.parameters()
}

assert head_params.issubset(ae_params)
assert head_params.isdisjoint(disc_params)

print("diff_head 已加入自编码器优化器")
print("diff_head 未加入判别器优化器")
print("\n优化器检查通过！")

