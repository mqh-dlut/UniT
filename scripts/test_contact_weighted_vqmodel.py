
import numpy as np
import cv2
import torch
import zarr
from pathlib import Path
from omegaconf import OmegaConf

from UniT.taming.models.vqgan import ContactWeightedVQModel


cfg = OmegaConf.load(
    "UniT/config/vqvae_representation_key.yaml"
)

zarr_path = str(
    Path(cfg.dataset.dataset_path)
    / "replay_buffer.zarr"
)

model = ContactWeightedVQModel(
    **cfg.model,
    background_zarr_path=zarr_path,
    background_frame_idx=5056,
    contact_lambda=1.0,
    diff_scale=0.12
)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

root = zarr.open(zarr_path, mode="r")
images = root["data"]["tactile_image"]


def load_frame(idx):
    img = np.asarray(
        images[idx]
    ).astype(np.float32) / 255.0

    img = cv2.resize(img, (160, 128))

    return torch.from_numpy(
        np.ascontiguousarray(img)
    ).permute(2, 0, 1).unsqueeze(0).to(device)


# 检查背景帧
bg = load_frame(5056)
bg_weight, bg_diff = model.make_contact_weight(bg)

print("===== Background =====")
print("Diff Mean:", bg_diff.mean().item())
print("Weight Mean:", bg_weight.mean().item())


# 检查接触帧
x = load_frame(781)

model.train()
model.zero_grad()

xrec, qloss = model(x)

contact_loss, weight, diff = (
    model.compute_contact_loss(x, xrec)
)

print("\n===== Contact Frame =====")
print("Input:", tuple(x.shape))
print("Reconstruction:", tuple(xrec.shape))
print("Diff Mean:", diff.mean().item())
print("Weight Mean:", weight.mean().item())
print("Contact Loss:", contact_loss.item())

assert torch.isfinite(contact_loss)
assert contact_loss.item() > 0

# 验证损失能更新 Encoder
contact_loss.backward()

encoder_grad = sum(
    p.grad.abs().sum().item()
    for p in model.encoder.parameters()
    if p.grad is not None
)

quant_grad = sum(
    p.grad.abs().sum().item()
    for p in model.quant_conv.parameters()
    if p.grad is not None
)

print("\n===== Gradient =====")
print("Encoder Grad Sum:", encoder_grad)
print("Quant Conv Grad Sum:", quant_grad)

assert encoder_grad > 0
assert quant_grad > 0

print("\n接触加权重建损失测试通过！")
