
import numpy as np
import cv2
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


# 1. 加载预训练数据
dataset_path = (
    "/mnt/storage/mmm/recur_UniT/"
    "allen_key_gelsight/replay_buffer.zarr"
)

root = zarr.open(dataset_path, mode="r")
images = root["data"]["tactile_image"]


# 2. 与 UniT Dataset 保持相同的图像预处理
def preprocess(image):
    image = np.asarray(image).astype(np.float32) / 255.0

    image = cv2.resize(
        image,
        (160, 128),
        interpolation=cv2.INTER_LINEAR
    )

    return image


# 3. 读取参考背景
background = preprocess(images[5056])


# 4. 选择代表性图片
indices = [
    5056,   # 参考背景
    0,      # 疑似无接触
    781,    # 明显接触
    1350,   # 疑似无接触
    1697,   # 明显接触
    2672,   # 疑似无接触
    4302,   # 明显接触
    5224,   # 疑似无接触
    7890,   # 明显接触
    9446    # 明显接触
]

thresholds = [0.04, 0.08, 0.12]


# 5. 可视化
fig, axes = plt.subplots(
    len(indices),
    2 + len(thresholds),
    figsize=(17, 3 * len(indices))
)

for row, idx in enumerate(indices):

    image = preprocess(images[idx])

    # 当前图像与参考背景的 RGB 平均绝对差
    diff = np.abs(
        image - background
    ).mean(axis=2)

    # 原始图片
    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"Original - {idx}")

    # 原始背景差分图
    axes[row, 1].imshow(
        diff,
        cmap="magma",
        vmin=0,
        vmax=0.2
    )
    axes[row, 1].set_title("Background Difference")

    print(f"\nFrame {idx}")
    print(f"Mean difference: {diff.mean():.5f}")

    for col, threshold in enumerate(thresholds, start=2):

        # 根据阈值生成变化区域掩码
        mask = diff > threshold

        # 在原图中使用红色标出变化区域
        overlay = image.copy()

        overlay[mask] = (
            0.55 * overlay[mask]
            + 0.45 * np.array([1.0, 0.0, 0.0])
        )

        axes[row, col].imshow(overlay)
        axes[row, col].set_title(
            f"Threshold = {threshold}"
        )

        # 统计有多少像素被选中
        ratio = mask.mean() * 100

        print(
            f"Threshold {threshold}: "
            f"{ratio:.2f}% pixels selected"
        )

    for ax in axes[row]:
        ax.axis("off")


plt.tight_layout()

output_dir = Path("scripts/background_check")
output_dir.mkdir(exist_ok=True)

save_path = output_dir / "contact_mask_comparison.png"

plt.savefig(
    save_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", save_path.resolve())
