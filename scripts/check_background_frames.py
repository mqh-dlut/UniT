"""
找背景帧
"""


import sys
from pathlib import Path
import numpy as np
import zarr
import matplotlib

# 服务器上无需打开图形窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================
# 1. 读取数据集
# ===========================

zarr_path = Path(sys.argv[1])
root = zarr.open(str(zarr_path), mode="r")

images = root["data"]["tactile_image"]
episode_ends = np.asarray(root["meta"]["episode_ends"][:])

print("图像形状:", images.shape)
print("Episode 数量:", len(episode_ends))

# 每个 Episode 的起点
episode_starts = np.concatenate(
    ([0], episode_ends[:-1])
)


# ===========================
# 2. 创建输出文件夹
# ===========================

output_dir = Path("scripts/background_check")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ===========================
# 3. 逐个 Episode 抽取图片
# ===========================

for ep_id, (start, end) in enumerate(
    zip(episode_starts, episode_ends)
):

    start = int(start)
    end = int(end)

    length = end - start

    print(
        f"\nEpisode {ep_id}: "
        f"start={start}, end={end}, "
        f"length={length}"
    )

    # 均匀选择20张图片
    sample_indices = np.linspace(
        start,
        end - 1,
        num=20,
        dtype=int
    )

    # 5行4列展示图片
    fig, axes = plt.subplots(
        5, 4,
        figsize=(16, 15)
    )

    for ax, idx in zip(
        axes.flat, sample_indices
    ):

        image = np.asarray(images[int(idx)])

        ax.imshow(image)
        ax.set_title(
            f"Frame {idx}",
            fontsize=11
        )
        ax.axis("off")

    fig.suptitle(
        f"Episode {ep_id} "
        f"(Frames {start} - {end - 1})",
        fontsize=16
    )

    plt.tight_layout()

    save_path = output_dir / (
        f"episode_{ep_id}_overview_copy.png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("已保存:", save_path)


print("\n全部 Episode 检查完成！")
