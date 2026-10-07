# 卷积测试图片（A 负责）

- `images/`：原始测试图片，提交到 Git，在本文件下方注明来源。
- `generated/`：由 `scripts/prepare_images.py` 转换出的各尺寸 `uint32` 原始二进制（row-major），Git 忽略，可随时重新生成。

矩阵输入不存文件，由测试程序按固定 seed 生成。卷积不指定 `--image` 时使用随机图片；本次 Part 8.1 的计时和展示都显式使用下列图片。转换命令见 `scripts/README.md`。

## 图片来源

三张图片均由本仓库 `scripts/generate_test_images.py` 原创生成，不是实拍照片，不使用外部图片。原图为 2048×2048、8-bit 灰度 PNG，像素范围 0–255；送入 C 程序时转换为 little-endian `uint32`。

| 文件 | 内容 | 验证用途 |
|---|---|---|
| `images/geometry.png` | 方框、圆环、三角形、不同宽度的斜线与渐变 | 观察轮廓、方向和零填充边界 |
| `images/frequency.png` | 渐变频率条纹、同心纹理与棋盘格 | 观察均值滤波对细密纹理的平滑及锐化后的对比变化 |
| `images/soft_scene.png` | 程序绘制并柔化的房屋、山体、树木和栅栏 | 观察柔化轮廓的边缘响应和锐化效果 |

原图 SHA-256 在 `images/manifest.json` 中。固定生成参数为尺寸 2048、seed 542；本次使用 NumPy 2.3.5、Pillow 12.3.0。`generate_test_images.py` 不覆盖内容不同的既有原图。

队友优先使用仓库中的这三张 PNG。`run_part81_cpu.py` 会核对原图哈希，再通过 `prepare_images.py` 转成 M=512/1024/2048 的 `.u32`。每个实验记录还保存对应 `.u32` 的 SHA-256，跨机器比较前必须一致；相同文件名或尺寸不足以证明输入相同。

512×512、K=3 的九份 CPU float32 输出保存在 `data/generated/<run-id>/`，可用同一脚本重新生成。它们保留负值和超出 255 的值，用于数值比较；PNG 效果图仅用于展示。
