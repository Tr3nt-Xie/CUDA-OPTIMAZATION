# 卷积测试图片（A 负责）

- `images/`：原始测试图片，提交到 Git，在本文件下方注明来源。
- `generated/`：由 `scripts/prepare_images.py` 转换出的各尺寸 `uint32` 原始二进制（row-major），Git 忽略，可随时重新生成。

矩阵输入不存文件，由测试程序按固定 seed 生成。

## 图片来源

（待补充）
