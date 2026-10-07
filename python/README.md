# Python 接口（C，Part 7 / 8.3）

C 依据实验文档完成 Part 7 / 8.3：分别通过共享库调用 Part 4 的 tiled 矩阵实现和 Part 8.2 的 CUDA 卷积实现。Python 代码的组织和文件命名由 C 决定。

当前共享库构建目标为 `build/liblab6_cuda.so`，导出 `gpu_matrix_multiply` / `gpu_convolve`，声明见 `include/lab6.h`。这两个入口目前分别调用尚未实现的 tiled 矩阵和 CUDA 卷积函数，返回 `LAB6_NOT_IMPLEMENTED`。

现有接口接收 host 指针，数据连续、行主序；输出由调用方分配。矩阵输入/输出为 float32，卷积图像为 uint32，滤波器/输出为 float32。调用前检查 dtype、形状和 C 连续布局；`ctypes` 参数类型匹配头文件，设置 `restype=c_int`，非零返回值按失败处理。

数据和计时口径见 `docs/CONTRACTS.md`；CUDA 运行与正式性能比较需要实验机。
