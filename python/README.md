# Python 接口（C，Part 7 / 8.3）

预留：
- `lab6.py`：加载 `build/liblab6_cuda.so`，配置 ctypes 参数和返回类型，封装矩阵/卷积调用。
- `bench.py`：读取共用输入，核验结果，测 Python end_to_end 并输出统一 CSV。
- `requirements.txt`：实施时记录实际使用的 NumPy 版本。

输出是调用方分配的 float32 NumPy 数组。矩阵输入 float32；卷积图像 uint32、滤波器 float32；均验证 C contiguous、形状和类型。非零返回码必须转成失败。

共享库已导出 gpu_matrix_multiply / gpu_convolve，内部调用 B 的实现；目前算法仍返回 NOT_IMPLEMENTED。
若需 GPU compute 时间，可绑定 lab6_matmul_tiled / lab6_convolve_cuda 及 lab6_timings 结构体。
Mac 可写封装和输入检查；CUDA 运行与正式测试在实验机完成。
