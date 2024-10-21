# pointnet_cuda_eval
UCAS2024课程GPU架构与编程大作业1，编写pointnet的cuda推理程序。

## v1
全部采用naive核函数，对样本点采样1024，使用cudaMallocAsync和cudaFreeAsync提高显存利用效率（但错误的在网络内部使用了cudaDeviceSynchronize）。

## v2
优化了显存池的使用，网络内部使用cudaStreamSynchronize，在达到缓存阈值后释放显存。

## v3
将线性层和卷积层中的gemm部分替换为2d blocktiling([参考1](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE))([参考2](https://github.com/siboehm/SGEMM_CUDA))实现，对形状不规则的tensor加入边界检查，对初始1000条测试样本pad至1024条。
