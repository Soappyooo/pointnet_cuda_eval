# pointnet_cuda_eval
UCAS2024课程GPU架构与编程大作业1，编写pointnet的cuda推理程序，在3D MNIST数据集上测试。 
关于triton的训练与推理，见[这里](https://github.com/Soappyooo/pointnet_triton_train)。

## v1
全部采用naive核函数，对样本点采样1024(参考[原文](https://arxiv.org/pdf/1612.00593)对点云分类的采样数量。~~实测采样64个就能跑，可以直接把速度提升16倍~~)，使用cudaMallocAsync和cudaFreeAsync提高显存利用效率（但错误的在网络内部使用了cudaDeviceSynchronize）。在v100上的推理时间约1.4秒。

## v2
优化了显存池的使用，网络内部使用cudaStreamSynchronize，在达到缓存阈值后释放显存。在v100上的推理时间约0.9秒。

## v3
将线性层和卷积层中的gemm部分替换为2d blocktiling([参考1](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE))([参考2](https://github.com/siboehm/SGEMM_CUDA))实现，对形状不规则的tensor加入边界检查，对初始1000条测试样本pad至1024条。在v100上的推理时间约0.66秒。

## v4
去除了大部分cudaStreamSynchronize。将二维和三维输入的batchnorm核函数、bmm采用blocktiling实现。模型添加半精度进行推理和tensor存储，核函数内部运算以单精度进行。在本地以半精度推理速度提高约一倍，但服务器上变慢(1024采样点情况下)，所以仍然采用单精度。在v100上的推理时间约0.49秒。

## v5
采样点设置为128，添加了符合边界条件的情况下使用tensor core推理([参考](https://github.com/nicolaswilde/cuda-tensorcore-hgemm))。尝试用最远点采样，但是效果似乎没有随机采样好。最终以半精度推理，部分线性层使用tensor core，随机采样。在v100上的推理时间约0.043秒。
