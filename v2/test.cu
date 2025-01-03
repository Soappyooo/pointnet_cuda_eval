// - Namespace cuda
//     - Namespace kernel
//     - sgemm, conv1d, add, relu,batchnorm1d
// - Namespace nn
//     - Class Module
//     int operator()(agrs) {return forward(args);}
//     - Class Conv1d, Linear, ReLU, BatchNorm1d
// - Namespace PointNet
//     - Class STN3d, STNkd, PointNetEncoder, PointNetClassifier

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_fp16.h> // enable half precision
#include <dirent.h>
#include <fstream>
#include <hdf5/serial/H5Cpp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define Cpp14
#define DEFINE_ATTRIBUTE(attr) {#attr, typeid(attr).name()}
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define check_cuda_error(err) __check_cuda_error(err, __LINE__, __FILE__)
void __check_cuda_error(cudaError_t err, int line, const char *file)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + " at " + std::string(file) +
                                 ":" + std::to_string(line));
    }
}

namespace misc
{
template <typename T>
std::map<std::string, std::vector<T>> get_sub_dict(std::map<std::string, std::vector<T>> &dict, std::string prefix)
{
    std::map<std::string, std::vector<T>> sub_dict;
    for (auto &item : dict)
    {
        if (item.first.find(prefix) == 0)
        {
            // remove prefix and '.'
            sub_dict[item.first.substr(prefix.size() + 1)] = item.second;
        }
    }
    return sub_dict;
}

} // namespace misc

namespace cu
{
namespace kernel
{
// https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE
// dim3 blockDim(256); <- 128*128/8/8
// dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
// mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
#pragma unroll
    for (int k = 0; k < K; k += BK)
    {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride)
        {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++)
        {
#pragma unroll // 循环展开，增加指令并行度
            for (int j = 0; j < TM; j++)
            {
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++)
    {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}

// dim3 blockDim(256); <- 128*128/8/8
// dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
// mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void myhgemm_v4(int M, int N, int K, float alpha, half *A, half *B, float beta, half *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
#pragma unroll
    for (int k = 0; k < K; k += BK)
    {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride)
        {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++)
        {
#pragma unroll // 循环展开，增加指令并行度
            for (int j = 0; j < TM; j++)
            {
                for (int l = 0; l < TN; l++)
                    // tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
                    tmp[j][l] += __half2float(As[(ty + j) * BK + i]) * __half2float(Bs[tx + l + i * BN]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++)
    {
        for (int l = 0; l < TN; l++)
            // C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
            C[(ty + j) * N + tx + l] = __float2half(alpha * tmp[j][l] + beta * __half2float(C[(ty + j) * N + tx + l]));
    }
}

__global__ void Sadd(const float *input1, const float *input2, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input1[idx] + input2[idx];
    }
}

__global__ void Sgemm(const float *input1, const float *input2, float *output, int m, int n, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n)
    {
        float sum = 0;
        for (int i = 0; i < k; ++i)
        {
            sum += input1[row * k + i] * input2[i * n + col];
        }
        output[row * n + col] = sum;
    }
}

__global__ void Sbmm(const float *input1, const float *input2, float *output, int m, int n, int k, int batch_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (row < m && col < n && batch < batch_size)
    {
        float sum = 0;
        for (int i = 0; i < k; ++i)
        {
            sum += input1[batch * m * k + row * k + i] * input2[batch * k * n + i * n + col];
        }
        output[batch * m * n + row * n + col] = sum;
    }
}

__global__ void Sbias_add(float *input, float *bias, float *output, int size, int bias_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] + bias[idx % bias_size];
    }
}

__global__ void linear(const float *input, const float *weight, const float *bias, float *output, int m, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n)
    {
        float sum = 0;
        for (int i = 0; i < k; ++i)
        {
            sum += input[row * k + i] * weight[i * n + col];
        }
        output[row * n + col] = sum + bias[col];
    }
}

__global__ void linear(const half *input, const half *weight, const half *bias, half *output, int m, int n, int k)
{
    const half2 *input_half2 = reinterpret_cast<const half2 *>(input);
    const half2 *weight_half2 = reinterpret_cast<const half2 *>(weight);
    const half2 *bias_half2 = reinterpret_cast<const half2 *>(bias);
    half2 *output_half2 = reinterpret_cast<half2 *>(output);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n / 2)
    {
        float2 sum = make_float2(0.0f, 0.0f);
        for (int i = 0; i < k / 2; ++i)
        {
            half2 a_half2 = input_half2[row * (k / 2) + i];
            half2 b1_half2 = weight_half2[i * 2 * (n / 2) + col];
            half2 b2_half2 = weight_half2[(i * 2 + 1) * (n / 2) + col];

            float2 a = __half22float2(a_half2);
            float2 b1 = __half22float2(b1_half2);
            float2 b2 = __half22float2(b2_half2);

            sum.x += a.x * b1.x + a.y * b2.x;
            sum.y += a.x * b1.y + a.y * b2.y;
        }
        half2 bias_val = bias_half2[col];
        float2 bias_val_float2 = __half22float2(bias_val);
        sum.x += bias_val_float2.x;
        sum.y += bias_val_float2.y;
        output_half2[row * (n / 2) + col] = __float22half2_rn(sum);
    }
}

__global__ void Sconv1d_k1_d2(const float *weight, const float *input, const float *bias, float *output, int m, int n,
                              int k)
{
    // TODO: avg 1e-4 error to actual conv1d
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n)
    {
        float sum = 0;
        for (int i = 0; i < k; ++i)
        {
            sum += input[i * n + col] * weight[row * k + i];
        }
        output[row * n + col] = sum + bias[row];
    }
}

__global__ void Sconv1d_k1_d3(const float *weight, const float *input, const float *bias, float *output, int m, int n,
                              int k, int batch_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (row < m && col < n && batch < batch_size)
    {
        float sum = 0;
        for (int i = 0; i < k; ++i)
        {
            sum += input[batch * n * k + i * n + col] * weight[row * k + i];
        }
        output[batch * m * n + row * n + col] = sum + bias[row];
    }
}

template <typename T>
__global__ void Stranspose(const T *input, T *output, int dim1, int dim2, int dim3, int size, bool is_3d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    if (is_3d)
    {
        int i = idx / (dim2 * dim3);
        int j = (idx % (dim2 * dim3)) / dim3;
        int k = idx % dim3;
        output[i * dim3 * dim2 + k * dim2 + j] = input[idx];
    }
    else
    {
        int i = idx / dim2;
        int j = idx % dim2;
        output[j * dim1 + i] = input[idx];
    }
}
__global__ void Srelu(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void Sbatchnorm1d_d2(float *input, float *output, float *mean, float *var, float *gamma, float *beta,
                                float epsilon, int length, int feature_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < length && col < feature_size)
    {
        int idx = row * feature_size + col;
        output[idx] = (input[idx] - mean[col]) / sqrtf(var[col] + epsilon) * gamma[col] + beta[col];
    }
}

__global__ void Sbatchnorm1d_d3(float *input, float *output, float *mean, float *var, float *gamma, float *beta,
                                float epsilon, int channel_size, int feature_size, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        int mean_var_idx = (idx / feature_size) % channel_size;
        __syncthreads();
        float x = input[idx];
        float m = mean[mean_var_idx];
        float v = var[mean_var_idx];
        float g = gamma[mean_var_idx];
        float b = beta[mean_var_idx];

        output[idx] = (x - m) / sqrtf(v + epsilon) * g + b;
    }
}

template <typename T> __global__ void max_d2(const T *input, T *output, int dim0, int dim1, int dim2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim0 * dim1)
        return;

    int i = idx / dim1;
    int j = idx % dim1;

    T max_val = input[i * dim1 * dim2 + j * dim2];
    for (int k = 1; k < dim2; ++k)
    {
        T val = input[i * dim1 * dim2 + j * dim2 + k];
        if (val > max_val)
        {
            max_val = val;
        }
    }

    output[i * dim1 + j] = max_val;
}

// int block_size = 256;
// int grid_size = (input.size + block_size - 1) / block_size;
// kernel::Srepeat<<<grid_size, block_size>>>(data, result.data, shape[0], shape[1], shape[2], repeats[0], repeats[1],
//                                            repeats[2]);

__global__ void Srepeat(const float *input, float *output, int dim0, int dim1, int dim2, int repeat0, int repeat1,
                        int repeat2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim0 * dim1 * dim2)
        return;

    int i = idx / (dim1 * dim2);
    int j = (idx / dim2) % dim1;
    int k = idx % dim2;
    // int new_dim0 = dim0 * repeat0;
    int new_dim1 = dim1 * repeat1;
    int new_dim2 = dim2 * repeat2;

    float value = input[idx];
    for (int ii = 0; ii < repeat0; ++ii)
    {
        for (int jj = 0; jj < repeat1; ++jj)
        {
            for (int kk = 0; kk < repeat2; ++kk)
            {
                int new_i = i + ii * dim0;
                int new_j = j + jj * dim1;
                int new_k = k + kk * dim2;
                output[new_i * new_dim1 * new_dim2 + new_j * new_dim2 + new_k] = value;
            }
        }
    }
}

// int block_size = 256;
// int grid_size = (n * n + block_size - 1) / block_size;
// kernel::Sidentity<<<grid_size, block_size>>>(result.data, n);
__global__ void Sidentity(float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n)
    {
        int i = idx / n;
        int j = idx % n;
        output[idx] = i == j ? 1.0f : 0.0f;
    }
}

template <typename T> __global__ void argmax_d2(const T *input, T *output, int dim0, int dim1, int dim2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim0 * dim1)
        return;

    int i = idx / dim1;
    int j = idx % dim1;

    int max_idx = 0;
    T max_val = input[i * dim1 * dim2 + j * dim2];
    for (int k = 1; k < dim2; ++k)
    {
        T val = input[i * dim1 * dim2 + j * dim2 + k];
        if (val > max_val)
        {
            max_val = val;
            max_idx = k;
        }
    }

    output[i * dim1 + j] = max_idx;
}

__global__ void calc_accuracy_kernel(const float *output, const float *target, int size, int *correct)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (output[idx] == target[idx])
        {
            atomicAdd(correct, 1);
            // *correct += 1;
        }
    }
}

} // namespace kernel

template <typename T> class Tensor
{
  public:
    // properties
    long long id;
    T *data;
    std::vector<int> shape;
    size_t size;
    bool is_cuda;

    // methods
    Tensor() : data(nullptr), size(0), is_cuda(false)
    {
        this->id =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
    }
    Tensor(std::vector<int> shape, bool is_cuda = true) : shape(shape), is_cuda(is_cuda)
    {
        this->id =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        size = 1;
        for (int i = 0; i < shape.size(); ++i)
        {
            int dim = shape[i];
            size *= dim;
        }

        if (is_cuda)
        {
            check_cuda_error(cudaMallocAsync(&this->data, size * sizeof(T), 0));
            check_cuda_error(cudaMemset(this->data, 0, size * sizeof(T)));
        }
        else
        {
            // data = new T[size];
            this->data = new T[size];
            // set all zero
            memset(this->data, 0, size * sizeof(T));
        }
    }
    Tensor(Tensor &&other) noexcept
        : shape(std::move(other.shape)), data(other.data), size(other.size), is_cuda(other.is_cuda)
    {
        id = other.id;
        other.data = nullptr;
        other.size = 0;
    }
    Tensor(const Tensor &other) : shape(other.shape), size(other.size), is_cuda(other.is_cuda)
    {
        this->id =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        if (is_cuda)
        {
            check_cuda_error(cudaMallocAsync(&data, size * sizeof(T), 0));
            check_cuda_error(cudaMemcpy(data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        else
        {
            data = new T[size];
            std::copy(other.data, other.data + size, data);
        }
    }

    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            if (is_cuda)
            {
                check_cuda_error(cudaFreeAsync(data, 0));
            }
            else
            {
                delete[] data;
            }

            shape = std::move(other.shape);
            data = other.data;
            size = other.size;
            is_cuda = other.is_cuda;

            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    Tensor &operator=(const Tensor &other)
    {
        if (this != &other)
        {
            if (is_cuda)
            {
                check_cuda_error(cudaFreeAsync(data, 0));
            }
            else
            {
                delete[] data;
            }

            shape = other.shape;
            size = other.size;
            is_cuda = other.is_cuda;

            if (is_cuda)
            {
                check_cuda_error(cudaMallocAsync(&data, size * sizeof(T), 0));
                check_cuda_error(cudaMemcpy(data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            else
            {
                data = new T[size];
                std::copy(other.data, other.data + size, data);
            }
        }
        return *this;
    }

    Tensor(Tensor<T> &other, bool is_cuda) : shape(other.shape), is_cuda(is_cuda), size(other.size)
    {
        this->id =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        if (this->is_cuda)
        {
            if (other.is_cuda)
            {
                check_cuda_error(cudaMallocAsync(&this->data, size * sizeof(T), 0));
                check_cuda_error(cudaMemcpy(this->data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            else
            {
                check_cuda_error(cudaMallocAsync(&this->data, size * sizeof(T), 0));
                check_cuda_error(cudaMemcpy(this->data, other.data, size * sizeof(T), cudaMemcpyHostToDevice));
            }
        }
        else
        {
            if (other.is_cuda)
            {
                this->data = new T[size];
                check_cuda_error(cudaMemcpy(this->data, other.data, size * sizeof(T), cudaMemcpyDeviceToHost));
            }
            else
            {
                this->data = new T[size];
                memcpy(this->data, other.data, size * sizeof(T));
            }
        }
    }

    ~Tensor()
    {
        if (is_cuda)
        {
            check_cuda_error(cudaFreeAsync(data, 0));
        }
        else
        {
            delete[] data;
        }
    }

    // addition operator overloading
    Tensor<T> operator+(const Tensor<T> &other)
    {
        return add(*this, other);
    }

    Tensor<T> operator*(const Tensor<T> &other)
    {
        return gemm(*this, other);
    }

    Tensor<T> transpose(bool transpose_shape_only = false)
    {
        if (shape.size() != 2 && shape.size() != 3)
        {
            throw std::invalid_argument("Transpose only supports 2D or 3D tensor");
        }

        std::vector<int> new_shape = shape;
        if (shape.size() == 2)
        {
            std::swap(new_shape[0], new_shape[1]);
        }
        else if (shape.size() == 3)
        {
            std::swap(new_shape[1], new_shape[2]);
        }

        if (transpose_shape_only)
        {
            this->view(new_shape);
            return *this;
        }

        Tensor<T> output(new_shape, is_cuda);

        if (is_cuda)
        {
            T *new_data;
            check_cuda_error(cudaMallocAsync(&new_data, size * sizeof(T), 0));
            check_cuda_error(cudaDeviceSynchronize());
            int block_size = 256;
            int grid_size = (size + block_size - 1) / block_size;
            kernel::Stranspose<<<grid_size, block_size>>>(data, new_data, shape[0], shape[1],
                                                          shape.size() == 3 ? shape[2] : 1, size, shape.size() == 3);
            check_cuda_error(cudaGetLastError());
            check_cuda_error(cudaDeviceSynchronize());
            output.data = new_data;
        }
        else
        {
            T *new_data = new T[size];
            if (shape.size() == 2)
            {
                for (int i = 0; i < shape[0]; ++i)
                {
                    for (int j = 0; j < shape[1]; ++j)
                    {
                        new_data[j * shape[0] + i] = data[i * shape[1] + j];
                    }
                }
            }
            else if (shape.size() == 3)
            {
                for (int i = 0; i < shape[0]; ++i)
                {
                    for (int j = 0; j < shape[1]; ++j)
                    {
                        for (int k = 0; k < shape[2]; ++k)
                        {
                            new_data[i * shape[2] * shape[1] + k * shape[1] + j] =
                                data[i * shape[1] * shape[2] + j * shape[2] + k];
                        }
                    }
                }
            }
            output.data = new_data;
        }

        return output;
    }
    void view(std::vector<int> new_shape)
    {
        int new_size = 1;
        int negative_one_index = -1;

        for (int i = 0; i < new_shape.size(); ++i)
        {
            if (new_shape[i] == -1)
            {
                if (negative_one_index != -1)
                {
                    throw std::invalid_argument("Only one dimension can be inferred");
                }
                negative_one_index = i;
            }
            else
            {
                new_size *= new_shape[i];
            }
        }

        if (negative_one_index != -1)
        {
            if (size % new_size != 0)
            {
                throw std::invalid_argument("View size must match tensor size");
            }
            new_shape[negative_one_index] = size / new_size;
        }
        else if (new_size != size)
        {
            throw std::invalid_argument("View size must match tensor size");
        }

        shape = new_shape;
    }

    void to_cuda()
    {
        if (!is_cuda)
        {
            T *device_data;
            check_cuda_error(cudaMallocAsync(&device_data, size * sizeof(T), 0));
            check_cuda_error(cudaMemcpy(device_data, data, size * sizeof(T), cudaMemcpyHostToDevice));
            delete[] data;
            data = device_data;
            is_cuda = true;
        }
    }

    void to_cpu()
    {
        if (is_cuda)
        {
            T *h_data = new T[size];
            check_cuda_error(cudaMemcpy(h_data, data, size * sizeof(T), cudaMemcpyDeviceToHost));
            check_cuda_error(cudaFreeAsync(data, 0));
            data = h_data;
            is_cuda = false;
        }
    }

    long long index_to_offset(std::vector<int> index) const
    {
        long long offset = 0;
        if (index.size() != shape.size())
        {
            throw std::invalid_argument("Index dimensions must match tensor dimensions");
        }
        for (int i = 0; i < index.size(); ++i)
        {
            offset += index[i] * stride(i);
        }

        if (offset >= size)
        {
#ifdef Cpp20
            throw std::out_of_range(std::format("Offset out of range: {} >= {}", offset, size));
#else
            throw std::out_of_range("Offset out of range");
#endif
        }
        return offset;
    }

    // offset to index
    std::vector<int> offset_to_index(long long offset) const
    {
        if (offset >= size)
        {
#ifdef Cpp20
            throw std::out_of_range(std::format("Offset out of range: {} >= {}", offset, size));
#else
            throw std::out_of_range("Offset out of range");
#endif
        }
        std::vector<int> index(shape.size(), 0);

        for (int i = shape.size() - 1; i >= 0; --i)
        {
            index[i] = offset % shape[i];
            offset /= shape[i];
        }

        for (int i = 0; i < shape.size(); ++i)
        {
            if (index[i] >= shape[i])
            {
#ifdef Cpp20
                throw std::out_of_range(std::format("Index out of range: {} >= {}", index[i], shape[i]));
#else
                throw std::out_of_range("Index out of range");
#endif
            }
        }
        return index;
    }

    // 计算 stride
    long long stride(int dim) const
    {
        long long stride = 1;
        for (int i = dim + 1; i < shape.size(); ++i)
        {
            stride *= shape[i];
        }
        return stride;
    }

    void load(std::vector<T> &data)
    {
        if (data.size() != this->size)
        {
            throw std::invalid_argument("Data size must match tensor size");
        }
        if (is_cuda)
        {
            check_cuda_error(cudaMemcpy(this->data, data.data(), this->size * sizeof(T), cudaMemcpyHostToDevice));
        }
        else
        {
            memcpy(this->data, data.data(), this->size * sizeof(T));
        }
    }
    void load(std::vector<T> &&data)
    {
        load(data);
    }

    T get_by_index(std::vector<int> index) const
    {
        if (index.size() != shape.size())
        {
            throw std::invalid_argument("Index dimensions must match tensor dimensions");
        }
        auto offset = index_to_offset(index);
        if (is_cuda)
        {
            T value;
            check_cuda_error(cudaMemcpy(&value, data + offset, sizeof(T), cudaMemcpyDeviceToHost));
            return value;
        }
        else
        {
            return data[offset];
        }
    }

    T get_by_offset(long long offset) const
    {
        if (is_cuda)
        {
            T value;
            check_cuda_error(cudaMemcpy(&value, data + offset, sizeof(T), cudaMemcpyDeviceToHost));
            return value;
        }
        else
        {
            return data[offset];
        }
    }

    void set_by_index(std::vector<int> index, T value)
    {
        if (index.size() != shape.size())
        {
            throw std::invalid_argument("Index dimensions must match tensor dimensions");
        }
        auto offset = index_to_offset(index);
        if (is_cuda)
        {
            check_cuda_error(cudaMemcpy(data + offset, &value, sizeof(T), cudaMemcpyHostToDevice));
        }
        else
        {
            data[offset] = value;
        }
    }

    void set_by_offset(long long offset, T value)
    {
        if (is_cuda)
        {
            check_cuda_error(cudaMemcpy(data + offset, &value, sizeof(T), cudaMemcpyHostToDevice));
        }
        else
        {
            data[offset] = value;
        }
    }

    Tensor<T> repeat(std::vector<int> repeats)
    {
        auto temp_shape = this->shape;
        if (repeats.size() != temp_shape.size())
        {
            throw std::invalid_argument("Repeats dimensions must match tensor dimensions");
        }
        switch (temp_shape.size())
        {
        case 1:
            this->view({temp_shape[0], 1, 1});
            repeats.push_back(1);
            repeats.push_back(1);
            break;
        case 2:
            this->view({temp_shape[0], temp_shape[1], 1});
            repeats.push_back(1);
            break;
        case 3:
            break;
        default:
            throw std::invalid_argument("Only 1D, 2D or 3D tensor is supported");
        }
        std::vector<int> new_shape;
        for (int i = 0; i < this->shape.size(); ++i)
        {
            new_shape.push_back(this->shape[i] * repeats[i]);
        }
        Tensor<T> result(new_shape, is_cuda);
        if (is_cuda)
        {
            check_cuda_error(cudaDeviceSynchronize());
            int block_size = 256;
            int grid_size = (this->size + block_size - 1) / block_size;
            kernel::Srepeat<<<grid_size, block_size>>>(data, result.data, this->shape[0], this->shape[1],
                                                       this->shape[2], repeats[0], repeats[1], repeats[2]);
            check_cuda_error(cudaGetLastError());
            check_cuda_error(cudaDeviceSynchronize());
        }
        else
        {
            for (int i = 0; i < this->shape[0]; ++i)
            {
                for (int j = 0; j < this->shape[1]; ++j)
                {
                    for (int k = 0; k < this->shape[2]; ++k)
                    {
                        T value = get_by_index({i, j, k});
                        for (int ii = 0; ii < repeats[0]; ++ii)
                        {
                            for (int jj = 0; jj < repeats[1]; ++jj)
                            {
                                for (int kk = 0; kk < repeats[2]; ++kk)
                                {
                                    result.set_by_index(
                                        {i + ii * this->shape[0], j + jj * this->shape[1], k + kk * this->shape[2]},
                                        value);
                                }
                            }
                        }
                    }
                }
            }
        }
        switch (temp_shape.size())
        {
        case 1:
            result.view({new_shape[0]});
            this->view({temp_shape[0]});
            break;
        case 2:
            result.view({new_shape[0], new_shape[1]});
            this->view({temp_shape[0], temp_shape[1]});
            break;
        case 3:
            break;
        }
        // *this = result;
        return result;
    }

    std::string to_string() const
    {
        std::string output;
        output += "Tensor<" + std::string(typeid(T).name()) + "> of shape (";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            output += std::to_string(shape[i]);
            if (i < shape.size() - 1)
                output += ", ";
        }
        output += "), device=" + std::string(is_cuda ? "cuda" : "cpu") + "\n";
        to_string_recursive(0, 0, output);
        output += "\n";
        return output;
    }

    friend std::ostream &operator<<(std::ostream &os, const cu::Tensor<T> &tensor)
    {
        os << tensor.to_string();
        return os;
    }

    void print() const
    {
        std::cout << to_string();
    }

    void to_string_recursive(int dim, int offset, std::string &output) const
    {
        if (dim == shape.size() - 1)
        {
            output += "[";
            int limit = shape[dim];
            for (int i = 0; i < limit; ++i)
            {
                if (i >= 3 && i < limit - 3) // 跳过中间部分
                {
                    if (i == 3) // 只在第一次跳过时添加省略号
                    {
                        output += ",  ...";
                    }
                    continue;
                }
                if (this->is_cuda)
                {
                    T value;
                    check_cuda_error(cudaMemcpy(&value, data + offset + i, sizeof(T), cudaMemcpyDeviceToHost));
                    float float_value = static_cast<float>(value);
                    output += (i > 0 ? ", " : "");
#ifdef Cpp20
                    output += std::format("{:9.4f}", float_value);
#else
                    output += std::to_string(float_value);
#endif
                }
                else
                {
                    float float_value = static_cast<float>(data[offset + i]);
                    output += (i > 0 ? ", " : "");
#ifdef Cpp20
                    output += std::format("{:9.4f}", float_value);
#else
                    output += std::to_string(float_value);
#endif
                }
            }
            output += "]";
        }
        else
        {
            output += "[";
            int limit = shape[dim];
            for (int i = 0; i < limit; ++i)
            {
                if (i >= 3 && i < limit - 3) // 跳过中间部分
                {
                    if (i == 3) // 只在第一次跳过时添加省略号
                    {
                        output += ",\n" + std::string(dim + 1, ' ') + "...";
                    }
                    continue;
                }
                if (i > 0)
                    output += ",\n" + std::string((shape.size() - dim > 2 ? 1 : 0), '\n') + std::string(dim + 1, ' ');
                to_string_recursive(dim + 1, offset + i * stride(dim), output);
            }
            output += "]";
        }
    }
};

template <typename T> Tensor<T> gemm(const Tensor<T> &tensor1, const Tensor<T> &tensor2)
{
    if (tensor1.shape[1] != tensor2.shape[0])
    {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    Tensor<T> result({tensor1.shape[0], tensor2.shape[1]}, tensor1.is_cuda);

    if (tensor1.is_cuda && tensor2.is_cuda)
    {
        check_cuda_error(cudaDeviceSynchronize());
        // naive sgemm
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(tensor1.shape[0], blockDim.x), CEIL_DIV(tensor2.shape[1], blockDim.y));
        kernel::Sgemm<<<gridDim, blockDim>>>(tensor1.data, tensor2.data, result.data, tensor1.shape[0],
                                             tensor2.shape[1], tensor2.shape[0]);
        check_cuda_error(cudaGetLastError());
        check_cuda_error(cudaDeviceSynchronize());
    }
    else if (!tensor1.is_cuda && !tensor2.is_cuda)
    {
        // cpu sgemm
        for (int i = 0; i < tensor1.shape[0]; ++i)
        {
            for (int j = 0; j < tensor2.shape[1]; ++j)
            {
                T sum = 0;
                for (int k = 0; k < tensor1.shape[1]; ++k)
                {
                    sum += tensor1.get_by_index({i, k}) * tensor2.get_by_index({k, j});
                }
                result.set_by_index({i, j}, sum);
            }
        }
    }
    else
    {
        throw std::invalid_argument("Tensor device mismatch");
    }

    return result;
}

template <typename T> Tensor<T> add(const Tensor<T> &tensor1, const Tensor<T> &tensor2)
{
    if (tensor1.shape != tensor2.shape)
    {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    Tensor<T> result(tensor1.shape, tensor1.is_cuda);

    if (tensor1.is_cuda && tensor2.is_cuda)
    {
        check_cuda_error(cudaDeviceSynchronize());
        int block_size = 256;
        int grid_size = (tensor1.size + block_size - 1) / block_size;
        kernel::Sadd<<<grid_size, block_size>>>(tensor1.data, tensor2.data, result.data, tensor1.size);
        check_cuda_error(cudaGetLastError());
        check_cuda_error(cudaDeviceSynchronize());
    }
    else if (!tensor1.is_cuda && !tensor2.is_cuda)
    {
        for (int i = 0; i < tensor1.size; ++i)
        {
            result.data[i] = tensor1.data[i] + tensor2.data[i];
        }
    }
    else
    {
        throw std::invalid_argument("Tensor device mismatch");
    }

    return result;
}

template <typename T> Tensor<T> bias_add(Tensor<T> &tensor, const Tensor<T> &bias, bool inplace = false)
{
    if (tensor.shape[1] != bias.shape[1])
    {
        throw std::invalid_argument("Tensor shape mismatch");
    }
    Tensor<T> *result;
    if (inplace)
    {
        result = &tensor;
    }
    else
    {
        result = new Tensor<T>(tensor.shape, tensor.is_cuda);
    }
    if (tensor.is_cuda && bias.is_cuda)
    {
        check_cuda_error(cudaDeviceSynchronize());
        // int block_size = 256;
        int block_size = 256;
        int grid_size = (tensor.size + block_size - 1) / block_size;
        kernel::Sbias_add<<<grid_size, block_size>>>(tensor.data, bias.data, result->data, tensor.size, bias.size);
        check_cuda_error(cudaGetLastError());
        check_cuda_error(cudaDeviceSynchronize());
    }
    else if (!tensor.is_cuda && !bias.is_cuda)
    {
        for (int i = 0; i < tensor.size; ++i)
        {
            result->data[i] = tensor.data[i] + bias.data[i % bias.size];
        }
    }
    else
    {
        throw std::invalid_argument("Tensor device mismatch");
    }
    return *result;
}
template <typename T> Tensor<T> bias_add(Tensor<T> &&tensor, const Tensor<T> &bias, bool inplace = false)
{
    return bias_add(tensor, bias, inplace);
}

template <typename T> Tensor<T> max(Tensor<T> &input, int dim, bool keepdim)
{
    if (dim != 2)
    {
        throw std::invalid_argument("Only dim=2 is supported");
    }
    if (input.shape.size() != 3)
    {
        throw std::invalid_argument("Only 3D tensor is supported");
    }

    std::vector<int> input_shape = input.shape;
    int dim0 = input_shape[0];
    int dim1 = input_shape[1];
    int dim2 = input_shape[2];

    std::vector<int> output_shape = input_shape;
    if (keepdim)
    {
        output_shape[dim] = 1;
    }
    else
    {
        output_shape.erase(output_shape.begin() + dim);
    }

    Tensor<T> output(output_shape, input.is_cuda);

    if (input.is_cuda)
    {
        int block_size = 256;
        int grid_size = (dim0 * dim1 + block_size - 1) / block_size;
        kernel::max_d2<<<grid_size, block_size>>>(input.data, output.data, dim0, dim1, dim2);
        cudaDeviceSynchronize();
    }
    else
    {
        for (int i = 0; i < dim0; ++i)
        {
            for (int j = 0; j < dim1; ++j)
            {
                T max_val = input.data[i * dim1 * dim2 + j * dim2];
                for (int k = 1; k < dim2; ++k)
                {
                    T val = input.data[i * dim1 * dim2 + j * dim2 + k];
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }

                output.data[i * dim1 + j] = max_val;
            }
        }
    }

    return output;
}

template <typename T> Tensor<float> argmax(Tensor<T> &input, int dim, bool keepdim)
{
    if (dim != 2)
    {
        throw std::invalid_argument("Only dim=2 is supported");
    }
    if (input.shape.size() != 3)
    {
        throw std::invalid_argument("Only 3D tensor is supported");
    }

    std::vector<int> input_shape = input.shape;
    int dim0 = input_shape[0];
    int dim1 = input_shape[1];
    int dim2 = input_shape[2];

    std::vector<int> output_shape = input_shape;
    if (keepdim)
    {
        output_shape[dim] = 1;
    }
    else
    {
        output_shape.erase(output_shape.begin() + dim);
    }

    Tensor<float> output(output_shape, input.is_cuda);

    if (input.is_cuda)
    {
        int block_size = 256;
        int grid_size = (dim0 * dim1 + block_size - 1) / block_size;
        kernel::argmax_d2<<<grid_size, block_size>>>(input.data, output.data, dim0, dim1, dim2);
        cudaDeviceSynchronize();
    }
    else
    {
        for (int i = 0; i < dim0; ++i)
        {
            for (int j = 0; j < dim1; ++j)
            {
                int max_idx = 0;
                T max_val = input.data[i * dim1 * dim2 + j * dim2];
                for (int k = 1; k < dim2; ++k)
                {
                    T val = input.data[i * dim1 * dim2 + j * dim2 + k];
                    if (val > max_val)
                    {
                        max_val = val;
                        max_idx = k;
                    }
                }

                output.data[i * dim1 + j] = max_idx;
            }
        }
    }

    return output;
}

template <typename T> Tensor<T> Identity(int n, bool is_cuda = true)
{
    Tensor<T> result({n, n}, is_cuda);
    if (is_cuda)
    {
        check_cuda_error(cudaDeviceSynchronize());
        int block_size = 256;
        int grid_size = (n * n + block_size - 1) / block_size;
        kernel::Sidentity<<<grid_size, block_size>>>(result.data, n);
        check_cuda_error(cudaGetLastError());
        check_cuda_error(cudaDeviceSynchronize());
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            result.data[i * n + i] = 1;
        }
    }
    return result;
}

template <typename T> Tensor<T> bmm(const Tensor<T> &tensor1, const Tensor<T> &tensor2)
{
    if (tensor1.shape[2] != tensor2.shape[1])
    {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    Tensor<T> result({tensor1.shape[0], tensor1.shape[1], tensor2.shape[2]}, tensor1.is_cuda);

    if (tensor1.is_cuda && tensor2.is_cuda)
    {
        check_cuda_error(cudaDeviceSynchronize());
        dim3 block_size(16, 16);
        dim3 grid_size(CEIL_DIV(tensor1.shape[1], block_size.x), CEIL_DIV(tensor2.shape[2], block_size.y),
                       tensor1.shape[0]);
        kernel::Sbmm<<<grid_size, block_size>>>(tensor1.data, tensor2.data, result.data, tensor1.shape[1],
                                                tensor2.shape[2], tensor1.shape[2], tensor1.shape[0]);
        check_cuda_error(cudaGetLastError());
        check_cuda_error(cudaDeviceSynchronize());
    }
    else if (!tensor1.is_cuda && !tensor2.is_cuda)
    {
        for (int i = 0; i < tensor1.shape[0]; ++i)
        {
            for (int j = 0; j < tensor1.shape[1]; ++j)
            {
                for (int k = 0; k < tensor2.shape[2]; ++k)
                {
                    T sum = 0;
                    for (int l = 0; l < tensor1.shape[2]; ++l)
                    {
                        sum += tensor1.get_by_index({i, j, l}) * tensor2.get_by_index({j, l, k});
                    }
                    result.set_by_index({i, j, k}, sum);
                }
            }
        }
    }
    else
    {
        throw std::invalid_argument("Tensor device mismatch");
    }

    return result;
}

} // namespace cu

namespace nn
{
template <typename T> class Module
{
  public:
    virtual cu::Tensor<T> forward(cu::Tensor<T> &input) = 0;

    cu::Tensor<T> operator()(cu::Tensor<T> &input)
    {
        return forward(input);
    }

    cu::Tensor<T> operator()(cu::Tensor<T> &&input)
    {
        return forward(input);
    }

    virtual void load_state_dict(std::map<std::string, std::vector<T>> &state_dict)
    {
    }

    virtual void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict)
    {
        load_state_dict(state_dict);
    }
};

template <typename T> class Conv1d : public Module<T>
{
  public:
    cu::Tensor<T> weight;
    cu::Tensor<T> bias;

    Conv1d() = default;

    Conv1d(int in_channels, int out_channels, int kernel_size, bool is_cuda = true)
    {
        if (kernel_size != 1)
        {
            throw std::invalid_argument("Not implemented for kernal size " + std::to_string(kernel_size));
        }
        weight = cu::Tensor<T>({out_channels, in_channels}, is_cuda);
        bias = cu::Tensor<T>({1, out_channels}, is_cuda);
    }

    void load_state(std::vector<T> &weight_data, std::vector<T> &bias_data)
    {
        weight.load(weight_data);
        bias.load(bias_data);
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        load_state(state_dict["weight"], state_dict["bias"]);
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &input) override
    {
        // weight * input + bias
        if (input.shape[input.shape.size() - 2] != weight.shape[1])
        {
            throw std::invalid_argument("Tensor shape mismatch");
        }
        if (input.is_cuda != weight.is_cuda || input.is_cuda != bias.is_cuda)
        {
            throw std::invalid_argument("Tensor device mismatch");
        }
        cu::Tensor<T> result;
        if (input.shape.size() == 2)
        {
            result = cu::Tensor<T>({weight.shape[0], input.shape[1]}, input.is_cuda);
            if (input.is_cuda)
            {
                // check_cuda_error(cudaDeviceSynchronize());
                if (std::is_same<T, float>::value)
                {
                    dim3 blockDim(16, 16);
                    dim3 gridDim(CEIL_DIV(input.shape[1], blockDim.x), CEIL_DIV(weight.shape[0], blockDim.y));
                    cu::kernel::Sconv1d_k1_d2<<<gridDim, blockDim>>>(weight.data, input.data, bias.data, result.data,
                                                                     weight.shape[0], input.shape[1], input.shape[0]);
                }
                else if (std::is_same<T, half>::value)
                {
                    // not implemented
                }
                else
                {
                    throw std::invalid_argument("Only float and half are supported");
                }
                check_cuda_error(cudaGetLastError());
                // check_cuda_error(cudaDeviceSynchronize());
                check_cuda_error(cudaStreamSynchronize(0));
            }
            else
            {
                for (int i = 0; i < weight.shape[0]; ++i)
                {
                    for (int j = 0; j < input.shape[1]; ++j)
                    {
                        T sum = 0;
                        for (int k = 0; k < input.shape[0]; ++k)
                        {
                            sum += input.get_by_index({k, j}) * weight.get_by_index({i, k});
                        }
                        result.set_by_index({i, j}, sum + bias.get_by_index({0, i}));
                    }
                }
            }
        }
        else if (input.shape.size() == 3)
        {
            result = cu::Tensor<T>({input.shape[0], weight.shape[0], input.shape[2]}, input.is_cuda);
            if (input.is_cuda)
            {
                // check_cuda_error(cudaDeviceSynchronize());
                if (std::is_same<T, float>::value)
                {
                    dim3 blockDim(16, 16);
                    dim3 gridDim(CEIL_DIV(input.shape[2], blockDim.x), CEIL_DIV(weight.shape[0], blockDim.y),
                                 input.shape[0]);
                    cu::kernel::Sconv1d_k1_d3<<<gridDim, blockDim>>>(weight.data, input.data, bias.data, result.data,
                                                                     weight.shape[0], input.shape[2], input.shape[1],
                                                                     input.shape[0]);
                }
                else if (std::is_same<T, half>::value)
                {
                    // not implemented
                }
                else
                {
                    throw std::invalid_argument("Only float and half are supported");
                }
                check_cuda_error(cudaGetLastError());
                // check_cuda_error(cudaDeviceSynchronize());
                check_cuda_error(cudaStreamSynchronize(0));
            }
            else
            {
                for (int i = 0; i < input.shape[0]; ++i)
                {
                    for (int j = 0; j < weight.shape[0]; ++j)
                    {
                        for (int k = 0; k < input.shape[2]; ++k)
                        {
                            T sum = 0;
                            for (int l = 0; l < input.shape[1]; ++l)
                            {
                                sum += input.get_by_index({i, l, k}) * weight.get_by_index({j, l});
                            }
                            result.set_by_index({i, j, k}, sum + bias.get_by_index({0, j}));
                        }
                    }
                }
            }
        }
        else
        {
            throw std::invalid_argument("Not implemented for input shape " + std::to_string(input.shape.size()));
        }
        return result;
    }
};

template <typename T> class Linear : public Module<T>
{
  public:
    cu::Tensor<T> weight;
    cu::Tensor<T> bias;

    Linear() = default;

    Linear(int in_features, int out_features, bool is_cuda = true)
    {
        weight = cu::Tensor<T>({in_features, out_features}, is_cuda);
        bias = cu::Tensor<T>({1, out_features}, is_cuda);
    }

    void load_state(std::vector<T> &weight_data, std::vector<T> &bias_data, bool transpose_weight = true)
    {
        if (transpose_weight)
        {
            weight = weight.transpose(true);
        }
        weight.load(weight_data);
        bias.load(bias_data);
        if (transpose_weight)
        {
            weight = weight.transpose();
        }
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        load_state(state_dict["weight"], state_dict["bias"], true);
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &input) override
    {
        if (input.shape[1] != weight.shape[0])
        {
            throw std::invalid_argument("Tensor shape mismatch");
        }
        if (input.is_cuda != weight.is_cuda || input.is_cuda != bias.is_cuda)
        {
            throw std::invalid_argument("Tensor device mismatch");
        }
        cu::Tensor<T> result({input.shape[0], weight.shape[1]}, input.is_cuda);
        if (input.is_cuda)
        {
            // check_cuda_error(cudaDeviceSynchronize());
            if (std::is_same<T, float>::value)
            {
                dim3 blockDim(16, 16);
                dim3 gridDim(CEIL_DIV(weight.shape[1], blockDim.x), CEIL_DIV(input.shape[0], blockDim.y));
                cu::kernel::linear<<<gridDim, blockDim>>>(input.data, weight.data, bias.data, result.data,
                                                          input.shape[0], weight.shape[1], input.shape[1]);
            }
            else if (std::is_same<T, half>::value)
            {
                dim3 blockDim(16, 16);
                dim3 gridDim(CEIL_DIV(weight.shape[1] / 2, blockDim.x), CEIL_DIV(input.shape[0], blockDim.y));
                cu::kernel::linear<<<gridDim, blockDim>>>(input.data, weight.data, bias.data, result.data,
                                                          input.shape[0], weight.shape[1], input.shape[1]);
            }
            else
            {
                throw std::invalid_argument("Only float and half are supported");
            }
            check_cuda_error(cudaGetLastError());
            // check_cuda_error(cudaDeviceSynchronize());
            check_cuda_error(cudaStreamSynchronize(0));
        }
        else
        {
            for (int i = 0; i < input.shape[0]; ++i)
            {
                for (int j = 0; j < weight.shape[1]; ++j)
                {
                    T sum = 0;
                    for (int k = 0; k < input.shape[1]; ++k)
                    {
                        sum += input.get_by_index({i, k}) * weight.get_by_index({k, j});
                    }
                    result.set_by_index({i, j}, sum + bias.get_by_index({0, j}));
                }
            }
        }
        return result;
    }
};

template <typename T> class ReLU : public Module<T>
{
  public:
    bool inplace;
    ReLU(bool inplace = true) : inplace(inplace)
    {
    }

    cu::Tensor<T> forward(cu::Tensor<T> &input) override
    {
        cu::Tensor<T> *result;
        if (inplace)
        {
            result = &input;
        }
        else
        {
            result = new cu::Tensor<T>(input.shape, input.is_cuda);
        }
        if (input.is_cuda)
        {
            // check_cuda_error(cudaDeviceSynchronize());
            if (std::is_same<T, float>::value)
            {
                int block_size = 256;
                int grid_size = (input.size + block_size - 1) / block_size;
                cu::kernel::Srelu<<<grid_size, block_size>>>(input.data, result->data, result->size);
            }
            else if (std::is_same<T, half>::value)
            {
                // not implemented
            }
            else
            {
                throw std::invalid_argument("Only float and half are supported");
            }
            check_cuda_error(cudaGetLastError());
            // check_cuda_error(cudaDeviceSynchronize());
            check_cuda_error(cudaStreamSynchronize(0));
        }
        else
        {
            for (int i = 0; i < input.size; ++i)
            {
                result->data[i] = input.data[i] > 0 ? input.data[i] : 0;
            }
        }
        if (inplace)
        {
            return *result;
        }
        else
        {
            cu::Tensor<T> new_result = std::move(*result);
            return new_result;
        }
    }
};

template <typename T> class BatchNorm1d : public Module<T>
{
  public:
    cu::Tensor<T> weight;
    cu::Tensor<T> bias;
    cu::Tensor<T> running_mean;
    cu::Tensor<T> running_var;
    float eps;
    bool inplace;

    BatchNorm1d() = default;

    BatchNorm1d(int num_features, bool is_cuda = true) : BatchNorm1d(num_features, 1e-5, 0.1, is_cuda, true)
    {
    }

    BatchNorm1d(int num_features, float eps = 1e-5, float momentum = 0.1, bool is_cuda = true, bool inplace = true)
        : eps(eps), inplace(inplace)
    {
        weight = cu::Tensor<T>({1, num_features}, is_cuda);
        bias = cu::Tensor<T>({1, num_features}, is_cuda);
        running_mean = cu::Tensor<T>({1, num_features}, is_cuda);
        running_var = cu::Tensor<T>({1, num_features}, is_cuda);
    }

    void load_state(std::vector<T> &weight_data, std::vector<T> &bias_data, std::vector<T> &running_mean_data,
                    std::vector<T> &running_var_data)
    {
        weight.load(weight_data);
        bias.load(bias_data);
        running_mean.load(running_mean_data);
        running_var.load(running_var_data);
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        load_state(state_dict["weight"], state_dict["bias"], state_dict["running_mean"], state_dict["running_var"]);
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &input) override
    {
        if (input.shape[1] != weight.shape[1])
        {
            throw std::invalid_argument("Tensor shape mismatch");
        }
        if (input.is_cuda != weight.is_cuda || input.is_cuda != bias.is_cuda)
        {
            throw std::invalid_argument("Tensor device mismatch");
        }
        cu::Tensor<T> *result;
        if (inplace)
        {
            result = &input;
        }
        else
        {
            result = new cu::Tensor<T>(input, input.is_cuda);
        }

        if (input.shape.size() == 2)
        {
            int feature_size = input.shape[1];
            int length = input.shape[0];
            if (input.is_cuda)
            {
                // check_cuda_error(cudaDeviceSynchronize());
                if (std::is_same<T, float>::value)
                {
                    dim3 block_size(16, 16);
                    dim3 grid_size(CEIL_DIV(feature_size, block_size.x), CEIL_DIV(length, block_size.y));
                    cu::kernel::Sbatchnorm1d_d2<<<grid_size, block_size>>>(input.data, result->data, running_mean.data,
                                                                           running_var.data, weight.data, bias.data,
                                                                           eps, length, feature_size);
                }
                else if (std::is_same<T, half>::value)
                {
                    // not implemented
                }
                else
                {
                    throw std::invalid_argument("Only float and half are supported");
                }

                check_cuda_error(cudaGetLastError());
                // check_cuda_error(cudaDeviceSynchronize());
                check_cuda_error(cudaStreamSynchronize(0));
            }
            else
            {
                for (int i = 0; i < feature_size; ++i)
                {
                    T mean = running_mean.get_by_index({0, i});
                    T var = running_var.get_by_index({0, i});
                    T scale = weight.get_by_index({0, i});
                    T offset = bias.get_by_index({0, i});
                    for (int j = 0; j < length; ++j)
                    {
                        T x = input.get_by_index({j, i});
                        result->set_by_index({j, i}, (x - mean) / sqrt(var + eps) * scale + offset);
                    }
                }
            }
        }
        else if (input.shape.size() == 3)
        {
            int batch_size = input.shape[0];
            int feature_size = input.shape[2];
            int channel_size = input.shape[1];
            if (input.is_cuda)
            {
                // check_cuda_error(cudaDeviceSynchronize());
                if (std::is_same<T, float>::value)
                {
                    int block_size = 256;
                    int grid_size = (input.size + block_size - 1) / block_size;
                    cu::kernel::Sbatchnorm1d_d3<<<grid_size, block_size>>>(input.data, result->data, running_mean.data,
                                                                           running_var.data, weight.data, bias.data,
                                                                           eps, channel_size, feature_size, input.size);
                }
                else if (std::is_same<T, half>::value)
                {
                    // not implemented
                }
                else
                {
                    throw std::invalid_argument("Only float and half are supported");
                }

                check_cuda_error(cudaGetLastError());
                // check_cuda_error(cudaDeviceSynchronize());
                check_cuda_error(cudaStreamSynchronize(0));
            }
            else
            {
                for (int j = 0; j < channel_size; ++j)
                {
                    T mean = running_mean.get_by_index({0, j});
                    T var = running_var.get_by_index({0, j});
                    T scale = weight.get_by_index({0, j});
                    T offset = bias.get_by_index({0, j});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        for (int k = 0; k < feature_size; ++k)
                        {
                            T x = input.get_by_index({i, j, k});
                            result->set_by_index({i, j, k}, (x - mean) / sqrt(var + eps) * scale + offset);
                        }
                    }
                }
            }
        }
        else
        {
            throw std::invalid_argument("BatchNorm1d only support 2D or 3D tensor");
        }

        if (inplace)
        {
            return *result;
        }
        else
        {
            cu::Tensor<T> new_result = std::move(*result);
            return new_result;
        }
    }
};

} // namespace nn

namespace misc
{
float calc_accuracy(cu::Tensor<float> &output, cu::Tensor<float> &target)
{
    if (output.shape != target.shape)
    {
        throw std::invalid_argument("Tensor shape mismatch");
    }
    if (output.is_cuda != target.is_cuda)
    {
        throw std::invalid_argument("Tensor device mismatch");
    }
    int correct = 0;
    if (output.is_cuda)
    {
        int *d_correct;
        cudaMallocAsync(&d_correct, sizeof(int), 0);
        cudaMemset(d_correct, 0, sizeof(int));

        cudaDeviceSynchronize();
        int block_size = 256;
        int grid_size = (output.size + block_size - 1) / block_size;
        cu::kernel::calc_accuracy_kernel<<<grid_size, block_size>>>(output.data, target.data, output.size, d_correct);
        cudaDeviceSynchronize();

        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFreeAsync(d_correct, 0);
    }
    else
    {
        for (int i = 0; i < output.size; ++i)
        {
            if (output.data[i] == target.data[i])
            {
                correct++;
            }
        }
    }
    return static_cast<float>(correct) / output.size;
}

std::vector<float> collate(std::vector<float> &points, int target_num)
{
    //  input: points = [x1, y1, z1, x2, y2, z2, ...], num=points.size()/3
    // process: random drop points to make the number of points to be 1024
    // output: points = [x1, y1, z1, x2, y2, z2, ...], num=1024
    if (points.size() % 3 != 0)
    {
        throw std::invalid_argument("Invalid points size");
    }
    int num = points.size() / 3;
    // int target_num = 1024;

    std::vector<float> result;
    result.reserve(target_num * 3);

    if (num > target_num)
    {
        // 随机选择 1024 个点
        std::vector<int> indices(num);
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        indices.resize(target_num);

        for (int i : indices)
        {
            result.push_back(points[i * 3]);
            result.push_back(points[i * 3 + 1]);
            result.push_back(points[i * 3 + 2]);
        }
    }
    else
    {
        // 复制所有点
        result.insert(result.end(), points.begin(), points.end());

        // 随机选择点进行补全
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<> dis(0, num - 1);

        while (result.size() < target_num * 3)
        {
            int i = dis(g);
            result.push_back(points[i * 3]);
            result.push_back(points[i * 3 + 1]);
            result.push_back(points[i * 3 + 2]);
        }
    }

    return result;
}

std::vector<half> to_half(std::vector<float> &data)
{
    std::vector<half> result(data.size());
    for (int i = 0; i < data.size(); ++i)
    {
        result[i] = static_cast<half>(data[i]);
    }
    return result;
}
std::vector<half> to_half(std::vector<float> &&data)
{
    return to_half(data);
}

} // namespace misc

namespace PointNet
{
template <typename T> class STN3d : public nn::Module<T>
{
  public:
    nn::Conv1d<T> conv1;
    nn::Conv1d<T> conv2;
    nn::Conv1d<T> conv3;
    nn::Linear<T> fc1;
    nn::Linear<T> fc2;
    nn::Linear<T> fc3;
    nn::ReLU<T> relu;
    nn::BatchNorm1d<T> bn1;
    nn::BatchNorm1d<T> bn2;
    nn::BatchNorm1d<T> bn3;
    nn::BatchNorm1d<T> bn4;
    nn::BatchNorm1d<T> bn5;
    cu::Tensor<T> iden;

    STN3d() = default;

    STN3d(int channel, bool is_cuda = true)
        : conv1(channel, 64, 1, is_cuda), conv2(64, 128, 1, is_cuda), conv3(128, 1024, 1, is_cuda),
          fc1(1024, 512, is_cuda), fc2(512, 256, is_cuda), fc3(256, 9, is_cuda), relu(true), bn1(64, is_cuda),
          bn2(128, is_cuda), bn3(1024, is_cuda), bn4(512, is_cuda), bn5(256, is_cuda)
    {
        iden = cu::Tensor<T>({1, 9}, is_cuda);
        iden.load(std::vector<T>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        std::vector<std::pair<std::string, nn::Module<T> *>> modules = {
            {"conv1", &conv1}, {"conv2", &conv2}, {"conv3", &conv3}, {"fc1", &fc1}, {"fc2", &fc2}, {"fc3", &fc3},
            {"bn1", &bn1},     {"bn2", &bn2},     {"bn3", &bn3},     {"bn4", &bn4}, {"bn5", &bn5}};
        for (auto &m : modules)
        {
            m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
        }
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &x) override
    {
        int batch_size = x.shape[0];
        x = relu(bn1(conv1(x)));
        x = relu(bn2(conv2(x)));
        x = relu(bn3(conv3(x)));
        x = cu::max(x, 2, true);
        x.view({-1, 1024});
        x = relu(bn4(fc1(x)));
        x = relu(bn5(fc2(x)));
        x = fc3(x);
        // iden = iden.repeat({batch_size, 1});
        // x = x + iden;
        x = x + iden.repeat({batch_size, 1});
        x.view({-1, 3, 3});
        return x;
    }
};

template <typename T> class STNkd : public nn::Module<T>
{
  public:
    nn::Conv1d<T> conv1;
    nn::Conv1d<T> conv2;
    nn::Conv1d<T> conv3;
    nn::Linear<T> fc1;
    nn::Linear<T> fc2;
    nn::Linear<T> fc3;
    nn::ReLU<T> relu;
    nn::BatchNorm1d<T> bn1;
    nn::BatchNorm1d<T> bn2;
    nn::BatchNorm1d<T> bn3;
    nn::BatchNorm1d<T> bn4;
    nn::BatchNorm1d<T> bn5;
    int k;
    cu::Tensor<T> iden;

    STNkd() = default;

    STNkd(int k, bool is_cuda = true)
        : conv1(k, 64, 1, is_cuda), conv2(64, 128, 1, is_cuda), conv3(128, 1024, 1, is_cuda), fc1(1024, 512, is_cuda),
          fc2(512, 256, is_cuda), fc3(256, k * k, is_cuda), relu(true), bn1(64, is_cuda), bn2(128, is_cuda),
          bn3(1024, is_cuda), bn4(512, is_cuda), bn5(256, is_cuda), k(k)
    {
        iden = cu::Identity<T>(k, is_cuda);
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        std::vector<std::pair<std::string, nn::Module<T> *>> modules = {
            {"conv1", &conv1}, {"conv2", &conv2}, {"conv3", &conv3}, {"fc1", &fc1}, {"fc2", &fc2}, {"fc3", &fc3},
            {"bn1", &bn1},     {"bn2", &bn2},     {"bn3", &bn3},     {"bn4", &bn4}, {"bn5", &bn5}};
        for (auto &m : modules)
        {
            m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
        }
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &x) override
    {
        int batch_size = x.shape[0];
        x = relu(bn1(conv1(x)));
        x = relu(bn2(conv2(x)));
        x = relu(bn3(conv3(x)));
        x = cu::max(x, 2, true);
        x.view({-1, 1024});
        x = relu(bn4(fc1(x)));
        x = relu(bn5(fc2(x)));
        x = fc3(x);
        iden.view({1, k * k});
        x = x + iden.repeat({batch_size, 1});
        x.view({-1, k, k});
        return x;
    }
};

template <typename T> class PointNetEncoder : public nn::Module<T>
{
  public:
    STN3d<T> stn;
    nn::Conv1d<T> conv1;
    nn::Conv1d<T> conv2;
    nn::Conv1d<T> conv3;
    nn::BatchNorm1d<T> bn1;
    nn::BatchNorm1d<T> bn2;
    nn::BatchNorm1d<T> bn3;
    STNkd<T> fstn;
    nn::ReLU<T> relu;
    bool global_feat;
    bool feature_transform;

    PointNetEncoder() = default;

    PointNetEncoder(bool global_feat = true, bool feature_transform = false, int channel = 3, bool is_cuda = true)
        : stn(channel, is_cuda), conv1(channel, 64, 1, is_cuda), conv2(64, 128, 1, is_cuda),
          conv3(128, 1024, 1, is_cuda), bn1(64, is_cuda), bn2(128, is_cuda), bn3(1024, is_cuda),
          global_feat(global_feat), feature_transform(feature_transform)
    {
        if (feature_transform)
        {
            fstn = STNkd<T>(64, is_cuda);
        }
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        std::vector<std::pair<std::string, nn::Module<T> *>> modules = {
            {"stn", &stn}, {"conv1", &conv1}, {"conv2", &conv2}, {"conv3", &conv3},
            {"bn1", &bn1}, {"bn2", &bn2},     {"bn3", &bn3},     {"fstn", &fstn}};
        for (auto &m : modules)
        {
            m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
        }
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &x) override
    {
        int B = x.shape[0];
        int D = x.shape[1];
        int N = x.shape[2];
        auto temp_x = x;
        auto trans = stn(temp_x);
        x = x.transpose();
        x = cu::bmm(x, trans);
        x = x.transpose();
        x = relu(bn1(conv1(x)));

        if (feature_transform)
        {
            temp_x = x;
            auto trans_feat = fstn(temp_x);
            x = x.transpose();
            x = cu::bmm(x, trans_feat);
            x = x.transpose();
        }
        x = relu(bn2(conv2(x)));
        x = bn3(conv3(x));
        x = cu::max(x, 2, true);
        x.view({-1, 1024});
        return x;
    }
};

template <typename T> class PointNetClassifier : public nn::Module<T>
{
  public:
    PointNetEncoder<T> feat;
    nn::Linear<T> fc1;
    nn::Linear<T> fc2;
    nn::Linear<T> fc3;
    // nn::Dropout dropout;
    nn::BatchNorm1d<T> bn1;
    nn::BatchNorm1d<T> bn2;
    nn::ReLU<T> relu;
    int k;

    PointNetClassifier() = default;

    PointNetClassifier(int k = 10, bool normal_channel = false, bool is_cuda = true)
        : feat(true, true, normal_channel ? 6 : 3, is_cuda), fc1(1024, 512, is_cuda), fc2(512, 256, is_cuda),
          fc3(256, k, is_cuda), bn1(512, is_cuda), bn2(256, is_cuda), relu(true), k(k)
    {
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &state_dict) override
    {
        std::vector<std::pair<std::string, nn::Module<T> *>> modules = {{"feat", &feat}, {"fc1", &fc1}, {"fc2", &fc2},
                                                                        {"fc3", &fc3},   {"bn1", &bn1}, {"bn2", &bn2}};
        for (auto &m : modules)
        {
            m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
        }
    }

    void load_state_dict(std::map<std::string, std::vector<T>> &&state_dict) override
    {
        load_state_dict(state_dict);
    }

    cu::Tensor<T> forward(cu::Tensor<T> &x) override
    {
        x = feat(x);
        x = relu(bn1(fc1(x)));
        x = relu(bn2(fc2(x)));
        x = fc3(x);
        x.view({1, -1, k});
        x = cu::argmax(x, 2, true);
        x.view({-1});
        return x;
    }
};
} // namespace PointNet

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string &dir)
{
    std::vector<std::string> files;
    DIR *dp;
    struct dirent *entry;
    if ((dp = opendir(dir.c_str())) != NULL)
    {
        while ((entry = readdir(dp)) != NULL)
        {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos)
            {
                files.push_back(filename);
            }
        }
        closedir(dp);
    }
    else
    {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string &filepath)
{
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open())
    {
        float value;
        while (file >> value)
        {
            data.push_back(value);
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir)
{
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto &file : param_files)
    {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string &file_path, std::vector<std::vector<float>> &list_of_points,
                  std::vector<int> &list_of_labels)
{
    try
    {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++)
        {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto &name : dataset_names)
        {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    }
    catch (FileIException &error)
    {
        error.printErrorStack();
    }
    catch (DataSetIException &error)
    {
        error.printErrorStack();
    }
    catch (DataSpaceIException &error)
    {
        error.printErrorStack();
    }
    catch (DataTypeIException &error)
    {
        error.printErrorStack();
    }
}

int main(int argc, char *argv[])
{
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    cudaFree(0);

    std::string dir = argv[1];
    // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签

    std::string file_path = "./data/test_point_clouds.h5";

    bool is_cuda = true;
    int pc_num = 1024;
    // 读取模型参数
    auto params = read_params(dir);
    // 读取训练集数据
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    std::vector<float> x_data;
    read_h5_file(file_path, list_of_points, list_of_labels);
    std::vector<float> label_data(list_of_labels.begin(), list_of_labels.end());
    for (auto &points : list_of_points)
    {
        auto temp_points = misc::collate(points, pc_num);
        x_data.insert(x_data.end(), temp_points.begin(), temp_points.end());
    }
    cu::Tensor<float> x({static_cast<int>(list_of_points.size()), pc_num, 3}, is_cuda);
    cu::Tensor<float> y({static_cast<int>(label_data.size())}, false);
    x.load(x_data);
    y.load(label_data);
    x = x.transpose();

    // 加载模型
    auto model = PointNet::PointNetClassifier<float>(10, false, is_cuda);
    model.load_state_dict(params);

    // warm up
    model(cu::Tensor<float>(x));

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    auto output = model(x);
    output.to_cpu();
    auto accuracy = misc::calc_accuracy(output, y);
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;

    return 0;
}
