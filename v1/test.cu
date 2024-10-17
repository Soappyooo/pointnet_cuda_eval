#include <algorithm>
#include <chrono>
#include <cstring>
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
        // print sub dict keys
        // std::cout << std::endl << "prefix: " << prefix << std::endl;
        // for (auto &item : sub_dict)
        // {
        //     std::cout << item.first << " ";
        // }
        // std::cout << std::endl;
        return sub_dict;
    }

} // namespace misc

namespace cu
{
    namespace kernel
    {

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

        __global__ void Slinear(const float *input, const float *weight, const float *bias, float *output, int m, int n, int k)
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

        template <typename T>
        __global__ void max_d2(const T *input, T *output, int dim0, int dim1, int dim2)
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

        template <typename T>
        __global__ void argmax_d2(const T *input, T *output, int dim0, int dim1, int dim2)
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

    template <typename T>
    class Tensor
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
            // only support transpose(2,1) for 3d tensor and transpose(1,0) for 2d tensor
            // if (shape.size() != 2)
            // {
            //     throw std::invalid_argument("Transpose only support 2D tensor");
            // }
            // std::swap(shape[0], shape[1]);
            // if (transpose_shape_only)
            // {
            //     return;
            // }
            // if (is_cuda)
            // {
            //     T *temp_data;
            //     check_cuda_error(cudaMallocAsync(&temp_data, size * sizeof(T), 0));
            //     check_cuda_error(cudaDeviceSynchronize());
            //     int block_size = 256;
            //     int grid_size = (size + block_size - 1) / block_size;
            //     kernel::Stranspose<<<grid_size, block_size>>>(data, temp_data, shape[0], shape[1], size);
            //     check_cuda_error(cudaGetLastError());
            //     check_cuda_error(cudaDeviceSynchronize());
            //     check_cuda_error(cudaFreeAsync(data, 0));
            //     data = temp_data;
            // }
            // else
            // {
            //     T *temp_data = new T[size];
            //     for (int i = 0; i < shape[0]; ++i)
            //     {
            //         for (int j = 0; j < shape[1]; ++j)
            //         {
            //             temp_data[i * shape[1] + j] = data[j * shape[0] + i];
            //         }
            //     }
            //     delete[] data;
            //     data = temp_data;
            // }
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
                        output += (i > 0 ? ", " : "");
#ifdef Cpp20
                        output += std::format("{:9.4f}", value);
#else
                        output += std::to_string(value);
#endif
                    }
                    else
                    {
                        output += (i > 0 ? ", " : "");
#ifdef Cpp20
                        output += std::format("{:9.4f}", data[offset + i]);
#else
                        output += std::to_string(data[offset + i]);
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

    template <typename T>
    Tensor<T> gemm(const Tensor<T> &tensor1, const Tensor<T> &tensor2)
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

    template <typename T>
    Tensor<T> add(const Tensor<T> &tensor1, const Tensor<T> &tensor2)
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

    template <typename T>
    Tensor<T> bias_add(Tensor<T> &tensor, const Tensor<T> &bias, bool inplace = false)
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
    template <typename T>
    Tensor<T> bias_add(Tensor<T> &&tensor, const Tensor<T> &bias, bool inplace = false)
    {
        return bias_add(tensor, bias, inplace);
    }

    template <typename T>
    Tensor<T> max(Tensor<T> &input, int dim, bool keepdim)
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

    template <typename T>
    Tensor<float> argmax(Tensor<T> &input, int dim, bool keepdim)
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

    template <typename T>
    Tensor<T> Identity(int n, bool is_cuda = true)
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

    template <typename T>
    Tensor<T> bmm(const Tensor<T> &tensor1, const Tensor<T> &tensor2)
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

    class Module
    {
    public:
        virtual cu::Tensor<float> forward(cu::Tensor<float> &input) = 0;

        cu::Tensor<float> operator()(cu::Tensor<float> &input)
        {
            return forward(input);
        }

        cu::Tensor<float> operator()(cu::Tensor<float> &&input)
        {
            return forward(input);
        }

        virtual void load_state_dict(std::map<std::string, std::vector<float>> &state_dict)
        {
        }

        virtual void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict)
        {
            load_state_dict(state_dict);
        }
    };

    class Conv1d : public Module
    {
    public:
        cu::Tensor<float> weight;
        cu::Tensor<float> bias;

        Conv1d() = default;

        Conv1d(int in_channels, int out_channels, int kernel_size, bool is_cuda = true)
        {
            if (kernel_size != 1)
            {
                throw std::invalid_argument("Not implemented for kernal size " + std::to_string(kernel_size));
            }
            weight = cu::Tensor<float>({out_channels, in_channels}, is_cuda);
            bias = cu::Tensor<float>({1, out_channels}, is_cuda);
        }

        void load_state(std::vector<float> &weight_data, std::vector<float> &bias_data)
        {
            weight.load(weight_data);
            bias.load(bias_data);
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            load_state(state_dict["weight"], state_dict["bias"]);
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &input) override
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
            cu::Tensor<float> result;
            if (input.shape.size() == 2)
            {
                result = cu::Tensor<float>({weight.shape[0], input.shape[1]}, input.is_cuda);
                if (input.is_cuda)
                {
                    check_cuda_error(cudaDeviceSynchronize());
                    dim3 blockDim(16, 16);
                    dim3 gridDim(CEIL_DIV(input.shape[1], blockDim.x), CEIL_DIV(weight.shape[0], blockDim.y));
                    cu::kernel::Sconv1d_k1_d2<<<gridDim, blockDim>>>(weight.data, input.data, bias.data, result.data,
                                                                     weight.shape[0], input.shape[1], input.shape[0]);
                    check_cuda_error(cudaGetLastError());
                    check_cuda_error(cudaDeviceSynchronize());
                }
                else
                {
                    for (int i = 0; i < weight.shape[0]; ++i)
                    {
                        for (int j = 0; j < input.shape[1]; ++j)
                        {
                            float sum = 0;
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
                result = cu::Tensor<float>({input.shape[0], weight.shape[0], input.shape[2]}, input.is_cuda);
                if (input.is_cuda)
                {
                    check_cuda_error(cudaDeviceSynchronize());
                    dim3 blockDim(16, 16);
                    dim3 gridDim(CEIL_DIV(input.shape[2], blockDim.x), CEIL_DIV(weight.shape[0], blockDim.y),
                                 input.shape[0]);
                    cu::kernel::Sconv1d_k1_d3<<<gridDim, blockDim>>>(weight.data, input.data, bias.data, result.data,
                                                                     weight.shape[0], input.shape[2], input.shape[1],
                                                                     input.shape[0]);
                    check_cuda_error(cudaGetLastError());
                    check_cuda_error(cudaDeviceSynchronize());
                }
                else
                {
                    for (int i = 0; i < input.shape[0]; ++i)
                    {
                        for (int j = 0; j < weight.shape[0]; ++j)
                        {
                            for (int k = 0; k < input.shape[2]; ++k)
                            {
                                float sum = 0;
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

    class Linear : public Module
    {
    public:
        cu::Tensor<float> weight;
        cu::Tensor<float> bias;

        Linear() = default;

        Linear(int in_features, int out_features, bool is_cuda = true)
        {
            weight = cu::Tensor<float>({in_features, out_features}, is_cuda);
            bias = cu::Tensor<float>({1, out_features}, is_cuda);
        }

        void load_state(std::vector<float> &weight_data, std::vector<float> &bias_data, bool transpose_weight = true)
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

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            load_state(state_dict["weight"], state_dict["bias"], true);
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &input) override
        {
            if (input.shape[1] != weight.shape[0])
            {
                throw std::invalid_argument("Tensor shape mismatch");
            }
            if (input.is_cuda != weight.is_cuda || input.is_cuda != bias.is_cuda)
            {
                throw std::invalid_argument("Tensor device mismatch");
            }
            cu::Tensor<float> result({input.shape[0], weight.shape[1]}, input.is_cuda);
            if (input.is_cuda)
            {
                // return bias_add(input * weight, bias, inline_bias_add);
                check_cuda_error(cudaDeviceSynchronize());
                dim3 blockDim(16, 16);
                dim3 gridDim(CEIL_DIV(weight.shape[1], blockDim.x), CEIL_DIV(input.shape[0], blockDim.y));
                cu::kernel::Slinear<<<gridDim, blockDim>>>(input.data, weight.data, bias.data, result.data, input.shape[0],
                                                           weight.shape[1], input.shape[1]);
                check_cuda_error(cudaGetLastError());
                check_cuda_error(cudaDeviceSynchronize());
            }
            else
            {
                for (int i = 0; i < input.shape[0]; ++i)
                {
                    for (int j = 0; j < weight.shape[1]; ++j)
                    {
                        float sum = 0;
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

    class ReLU : public Module
    {
    public:
        bool inplace;
        ReLU(bool inplace = true) : inplace(inplace)
        {
        }

        cu::Tensor<float> forward(cu::Tensor<float> &input) override
        {
            cu::Tensor<float> *result;
            if (inplace)
            {
                result = &input;
            }
            else
            {
                result = new cu::Tensor<float>(input.shape, input.is_cuda);
            }
            // Tensor<T> result(tensor, tensor.is_cuda);
            if (input.is_cuda)
            {
                check_cuda_error(cudaDeviceSynchronize());
                int block_size = 256;
                int grid_size = (input.size + block_size - 1) / block_size;
                cu::kernel::Srelu<<<grid_size, block_size>>>(input.data, result->data, result->size);
                check_cuda_error(cudaGetLastError());
                check_cuda_error(cudaDeviceSynchronize());
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
                cu::Tensor<float> new_result = std::move(*result);
                return new_result;
            }
        }
    };

    class BatchNorm1d : public Module
    {
    public:
        cu::Tensor<float> weight;
        cu::Tensor<float> bias;
        cu::Tensor<float> running_mean;
        cu::Tensor<float> running_var;
        float eps;
        bool inplace;

        BatchNorm1d() = default;

        BatchNorm1d(int num_features, bool is_cuda = true) : BatchNorm1d(num_features, 1e-5, 0.1, is_cuda, true)
        {
        }

        BatchNorm1d(int num_features, float eps = 1e-5, float momentum = 0.1, bool is_cuda = true, bool inplace = true)
            : eps(eps), inplace(inplace)
        {
            weight = cu::Tensor<float>({1, num_features}, is_cuda);
            bias = cu::Tensor<float>({1, num_features}, is_cuda);
            running_mean = cu::Tensor<float>({1, num_features}, is_cuda);
            running_var = cu::Tensor<float>({1, num_features}, is_cuda);
        }

        void load_state(std::vector<float> &weight_data, std::vector<float> &bias_data,
                        std::vector<float> &running_mean_data, std::vector<float> &running_var_data)
        {
            weight.load(weight_data);
            bias.load(bias_data);
            running_mean.load(running_mean_data);
            running_var.load(running_var_data);
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            load_state(state_dict["weight"], state_dict["bias"], state_dict["running_mean"], state_dict["running_var"]);
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &input) override
        {
            if (input.shape[1] != weight.shape[1])
            {
                throw std::invalid_argument("Tensor shape mismatch");
            }
            if (input.is_cuda != weight.is_cuda || input.is_cuda != bias.is_cuda)
            {
                throw std::invalid_argument("Tensor device mismatch");
            }
            cu::Tensor<float> *result;
            if (inplace)
            {
                result = &input;
            }
            else
            {
                result = new cu::Tensor<float>(input, input.is_cuda);
            }

            if (input.shape.size() == 2)
            {
                int feature_size = input.shape[1];
                int length = input.shape[0];
                if (input.is_cuda)
                {
                    check_cuda_error(cudaDeviceSynchronize());
                    dim3 block_size(16, 16);
                    dim3 grid_size(CEIL_DIV(feature_size, block_size.x), CEIL_DIV(length, block_size.y));
                    cu::kernel::Sbatchnorm1d_d2<<<grid_size, block_size>>>(input.data, result->data, running_mean.data,
                                                                           running_var.data, weight.data, bias.data, eps,
                                                                           length, feature_size);
                    check_cuda_error(cudaGetLastError());
                    check_cuda_error(cudaDeviceSynchronize());
                }
                else
                {
                    for (int i = 0; i < feature_size; ++i)
                    {
                        float mean = running_mean.get_by_index({0, i});
                        float var = running_var.get_by_index({0, i});
                        float scale = weight.get_by_index({0, i});
                        float offset = bias.get_by_index({0, i});
                        for (int j = 0; j < length; ++j)
                        {
                            float x = input.get_by_index({j, i});
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
                    check_cuda_error(cudaDeviceSynchronize());
                    int block_size = 256;
                    int grid_size = (input.size + block_size - 1) / block_size;
                    cu::kernel::Sbatchnorm1d_d3<<<grid_size, block_size>>>(input.data, result->data, running_mean.data,
                                                                           running_var.data, weight.data, bias.data, eps,
                                                                           channel_size, feature_size, input.size);
                    check_cuda_error(cudaGetLastError());
                    check_cuda_error(cudaDeviceSynchronize());
                }
                else
                {
                    for (int j = 0; j < channel_size; ++j)
                    {
                        float mean = running_mean.get_by_index({0, j});
                        float var = running_var.get_by_index({0, j});
                        float scale = weight.get_by_index({0, j});
                        float offset = bias.get_by_index({0, j});
                        for (int i = 0; i < batch_size; ++i)
                        {
                            for (int k = 0; k < feature_size; ++k)
                            {
                                float x = input.get_by_index({i, j, k});
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
                cu::Tensor<float> new_result = std::move(*result);
                return new_result;
            }
        }
    };

    namespace F
    {
        cu::Tensor<float> relu(cu::Tensor<float> &input, bool inplace = true)
        {
            ReLU relu_object(inplace);
            return relu_object(input);
        }
    } // namespace F

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
        // if (output.shape.size() != 2)
        // {
        //     throw std::invalid_argument("Only 2D tensor is supported");
        // }
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

} // namespace misc

namespace PointNet
{
    class STN3d : public nn::Module
    {
    public:
        nn::Conv1d conv1;
        nn::Conv1d conv2;
        nn::Conv1d conv3;
        nn::Linear fc1;
        nn::Linear fc2;
        nn::Linear fc3;
        nn::ReLU relu;
        nn::BatchNorm1d bn1;
        nn::BatchNorm1d bn2;
        nn::BatchNorm1d bn3;
        nn::BatchNorm1d bn4;
        nn::BatchNorm1d bn5;
        cu::Tensor<float> iden;

        STN3d() = default;

        STN3d(int channel, bool is_cuda = true)
            : conv1(channel, 64, 1, is_cuda), conv2(64, 128, 1, is_cuda), conv3(128, 1024, 1, is_cuda),
              fc1(1024, 512, is_cuda), fc2(512, 256, is_cuda), fc3(256, 9, is_cuda), relu(true), bn1(64, is_cuda),
              bn2(128, is_cuda), bn3(1024, is_cuda), bn4(512, is_cuda), bn5(256, is_cuda)
        {
            iden = cu::Tensor<float>({1, 9}, is_cuda);
            iden.load(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1});
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            std::vector<std::pair<std::string, Module *>> modules = {
                {"conv1", &conv1}, {"conv2", &conv2}, {"conv3", &conv3}, {"fc1", &fc1}, {"fc2", &fc2}, {"fc3", &fc3}, {"bn1", &bn1}, {"bn2", &bn2}, {"bn3", &bn3}, {"bn4", &bn4}, {"bn5", &bn5}};
            for (auto &m : modules)
            {
                m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
            }
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &x) override
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

    class STNkd : public nn::Module
    {
    public:
        nn::Conv1d conv1;
        nn::Conv1d conv2;
        nn::Conv1d conv3;
        nn::Linear fc1;
        nn::Linear fc2;
        nn::Linear fc3;
        nn::ReLU relu;
        nn::BatchNorm1d bn1;
        nn::BatchNorm1d bn2;
        nn::BatchNorm1d bn3;
        nn::BatchNorm1d bn4;
        nn::BatchNorm1d bn5;
        int k;
        cu::Tensor<float> iden;

        STNkd() = default;

        STNkd(int k, bool is_cuda = true)
            : conv1(k, 64, 1, is_cuda), conv2(64, 128, 1, is_cuda), conv3(128, 1024, 1, is_cuda), fc1(1024, 512, is_cuda),
              fc2(512, 256, is_cuda), fc3(256, k * k, is_cuda), relu(true), bn1(64, is_cuda), bn2(128, is_cuda),
              bn3(1024, is_cuda), bn4(512, is_cuda), bn5(256, is_cuda), k(k)
        {
            iden = cu::Identity<float>(k, is_cuda);
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            std::vector<std::pair<std::string, Module *>> modules = {
                {"conv1", &conv1}, {"conv2", &conv2}, {"conv3", &conv3}, {"fc1", &fc1}, {"fc2", &fc2}, {"fc3", &fc3}, {"bn1", &bn1}, {"bn2", &bn2}, {"bn3", &bn3}, {"bn4", &bn4}, {"bn5", &bn5}};
            for (auto &m : modules)
            {
                m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
            }
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &x) override
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
            // iden = iden.repeat({batch_size, 1});
            // x = x + iden;
            x = x + iden.repeat({batch_size, 1});
            x.view({-1, k, k});
            return x;
        }
    };

    class PointNetEncoder : public nn::Module
    {
    public:
        STN3d stn;
        nn::Conv1d conv1;
        nn::Conv1d conv2;
        nn::Conv1d conv3;
        nn::BatchNorm1d bn1;
        nn::BatchNorm1d bn2;
        nn::BatchNorm1d bn3;
        STNkd fstn;
        nn::ReLU relu;
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
                fstn = STNkd(64, is_cuda);
            }
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            std::vector<std::pair<std::string, Module *>> modules = {
                {"stn", &stn}, {"conv1", &conv1}, {"conv2", &conv2}, {"conv3", &conv3}, {"bn1", &bn1}, {"bn2", &bn2}, {"bn3", &bn3}, {"fstn", &fstn}};
            for (auto &m : modules)
            {
                m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
            }
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &x) override
        {
            int B = x.shape[0];
            int D = x.shape[1];
            int N = x.shape[2];
            // TODO: parallelize
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

    class PointNetClassifier : public nn::Module
    {
    public:
        PointNetEncoder feat;
        nn::Linear fc1;
        nn::Linear fc2;
        nn::Linear fc3;
        // nn::Dropout dropout;
        nn::BatchNorm1d bn1;
        nn::BatchNorm1d bn2;
        nn::ReLU relu;
        int k;

        PointNetClassifier() = default;

        PointNetClassifier(int k = 10, bool normal_channel = false, bool is_cuda = true)
            : feat(true, true, normal_channel ? 6 : 3, is_cuda), fc1(1024, 512, is_cuda), fc2(512, 256, is_cuda),
              fc3(256, k, is_cuda), bn1(512, is_cuda), bn2(256, is_cuda), relu(true), k(k)
        {
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &state_dict) override
        {
            std::vector<std::pair<std::string, Module *>> modules = {{"feat", &feat}, {"fc1", &fc1}, {"fc2", &fc2}, {"fc3", &fc3}, {"bn1", &bn1}, {"bn2", &bn2}};
            for (auto &m : modules)
            {
                m.second->load_state_dict(misc::get_sub_dict(state_dict, m.first));
            }
        }

        void load_state_dict(std::map<std::string, std::vector<float>> &&state_dict) override
        {
            load_state_dict(state_dict);
        }

        cu::Tensor<float> forward(cu::Tensor<float> &x) override
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
    cudaFree(0);

    std::string dir = argv[1];
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
    auto model = PointNet::PointNetClassifier(10, false, is_cuda);
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
