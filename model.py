"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // C[m*N + n] = sum over k of A[m*K + k] * B[k*N + n]

    for(int m_i = 0 ; m_i < M ; m_i++) {
        for(int n_i = 0 ; n_i < N ; n_i++) {
            int idx = m_i * N + n_i;
            C[idx] = 0;
            for(int k_i = 0 ; k_i < K ; k_i++) {
                C[idx] += A[m_i * K + k_i] * B[k_i * N + n_i];
            }
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    // largest |a[i] - b[i]| over n elements, 0 for n == 0

    float max_diff = 0.0f;

    for(int i = 0 ; i < n ; i++) {
        max_diff = max(max_diff, a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
    }

    return max_diff;
}

# Step 3 - matmul_naive_kernel
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // row from threadIdx.x, col from threadIdx.y, guard, accumulate over k, store
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k_i = 0 ; k_i < K ; k_i++) {
            sum += A[row * K + k_i] * B[k_i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // 16x16 blocks, grid ((M+15)/16, (N+15)/16)
    dim3 block(16,16);
    dim3 grid((M + 15)/16, (N + 15)/16);

    // launch
    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 4 - matmul_coalesced_kernel
#include <cuda_runtime.h>

__global__ void matmul_coalesced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // col from threadIdx.x, row from threadIdx.y, guard, accumulate over k, store

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k_i = 0 ; k_i < K ; k_i++) {
            sum += A[row * K + k_i] * B[k_i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    // 16x16 blocks, grid ((N+15)/16, (M+15)/16)

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);

    matmul_coalesced_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 5 - time_launch_ms
#include <cuda_runtime.h>

typedef void (*matmul_launch_fn)(const float*, const float*, float*, int, int, int);

float time_launch_ms(matmul_launch_fn launch, const float* dA, const float* dB, float* dC, int M, int N, int K, int iters) {

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    // warm-up launch
    launch<<<grid, block>>>(dA, dB, dC, M, N, K);

    // sync
    cudaDeviceSynchronize();
    
    // create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record event
    cudaEventRecord(start);
    
    // iters launches
    for(int iter = 0 ; iter < iters ; iter++) {
        launch<<<grid, block>>>(dA, dB, dC, M, N, K);
    }

    // record event
    cudaEventRecord(stop);

    // sync
    cudaEventSynchronize(stop);

    // return elapsed / iters
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / iters;
}

double matmul_gflops(int M, int N, int K, float ms) {
    // 2*M*N*K flops over ms milliseconds, in GFLOP/s
    return 2e-6 * M * N * K / ms;
}

# Step 6 - matmul_tiled_kernel (not yet solved)
# TODO: implement

# Step 7 - matmul_tiled_1d_kernel (not yet solved)
# TODO: implement

# Step 8 - matmul_tiled_2d_kernel (not yet solved)
# TODO: implement

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

