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

# Step 6 - matmul_tiled_kernel
#include <cuda_runtime.h>

constexpr int TILE_SMEM = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // __shared__ As[TILE][TILE], Bs[TILE][TILE]
    __shared__ float As[TILE_SMEM][TILE_SMEM];
    __shared__ float Bs[TILE_SMEM][TILE_SMEM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_SMEM + ty;
    int col = blockIdx.x * TILE_SMEM + tx;

    float sum = 0.0f;

    // Loop over tiles along the K dimension    
    // for (int tile = 0; tile < (K + TILE_SMEM - 1) / TILE_SMEM; ++tile) {
    for (int kBase = 0; kBase < K; kBase += TILE_SMEM) {
        // Columns of A and rows of B for this tile
        // int aCol = tile * TILE_SMEM + tx;
        // int bRow = tile * TILE_SMEM + ty;
        int aCol = kBase + tx;
        int bRow = kBase + ty;

        // Load into shared memory - column of A and row of B
        As[ty][tx] = row < M && aCol < K ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = bRow < K && col < N ? B[bRow * N + col] : 0.0f;

        // Wait for threads to finished shared mem load
        __syncthreads();

        // Accumulate partial dot product
        for(int k = 0 ; k < TILE_SMEM ; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Ensure all threads have finished before the next tile loads
        __syncthreads();
    }    

    // Store
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // 16x16 blocks, grid ((N+15)/16, (M+15)/16)

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);

    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 7 - matmul_tiled_1d_kernel
#include <cuda_runtime.h>

constexpr int R1_BM = 64, R1_BN = 64, R1_BK = 8, R1_TM = 8;
constexpr int TX = R1_BN / R1_TM; // 8
constexpr int THREADS = R1_BM * TX; // 512

__global__ void matmul_tiled_1d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: __shared__ As[R1_BM * R1_BK], Bs[R1_BK * R1_BN];
    __shared__ float As[R1_BM * R1_BK];
    __shared__ float Bs[R1_BK * R1_BN];

    // thread owns column tid % 64 and rows (tid / 64) * 8 .. +7
    int tid = threadIdx.x;

    const int ty = tid / TX;  // Logical output row
    const int tx = tid % TX;  // Logical column group
    
    // output row, col
    int row = blockIdx.y * R1_BM + ty;
    int col = blockIdx.x * R1_BN + tx * R1_TM;

    float acc[R1_TM] = {0.0f};

    // Process the reduction dimension in BK-sized tiles.
    for (int k0 = 0; k0 < K; k0 += R1_BK) {
        // Cooperatively load A tile: BM x BK.
        for (int idx = tid; idx < R1_BM * R1_BK; idx += THREADS) {
            const int ar = idx / R1_BK;
            const int ak = idx % R1_BK;

            const int gr = blockIdx.y * R1_BM + ar;
            const int gk = k0 + ak;

            As[ar * R1_BK + ak] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }

        // Cooperatively load A tile: BK x BN.
        for (int idx = tid; idx < R1_BK * R1_BN; idx += THREADS) {
            const int bk = idx / R1_BN;
            const int bn = idx % R1_BN;

            const int gr = k0 + bk;
            const int gc = blockIdx.x * R1_BN + bn;

            Bs[bk * R1_BN + bn] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }        

        // sync
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < R1_BK; ++kk) {
            const float a = As[ty * R1_BK + kk];

            #pragma unroll
            for (int r = 0; r < R1_TM; ++r) {
                acc[r] = fmaf(a, Bs[kk * R1_BN + tx * R1_TM + r], acc[r]);
            }
        }

        // sync
        __syncthreads();
    }
    
    // store result to output
    if(row < M && col < N) {
        #pragma unroll
        for (int r = 0; r < R1_TM; ++r) {
            const int c = col + r;
            if (c < N) {
                C[row * N + c] = acc[r];
            }
        }
    }
}

void launch_matmul_tiled_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(THREADS);
    dim3 grid(
        (N + R1_BN - 1) / R1_BN,
        (M + R1_BM - 1) / R1_BM
    );

    matmul_tiled_1d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 8 - matmul_tiled_2d_kernel
#include <cuda_runtime.h>

constexpr int R2_BM = 64, R2_BN = 64, R2_BK = 8, R2_TM = 4, R2_TN = 4;
constexpr int TX_M = R2_BM / R2_TM; // 16
constexpr int TX_N = R2_BN / R2_TN; // 16
constexpr int R2_THREADS = TX_M * TX_N; // 256

__global__ void matmul_tiled_2d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // shared tiles; thread rows (tid/16)*4.., cols (tid%16)*4..; strided guarded loads of 2 A and 2 B elements per thread
    // per k: regA[4] from As, regB[4] from Bs, 16 FMAs; guarded 4x4 store

    __shared__ float As[R2_BM * R2_BK];
    __shared__ float Bs[R2_BK * R2_BN];

    // thread owns rows (tid / 16) * 4 .. and columns (tid % 16) * 4 ..
    int tid = threadIdx.x;

    const int ty = tid / TX_N;  // Logical output row
    const int tx = tid % TX_N;  // Logical column group

    // output row, col
    int row = blockIdx.y * R2_BM + ty * R2_TM;
    int col = blockIdx.x * R2_BN + tx * R2_TN;

    float acc[R2_TM][R2_TN] = {0.0f};
    float regA[R2_TM];
    float regB[R2_TN];

    // Process the reduction dimension in BK-sized tiles.
    for (int k0 = 0; k0 < K; k0 += R2_BK) {

        // Cooperatively load A tile: BM x BK.
        for (int idx = tid; idx < R2_BM * R2_BK; idx += R2_THREADS) {
            const int ar = idx / R2_BK;
            const int ak = idx % R2_BK;

            const int gr = blockIdx.y * R2_BM + ar;
            const int gk = k0 + ak;

            As[ar * R2_BK + ak] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;            
        }

        // Cooperatively load A tile: BK x BN.
        for (int idx = tid; idx < R2_BK * R2_BN; idx += R2_THREADS) {
            const int bk = idx / R2_BN;
            const int bn = idx % R2_BN;

            const int gr = k0 + bk;
            const int gc = blockIdx.x * R2_BN + bn;

            Bs[bk * R2_BN + bn] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }        

        // sync
        __syncthreads();

        // Compute all 8 reduction terms.
        #pragma unroll
        for (int kk = 0; kk < R2_BK; ++kk) {

            #pragma unroll
            for (int i = 0; i < R2_TM; ++i) {
                regA[i] = As[(ty * R2_TM + i) * R2_BK + kk];
            }

            #pragma unroll
            for (int j = 0; j < R2_TN; ++j) {
                regB[j] = Bs[kk * R2_BN + tx * R2_TN + j];
            }

            #pragma unroll
            for (int i = 0; i < R2_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < R2_TN; ++j) {
                    acc[i][j] = fmaf(
                        regA[i], regB[j], acc[i][j]);
                }
            }
        }

        // sync
        __syncthreads();
    }

    // store result to output
    if(row < M && col < N) {
        #pragma unroll
        for(int i = 0 ; i < R2_TM ; i++) {
            const int r = row + i;
            #pragma unroll
            for(int j = 0; j < R2_TN ; j++) {
                const int c = col + j;
                if(r < M && c < N) {
                    C[r * N + c] = acc[i][j];
                }
            }
        }
    }
}

void launch_matmul_tiled_2d(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(R2_THREADS);
    dim3 grid(
        (N + R2_BN - 1) / R2_BN,
        (M + R2_BM - 1) / R2_BM
    );

    matmul_tiled_2d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

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

