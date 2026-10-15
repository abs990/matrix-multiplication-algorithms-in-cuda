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

# Step 2 - max_abs_diff (not yet solved)
# TODO: implement

# Step 3 - matmul_naive_kernel (not yet solved)
# TODO: implement

# Step 4 - matmul_coalesced_kernel (not yet solved)
# TODO: implement

# Step 5 - time_launch_ms (not yet solved)
# TODO: implement

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

