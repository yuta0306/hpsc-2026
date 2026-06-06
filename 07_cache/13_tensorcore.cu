#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

constexpr int TILE_M = 256;
constexpr int TILE_N = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int TILE_K = 128;
constexpr int SMEM_PAD = 8;
constexpr int SMEM_A_LD = TILE_M + SMEM_PAD;
constexpr int SMEM_B_LD = TILE_K + SMEM_PAD;
constexpr int LOAD_VECTOR_WIDTH = 4;
constexpr int REUSE_B_FRAGMENTS = 1;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WARP_M_FRAGS = 2;
constexpr int WARP_N_FRAGS = TILE_N / WMMA_N;
constexpr int DYNAMIC_SMEM_BYTES = (TILE_K * SMEM_A_LD + TILE_N * SMEM_B_LD) * int(sizeof(half));

static const char *cublas_status_name(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#ifdef CUBLAS_STATUS_NOT_SUPPORTED
  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  default: return "CUBLAS_STATUS_UNKNOWN";
  }
}

static void check_cuda(cudaError_t status, const char *call, const char *file, int line) {
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",
            file, line, call, cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  }
}

static void check_cublas(cublasStatus_t status, const char *call, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS error at %s:%d: %s failed: %s\n",
            file, line, call, cublas_status_name(status));
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) check_cublas((call), #call, __FILE__, __LINE__)

struct TimingResult {
  double total_ms;
  double avg_ms;
  double wall_ms;
};

template <typename Launch>
TimingResult measure_gpu(int warmup, int repeat, Launch launch) {
  for (int i = 0; i < warmup; i++) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  double total_ms = 0.0;
  auto wall_start = chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_ms += elapsed_ms;
  }
  auto wall_stop = chrono::steady_clock::now();

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  TimingResult result;
  result.total_ms = total_ms;
  result.avg_ms = total_ms / repeat;
  result.wall_ms = chrono::duration<double, milli>(wall_stop - wall_start).count();
  return result;
}

static double tflops_from_ms(int64_t flops, double ms) {
  return double(flops) / (ms * 1.0e-3) / 1.0e12;
}

__global__ void convert_float_to_half(const float *input, half *output, int64_t elements) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       const half *d_a, const half *d_b, float *d_c) {
  int offset_a_m = TILE_M * blockIdx.x;
  int offset_b_n = TILE_N * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  extern __shared__ half shared_storage[];
  half *block_a = shared_storage;
  half *block_b = block_a + TILE_K * SMEM_A_LD;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_M_FRAGS][WARP_N_FRAGS];
  for (int r = 0; r < WARP_M_FRAGS; r++)
    for (int c = 0; c < WARP_N_FRAGS; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += TILE_K) {
    __syncthreads();
    for (int load = i; load < TILE_K * (TILE_M / LOAD_VECTOR_WIDTH); load += THREADS_PER_BLOCK) {
      int a_k = load / (TILE_M / LOAD_VECTOR_WIDTH);
      int a_m = (load % (TILE_M / LOAD_VECTOR_WIDTH)) * LOAD_VECTOR_WIDTH;
      const uint2 *src = reinterpret_cast<const uint2 *>(&d_a[(k + a_k) * dim_m + offset_a_m + a_m]);
      uint2 *dst = reinterpret_cast<uint2 *>(&block_a[a_k * SMEM_A_LD + a_m]);
      *dst = *src;
    }
    for (int load = i; load < TILE_N * (TILE_K / LOAD_VECTOR_WIDTH); load += THREADS_PER_BLOCK) {
      int b_n = load / (TILE_K / LOAD_VECTOR_WIDTH);
      int b_k = (load % (TILE_K / LOAD_VECTOR_WIDTH)) * LOAD_VECTOR_WIDTH;
      const uint2 *src = reinterpret_cast<const uint2 *>(&d_b[(offset_b_n + b_n) * dim_k + k + b_k]);
      uint2 *dst = reinterpret_cast<uint2 *>(&block_b[b_n * SMEM_B_LD + b_k]);
      *dst = *src;
    }
    __syncthreads();
    for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[WARP_N_FRAGS];
      for (int c = 0; c < WARP_N_FRAGS; c++) {
        wmma::load_matrix_sync(b_frag[c], &block_b[(c * WMMA_N) * SMEM_B_LD + kk], SMEM_B_LD);
      }
      for (int r = 0; r < WARP_M_FRAGS; r++) {
        int row_tile = warp_id * WARP_M_FRAGS + r;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[kk * SMEM_A_LD + row_tile * WMMA_M], SMEM_A_LD);
        for (int c = 0; c < WARP_N_FRAGS; c++) {
          wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
        }
      }
    }
  }
  for (int r = 0; r < WARP_M_FRAGS; r++) {
    for (int c = 0; c < WARP_N_FRAGS; c++) {
      int c_m = offset_a_m + (warp_id * WARP_M_FRAGS + r) * WMMA_M;
      int c_n = offset_b_n + c * WMMA_N;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int repeat = 10;
  int warmup = 2;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);
  if (argc > 4) repeat = atoi(argv[4]);
  if (argc > 5) warmup = atoi(argv[5]);
  if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0 || warmup < 0) {
    fprintf(stderr, "usage: %s [m n k repeat warmup]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if ((m % TILE_M) != 0 || (n % TILE_N) != 0 || (k % TILE_K) != 0) {
    fprintf(stderr, "m must be a multiple of %d, n must be a multiple of %d, and k must be a multiple of %d\n",
            TILE_M, TILE_N, TILE_K);
    return EXIT_FAILURE;
  }

  float *A, *B, *C, *C2;
  half *A_half, *B_half;
  CUDA_CHECK(cudaMallocManaged(&A, m * k * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&B, k * n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C, m * n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C2, m * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&A_half, m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&B_half, k * n * sizeof(half)));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  dim3 block = dim3(THREADS_PER_BLOCK);
  dim3 grid = dim3((m + TILE_M - 1) / TILE_M, (n + TILE_N - 1) / TILE_N);
  int convert_block = 256;
  int convert_a_grid = (int)((int64_t(m) * int64_t(k) + convert_block - 1) / convert_block);
  int convert_b_grid = (int)((int64_t(k) * int64_t(n) + convert_block - 1) / convert_block);

  printf("CONFIG m=%d n=%d k=%d repeat=%d warmup=%d\n", m, n, k, repeat, warmup);
  printf("CONFIG custom_tile_m=%d custom_tile_n=%d custom_tile_k=%d warps=%d block=(%d,%d,%d) grid=(%d,%d,%d)\n",
         TILE_M, TILE_N, TILE_K, WARPS_PER_BLOCK, block.x, block.y, block.z, grid.x, grid.y, grid.z);
  printf("CONFIG wmma_m=%d wmma_n=%d wmma_k=%d k_stages_per_load=%d\n",
         WMMA_M, WMMA_N, WMMA_K, TILE_K / WMMA_K);
  printf("CONFIG warp_m_fragments=%d warp_n_fragments=%d\n",
         WARP_M_FRAGS, WARP_N_FRAGS);
  printf("CONFIG smem_pad=%d smem_a_ld=%d smem_b_ld=%d\n",
         SMEM_PAD, SMEM_A_LD, SMEM_B_LD);
  printf("CONFIG dynamic_smem_bytes=%d\n", DYNAMIC_SMEM_BYTES);
  printf("CONFIG load_vector_width=%d\n", LOAD_VECTOR_WIDTH);
  printf("CONFIG reuse_b_fragments=%d\n", REUSE_B_FRAGMENTS);
  printf("CONFIG convert_block=%d convert_grid_a=%d convert_grid_b=%d\n",
         convert_block, convert_a_grid, convert_b_grid);
  printf("CONFIG flops=%lld\n", (long long)num_flops);
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  DYNAMIC_SMEM_BYTES));

  TimingResult convert_time = measure_gpu(warmup, repeat, [&]() {
    convert_float_to_half<<< convert_a_grid, convert_block >>>(A, A_half, int64_t(m) * int64_t(k));
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_half<<< convert_b_grid, convert_block >>>(B, B_half, int64_t(k) * int64_t(n));
    CUDA_CHECK(cudaGetLastError());
  });

  TimingResult cublas_time = measure_gpu(warmup, repeat, [&]() {
    CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              A, CUDA_R_32F, m,
                              B, CUDA_R_32F, k,
                              &beta,
                              C, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  });
  double cublas_tflops = tflops_from_ms(num_flops, cublas_time.avg_ms);

  TimingResult custom_time = measure_gpu(warmup, repeat, [&]() {
    kernel<<< grid, block, DYNAMIC_SMEM_BYTES >>>(m,
						  n,
						  k,
						  A_half,
						  B_half,
						  C2);
    CUDA_CHECK(cudaGetLastError());
  });
  double custom_tflops = tflops_from_ms(num_flops, custom_time.avg_ms);
  double custom_vs_cublas = custom_tflops / cublas_tflops;
  double custom_with_convert_ms = convert_time.avg_ms + custom_time.avg_ms;
  double custom_with_convert_tflops = tflops_from_ms(num_flops, custom_with_convert_ms);
  double custom_with_convert_vs_cublas = custom_with_convert_tflops / cublas_tflops;

  printf("PROFILE convert avg_ms=%.3f total_ms=%.3f wall_ms=%.3f\n",
         convert_time.avg_ms, convert_time.total_ms, convert_time.wall_ms);
  printf("PROFILE cublas avg_ms=%.3f total_ms=%.3f wall_ms=%.3f tflops=%.3f\n",
         cublas_time.avg_ms, cublas_time.total_ms, cublas_time.wall_ms, cublas_tflops);
  printf("PROFILE custom avg_ms=%.3f total_ms=%.3f wall_ms=%.3f tflops=%.3f ratio_to_cublas=%.4f\n",
         custom_time.avg_ms, custom_time.total_ms, custom_time.wall_ms,
         custom_tflops, custom_vs_cublas);
  printf("PROFILE custom_with_convert avg_ms=%.3f tflops=%.3f ratio_to_cublas=%.4f\n",
         custom_with_convert_ms, custom_with_convert_tflops, custom_with_convert_vs_cublas);
  printf("PROFILE total_measured_ms=%.3f convert_total_ms=%.3f cublas_total_ms=%.3f custom_total_ms=%.3f\n",
         convert_time.total_ms + cublas_time.total_ms + custom_time.total_ms,
         convert_time.total_ms, cublas_time.total_ms, custom_time.total_ms);
  printf("COMMIT_SUBJECT 修正内容: xx.xxx ms -> %.3f ms\n", custom_time.avg_ms);
  printf("COMMIT_SUBJECT_WITH_CONVERT 修正内容: xx.xxx ms -> %.3f ms\n", custom_with_convert_ms);

  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error_avg_abs=%lf\n", err/n/m);
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(C2));
  CUDA_CHECK(cudaFree(A_half));
  CUDA_CHECK(cudaFree(B_half));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
