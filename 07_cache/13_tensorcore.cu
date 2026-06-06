#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

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

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  int offset_a_m = 64 * blockIdx.x;
  int offset_b_n = 64 * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  __shared__ half block_a[16][64];
  __shared__ half block_b[64][16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += 16) {
    __syncthreads();
    for (int j = 0; j < 16; ++j) {
      block_a[j][i] = __float2half(d_a[(k + j) * dim_m + offset_a_m + i]);
    }
    for (int load = i; load < 16 * 64; load += 64) {
      int b_n = load / 16;
      int b_k = load % 16;
      block_b[b_n][b_k] = __float2half(d_b[(offset_b_n + b_n) * dim_k + k + b_k]);
    }
    __syncthreads();
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[0][row_tile * 16], 64);
      for (int c = 0; c < 4; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[c * 16][0], 16);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }
  }
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_id * 2 + r) * 16;
      int c_n = offset_b_n + c * 16;
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
  if ((m % 64) != 0 || (n % 64) != 0 || (k % 16) != 0) {
    fprintf(stderr, "m and n must be multiples of 64, and k must be a multiple of 16\n");
    return EXIT_FAILURE;
  }

  float *A, *B, *C, *C2;
  CUDA_CHECK(cudaMallocManaged(&A, m * k * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&B, k * n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C, m * n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C2, m * n * sizeof(float)));
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
  int tile = 64;
  dim3 block = dim3(tile);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);

  printf("CONFIG m=%d n=%d k=%d repeat=%d warmup=%d\n", m, n, k, repeat, warmup);
  printf("CONFIG custom_tile=%d block=(%d,%d,%d) grid=(%d,%d,%d)\n",
         tile, block.x, block.y, block.z, grid.x, grid.y, grid.z);
  printf("CONFIG flops=%lld\n", (long long)num_flops);

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
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
    CUDA_CHECK(cudaGetLastError());
  });
  double custom_tflops = tflops_from_ms(num_flops, custom_time.avg_ms);
  double custom_vs_cublas = custom_tflops / cublas_tflops;

  printf("PROFILE cublas avg_ms=%.3f total_ms=%.3f wall_ms=%.3f tflops=%.3f\n",
         cublas_time.avg_ms, cublas_time.total_ms, cublas_time.wall_ms, cublas_tflops);
  printf("PROFILE custom avg_ms=%.3f total_ms=%.3f wall_ms=%.3f tflops=%.3f ratio_to_cublas=%.4f\n",
         custom_time.avg_ms, custom_time.total_ms, custom_time.wall_ms,
         custom_tflops, custom_vs_cublas);
  printf("PROFILE total_measured_ms=%.3f cublas_total_ms=%.3f custom_total_ms=%.3f\n",
         cublas_time.total_ms + custom_time.total_ms, cublas_time.total_ms, custom_time.total_ms);
  printf("COMMIT_SUBJECT 修正内容: xx.xxx ms -> %.3f ms\n", custom_time.avg_ms);

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
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
