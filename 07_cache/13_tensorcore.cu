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
#include <chrono>
using namespace std;

constexpr int TILE_M = 256;
constexpr int TILE_N = 128;
constexpr int TILE_K = 256;
constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = 128;
constexpr int WGMMA_K = 16;
constexpr int WARPGROUPS_PER_BLOCK = TILE_M / WGMMA_M;
constexpr int WARPS_PER_BLOCK = WARPGROUPS_PER_BLOCK * 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WGMMA_A_TILE_ELEMENTS = WGMMA_M * WGMMA_K;
constexpr int WGMMA_B_TILE_ELEMENTS = WGMMA_N * WGMMA_K;
constexpr int WGMMA_STAGES_PER_TILE = TILE_K / WGMMA_K;
constexpr int WGMMA_MN_GROUP_STRIDE = 64;
constexpr int WGMMA_A_K_GROUP_STRIDE = (WGMMA_M / 8) * WGMMA_MN_GROUP_STRIDE;
constexpr int WGMMA_B_K_GROUP_STRIDE = (WGMMA_N / 8) * WGMMA_MN_GROUP_STRIDE;
constexpr int WGMMA_A_LBO_BYTES = WGMMA_A_K_GROUP_STRIDE * int(sizeof(half));
constexpr int WGMMA_B_LBO_BYTES = WGMMA_B_K_GROUP_STRIDE * int(sizeof(half));
constexpr int WGMMA_SBO_BYTES = WGMMA_MN_GROUP_STRIDE * int(sizeof(half));
constexpr int DYNAMIC_SMEM_BYTES =
  (WARPGROUPS_PER_BLOCK * WGMMA_STAGES_PER_TILE * WGMMA_A_TILE_ELEMENTS +
   WGMMA_STAGES_PER_TILE * WGMMA_B_TILE_ELEMENTS) * int(sizeof(half));

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

__device__ __forceinline__ int wgmma_k_major_index(int mn, int k, int k_group_stride) {
  return (mn & 7) * 8 + (k & 7) +
         (mn >> 3) * WGMMA_MN_GROUP_STRIDE +
         (k >> 3) * k_group_stride;
}

__device__ __forceinline__ uint32_t smem_address(const void *ptr) {
  uint32_t address;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(address) : "l"(ptr));
  return address;
}

__device__ __forceinline__ uint64_t wgmma_descriptor(const void *ptr, int lbo_bytes) {
  uint64_t address = smem_address(ptr);
  uint64_t desc = 0;
  desc |= ((address >> 4) & 0x3ffffull);
  desc |= (uint64_t(lbo_bytes >> 4) & 0x3ffffull) << 16;
  desc |= (uint64_t(WGMMA_SBO_BYTES >> 4) & 0x3ffffull) << 32;
  return desc;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_shared_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n128k16_f32_f16_f16(float d[64],
                                                             uint64_t desc_a,
                                                             uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b)
      : "memory");
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       const half *d_a, const half *d_b, float *d_c) {
  int offset_a_m = TILE_M * blockIdx.x;
  int offset_b_n = TILE_N * blockIdx.y;
  int wg_id = threadIdx.x / 128;
  int wg_tid = threadIdx.x & 127;
  int lane = wg_tid & 31;
  int warp_in_wg = wg_tid >> 5;

  extern __shared__ half shared_storage[];
  half *block_a = shared_storage;
  half *block_b = block_a + WARPGROUPS_PER_BLOCK * WGMMA_STAGES_PER_TILE * WGMMA_A_TILE_ELEMENTS;

  float acc[64];
  for (int i = 0; i < 64; i++) {
    acc[i] = 0.0f;
  }
  wgmma_fence();

  for (int k = 0; k < dim_k; k += TILE_K) {
    for (int load = threadIdx.x; load < WARPGROUPS_PER_BLOCK * WGMMA_STAGES_PER_TILE * WGMMA_A_TILE_ELEMENTS; load += THREADS_PER_BLOCK) {
      int wg_stage = load / WGMMA_A_TILE_ELEMENTS;
      int rem = load - wg_stage * WGMMA_A_TILE_ELEMENTS;
      int wg = wg_stage / WGMMA_STAGES_PER_TILE;
      int stage = wg_stage - wg * WGMMA_STAGES_PER_TILE;
      int kk = rem / WGMMA_M;
      int row = rem - kk * WGMMA_M;
      block_a[wg_stage * WGMMA_A_TILE_ELEMENTS + wgmma_k_major_index(row, kk, WGMMA_A_K_GROUP_STRIDE)] =
        d_a[(k + stage * WGMMA_K + kk) * dim_m + offset_a_m + wg * WGMMA_M + row];
    }
    for (int load = threadIdx.x; load < WGMMA_STAGES_PER_TILE * WGMMA_B_TILE_ELEMENTS; load += THREADS_PER_BLOCK) {
      int stage = load / WGMMA_B_TILE_ELEMENTS;
      int rem = load - stage * WGMMA_B_TILE_ELEMENTS;
      int kk = rem / WGMMA_N;
      int col = rem - kk * WGMMA_N;
      block_b[stage * WGMMA_B_TILE_ELEMENTS + wgmma_k_major_index(col, kk, WGMMA_B_K_GROUP_STRIDE)] =
        d_b[(offset_b_n + col) * dim_k + k + stage * WGMMA_K + kk];
    }
    __syncthreads();
    wgmma_shared_fence();

    half *wg_a_base = block_a + wg_id * WGMMA_STAGES_PER_TILE * WGMMA_A_TILE_ELEMENTS;
    for (int stage = 0; stage < WGMMA_STAGES_PER_TILE; stage++) {
      uint64_t desc_a = wgmma_descriptor(wg_a_base + stage * WGMMA_A_TILE_ELEMENTS, WGMMA_A_LBO_BYTES);
      uint64_t desc_b = wgmma_descriptor(block_b + stage * WGMMA_B_TILE_ELEMENTS, WGMMA_B_LBO_BYTES);
      wgmma_m64n128k16_f32_f16_f16(acc, desc_a, desc_b);
    }
    wgmma_commit_group();
    wgmma_wait_group();
    __syncthreads();
  }

  int row_base = warp_in_wg * 16 + (lane >> 2);
  int col_base = (lane & 3) * 2;
  int global_row_base = offset_a_m + wg_id * WGMMA_M + row_base;
  for (int group = 0; group < WGMMA_N / 8; group++) {
    int col0 = offset_b_n + group * 8 + col_base;
    int reg = group * 4;
    d_c[(col0 + 0) * dim_m + global_row_base + 0] = acc[reg + 0];
    d_c[(col0 + 1) * dim_m + global_row_base + 0] = acc[reg + 1];
    d_c[(col0 + 0) * dim_m + global_row_base + 8] = acc[reg + 2];
    d_c[(col0 + 1) * dim_m + global_row_base + 8] = acc[reg + 3];
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
  printf("CONFIG custom_kernel=inline_ptx_wgmma\n");
  printf("CONFIG wgmma_m=%d wgmma_n=%d wgmma_k=%d warpgroups_per_block=%d\n",
         WGMMA_M, WGMMA_N, WGMMA_K, WARPGROUPS_PER_BLOCK);
  printf("CONFIG wgmma_a_lbo_bytes=%d wgmma_b_lbo_bytes=%d wgmma_sbo_bytes=%d\n",
         WGMMA_A_LBO_BYTES, WGMMA_B_LBO_BYTES, WGMMA_SBO_BYTES);
  printf("CONFIG dynamic_smem_bytes=%d\n", DYNAMIC_SMEM_BYTES);
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
