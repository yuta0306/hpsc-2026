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

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 256;
constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = 256;
constexpr int WGMMA_K = 16;
constexpr int WARPGROUPS_PER_BLOCK = TILE_M / WGMMA_M;
constexpr int WARPS_PER_BLOCK = WARPGROUPS_PER_BLOCK * 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WGMMA_B_TILE_ELEMENTS = WGMMA_N * WGMMA_K;
constexpr int WGMMA_STAGES_PER_TILE = TILE_K / WGMMA_K;
constexpr int WGMMA_B_K_PACK = 8;
constexpr int WGMMA_B_VECTORS_PER_STAGE = WGMMA_N * (WGMMA_K / WGMMA_B_K_PACK);
constexpr int WGMMA_B_VECTORS_PER_TILE = WGMMA_STAGES_PER_TILE * WGMMA_B_VECTORS_PER_STAGE;
constexpr int B_LOADS_PER_THREAD =
  WGMMA_B_VECTORS_PER_TILE / THREADS_PER_BLOCK;
constexpr int B_BUFFER_COUNT = 1;
constexpr int WGMMA_MN_GROUP_STRIDE = 64;
constexpr int WGMMA_B_K_GROUP_STRIDE = (WGMMA_N / 8) * WGMMA_MN_GROUP_STRIDE;
constexpr int WGMMA_B_LBO_BYTES = WGMMA_B_K_GROUP_STRIDE * int(sizeof(half));
constexpr int WGMMA_SBO_BYTES = WGMMA_MN_GROUP_STRIDE * int(sizeof(half));
constexpr int WGMMA_B_SW128_LEADING_BYTES = 16;
constexpr int WGMMA_B_SW128_STRIDE_BYTES = 1024;
constexpr int DYNAMIC_SMEM_BYTES =
  B_BUFFER_COUNT * WGMMA_STAGES_PER_TILE * WGMMA_B_TILE_ELEMENTS * int(sizeof(half));

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

__global__ void convert_float_to_half(const float *__restrict__ input, half *__restrict__ output, int64_t elements) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void convert_a_to_wgmma_rs(const float *__restrict__ input, half *__restrict__ output, int dim_m, int dim_k) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packed_elements = int64_t(dim_m) * int64_t(dim_k) / 2;
  if (idx < packed_elements) {
    int frag = int(idx & 3);
    int64_t tmp = idx >> 2;
    int wg_tid = int(tmp & 127);
    tmp >>= 7;
    int k_stage = int(tmp % (dim_k / WGMMA_K));
    int row64 = int(tmp / (dim_k / WGMMA_K));

    int lane = wg_tid & 31;
    int warp_in_wg = wg_tid >> 5;
    int row_base = row64 * WGMMA_M + warp_in_wg * 16 + (lane >> 2);
    int k_pair = (lane & 3) * 2;
    int row = row_base + ((frag & 1) ? 8 : 0);
    int col = k_stage * WGMMA_K + k_pair + ((frag & 2) ? 8 : 0);

    half h0 = __float2half(input[int64_t(col) * dim_m + row]);
    half h1 = __float2half(input[int64_t(col + 1) * dim_m + row]);
    uint32_t packed = uint32_t(__half_as_ushort(h0)) | (uint32_t(__half_as_ushort(h1)) << 16);
    reinterpret_cast<uint32_t *>(output)[idx] = packed;
  }
}

__global__ void convert_b_to_wgmma_vectors(const float *__restrict__ input, half *__restrict__ output, int dim_k, int dim_n) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packed_vectors = int64_t(dim_k) * int64_t(dim_n) / WGMMA_B_K_PACK;
  if (idx < packed_vectors) {
    int vector_in_stage = int(idx % WGMMA_B_VECTORS_PER_STAGE);
    int64_t tmp = idx / WGMMA_B_VECTORS_PER_STAGE;
    int k_stage = int(tmp % (dim_k / WGMMA_K));
    int tile_n = int(tmp / (dim_k / WGMMA_K));
    int col = vector_in_stage / (WGMMA_K / WGMMA_B_K_PACK);
    int kk = (vector_in_stage - col * (WGMMA_K / WGMMA_B_K_PACK)) * WGMMA_B_K_PACK;
    int k_tile = k_stage / WGMMA_STAGES_PER_TILE;
    int stage = k_stage - k_tile * WGMMA_STAGES_PER_TILE;
    int global_col = tile_n * WGMMA_N + col;
    int global_k = k_stage * WGMMA_K + kk;

    uint32_t packed01 = uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 0]))) |
                        (uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 1]))) << 16);
    uint32_t packed23 = uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 2]))) |
                        (uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 3]))) << 16);
    uint32_t packed45 = uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 4]))) |
                        (uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 5]))) << 16);
    uint32_t packed67 = uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 6]))) |
                        (uint32_t(__half_as_ushort(__float2half(input[int64_t(global_col) * dim_k + global_k + 7]))) << 16);
    int stage_k = stage * WGMMA_K + kk;
    int logical_half_offset = (stage_k >> 6) * (WGMMA_N * 64) + col * 64 + (stage_k & 63);
    int logical_byte_offset = logical_half_offset * int(sizeof(half));
    int physical_byte_offset = logical_byte_offset ^ ((logical_byte_offset & (0x7 << 7)) >> 3);
    int physical_vector = physical_byte_offset >> 4;
    int64_t out_idx =
      (int64_t(tile_n) * (dim_k / TILE_K) + k_tile) * WGMMA_B_VECTORS_PER_TILE + physical_vector;
    reinterpret_cast<uint4 *>(output)[out_idx] = make_uint4(packed01, packed23, packed45, packed67);
  }
}

__device__ __forceinline__ int wgmma_k_major_index(int mn, int k, int k_group_stride) {
  return (mn & 7) * 8 + (k & 7) +
         (mn >> 3) * WGMMA_MN_GROUP_STRIDE +
         (k >> 3) * k_group_stride;
}

__device__ __forceinline__ int wgmma_k_sw128_logical_index(int mn, int k) {
  return (k >> 6) * (WGMMA_N * 64) + mn * 64 + (k & 63);
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

__device__ __forceinline__ uint64_t wgmma_descriptor_b_sw128(const void *ptr) {
  uint64_t address = smem_address(ptr);
  uint64_t desc = 0;
  desc |= ((address >> 4) & 0x3fffull);
  desc |= (uint64_t(WGMMA_B_SW128_LEADING_BYTES >> 4) & 0x3fffull) << 16;
  desc |= (uint64_t(WGMMA_B_SW128_STRIDE_BYTES >> 4) & 0x3fffull) << 32;
  desc |= 1ull << 62;
  return desc;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_group0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_shared_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_16(void *shared_dst, const void *global_src) {
  uint32_t shared_addr = smem_address(shared_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :: "r"(shared_addr), "l"(global_src) : "memory");
}

__device__ __forceinline__ void cp_async_16_smem_addr(uint32_t shared_addr, const void *global_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :: "r"(shared_addr), "l"(global_src) : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group0() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n128k16_f32_f16_f16_rs(float d[64],
                                                                const uint32_t a[4],
                                                                uint64_t desc_b,
                                                                int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
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
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d)
      : "memory");
}

__device__ __forceinline__ void wgmma_m64n256k16_f32_f16_f16_rs(float d[128],
                                                                const uint32_t a[4],
                                                                uint64_t desc_b,
                                                                int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %133, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, "
      "%72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, "
      "%88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, "
      "%104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, "
      "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
      "}\n"
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
        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),
        "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
        "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
        "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),
        "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]),
        "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),
        "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
        "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
        "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d)
      : "memory");
}

template <int STATIC_M, int STATIC_N, int STATIC_K>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void kernel(const half *__restrict__ d_a,
            const half *__restrict__ d_b,
            float *__restrict__ d_c) {
  constexpr int dim_m = STATIC_M;
  constexpr int dim_k = STATIC_K;
  int offset_a_m = TILE_M * blockIdx.x;
  int offset_b_n = TILE_N * blockIdx.y;
  int wg_id = threadIdx.x / 128;
  int wg_tid = threadIdx.x & 127;
  int lane = wg_tid & 31;
  int warp_in_wg = wg_tid >> 5;

  extern __shared__ half shared_storage[];
  half *block_b = shared_storage;
  uint64_t desc_b[B_BUFFER_COUNT][WGMMA_STAGES_PER_TILE];
  #pragma unroll
  for (int buffer = 0; buffer < B_BUFFER_COUNT; buffer++) {
    #pragma unroll
    for (int stage = 0; stage < WGMMA_STAGES_PER_TILE; stage++) {
      desc_b[buffer][stage] =
        wgmma_descriptor_b_sw128(block_b + buffer * WGMMA_STAGES_PER_TILE * WGMMA_B_TILE_ELEMENTS +
                                 wgmma_k_sw128_logical_index(0, stage * WGMMA_K));
    }
  }

  float acc[128];
  const uint32_t *a_packed = reinterpret_cast<const uint32_t *>(d_a);
  const uint4 *b_packed = reinterpret_cast<const uint4 *>(d_b);
  int a_row64 = (offset_a_m / WGMMA_M) + wg_id;
  int b_tile_n = offset_b_n / WGMMA_N;
  int k_stage_count = dim_k / WGMMA_K;
  int k_tile_count = dim_k / TILE_K;

  for (int k = 0; k < dim_k; k += TILE_K) {
    int buffer = 0;
    half *load_b = block_b + buffer * WGMMA_STAGES_PER_TILE * WGMMA_B_TILE_ELEMENTS;
    uint32_t load_b_smem = smem_address(load_b);
    const uint4 *b_tile_src =
      b_packed + (int64_t(b_tile_n) * k_tile_count + (k / TILE_K)) * WGMMA_B_VECTORS_PER_TILE;
    #pragma unroll
    for (int load_iter = 0; load_iter < B_LOADS_PER_THREAD; load_iter++) {
      int load = threadIdx.x + load_iter * THREADS_PER_BLOCK;
      cp_async_16_smem_addr(load_b_smem + load * 16, b_tile_src + load);
    }
    cp_async_commit_group();
    cp_async_wait_group0();
    __syncthreads();
    wgmma_shared_fence();
    wgmma_fence();

    int k_stage_base = k / WGMMA_K;
    const uint32_t *a_tile_src =
      a_packed + ((int64_t(a_row64) * k_stage_count + k_stage_base) * 128 + wg_tid) * 4;
    #pragma unroll
    for (int stage = 0; stage < WGMMA_STAGES_PER_TILE; stage++) {
      uint4 packed_a = *reinterpret_cast<const uint4 *>(a_tile_src + stage * 512);
      uint32_t a_frag[4] = {packed_a.x, packed_a.y, packed_a.z, packed_a.w};
      int scale_d = (k == 0 && stage == 0) ? 0 : 1;
      wgmma_m64n256k16_f32_f16_f16_rs(acc, a_frag, desc_b[buffer][stage], scale_d);
    }
    wgmma_commit_group();
    wgmma_wait_group0();
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
  if (m != 10240 || n != 8192 || k != 4096) {
    fprintf(stderr, "this optimized kernel is specialized for m=10240 n=8192 k=4096\n");
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
  int convert_a_grid = (int)((int64_t(m) * int64_t(k) / 2 + convert_block - 1) / convert_block);
  int convert_b_grid = (int)((int64_t(k) * int64_t(n) / WGMMA_B_K_PACK + convert_block - 1) / convert_block);

  printf("CONFIG m=%d n=%d k=%d repeat=%d warmup=%d\n", m, n, k, repeat, warmup);
  printf("CONFIG custom_tile_m=%d custom_tile_n=%d custom_tile_k=%d warps=%d block=(%d,%d,%d) grid=(%d,%d,%d)\n",
         TILE_M, TILE_N, TILE_K, WARPS_PER_BLOCK, block.x, block.y, block.z, grid.x, grid.y, grid.z);
  printf("CONFIG custom_kernel=inline_ptx_wgmma\n");
  printf("CONFIG wgmma_m=%d wgmma_n=%d wgmma_k=%d warpgroups_per_block=%d\n",
         WGMMA_M, WGMMA_N, WGMMA_K, WARPGROUPS_PER_BLOCK);
  printf("CONFIG wgmma_b_lbo_bytes=%d wgmma_sbo_bytes=%d b_vectors_per_stage=%d\n",
         WGMMA_B_LBO_BYTES, WGMMA_SBO_BYTES, WGMMA_B_VECTORS_PER_STAGE);
  printf("CONFIG dynamic_smem_bytes=%d\n", DYNAMIC_SMEM_BYTES);
  printf("CONFIG convert_block=%d convert_grid_a=%d convert_grid_b=%d\n",
         convert_block, convert_a_grid, convert_b_grid);
  printf("CONFIG flops=%lld\n", (long long)num_flops);
  CUDA_CHECK(cudaFuncSetAttribute(kernel<10240, 8192, 4096>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  DYNAMIC_SMEM_BYTES));
  CUDA_CHECK(cudaFuncSetAttribute(kernel<10240, 8192, 4096>,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxL1));

  TimingResult convert_time = measure_gpu(warmup, repeat, [&]() {
    convert_a_to_wgmma_rs<<< convert_a_grid, convert_block >>>(A, A_half, m, k);
    CUDA_CHECK(cudaGetLastError());
    convert_b_to_wgmma_vectors<<< convert_b_grid, convert_block >>>(B, B_half, k, n);
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
    kernel<10240, 8192, 4096><<< grid, block, DYNAMIC_SMEM_BYTES >>>(A_half,
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
