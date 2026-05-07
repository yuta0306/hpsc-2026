#include <cstdio>
#include <cstdlib>

__global__ void histogram(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void scan(int *offset, int *temp, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j = 1; j < range; j <<= 1) {
    temp[i] = offset[i];
    __syncthreads();
    if (i >= j) offset[i] += temp[i - j];
    __syncthreads();
  }
}

__global__ void place(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  int start = (i == 0) ? 0 : offset[i - 1];
  for (int j = 0; j < bucket[i]; j++) {
    key[start + j] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *offset, *temp;
  cudaMallocManaged(&key,    n     * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&offset, range * sizeof(int));
  cudaMallocManaged(&temp,   range * sizeof(int));

  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  for (int i = 0; i < range; i++) bucket[i] = 0;

  const int M = 32;
  histogram<<<(n + M - 1) / M, M>>>(key, bucket, n);
  cudaDeviceSynchronize();

  for (int i = 0; i < range; i++) offset[i] = bucket[i];
  scan<<<1, range>>>(offset, temp, range);
  cudaDeviceSynchronize();

  place<<<1, range>>>(key, bucket, offset, range);
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) printf("%d ", key[i]);
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
  cudaFree(temp);
}
