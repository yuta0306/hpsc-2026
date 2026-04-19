#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range, 0);
#pragma omp parallel
  {
    std::vector<int> local_bucket(range, 0);
#pragma omp for
    for (int i = 0; i < n; i++) {
      local_bucket[key[i]]++;
    }
    for (int i = 0; i < range; i++) {
#pragma omp atomic
      bucket[i] += local_bucket[i];
    }
  }

  std::vector<int> offset(range);
  for (int i = 0; i < range; i++)
    offset[i] = bucket[i];

  std::vector<int> temp(range);
#pragma omp parallel
  for (int j = 1; j < range; j <<= 1) {
#pragma omp for
    for (int i = 0; i < range; i++)
      temp[i] = offset[i];
#pragma omp for
    for (int i = j; i < range; i++)
      offset[i] += temp[i - j];
  }

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < range; i++) {
    int start = (i == 0) ? 0 : offset[i - 1];
    for (int j = 0; j < bucket[i]; j++) {
      key[start + j] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

/*
  time ./14_bucket_sort
  2 4 3 3 0 2 4 3 3 4 0 0 2 2 2 3 2 4 0 2 3 4 4 2 0 3 4 3 1 0 2 1 2 2 0 3 4 4 4 1 2 2 3 1 0 0 3 1 4 2 
  0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 
  ./14_bucket_sort  0.01s user 0.00s system 89% cpu 0.018 total

  Using OpenMP:
  time ./14_bucket_sort
  2 4 3 3 0 2 4 3 3 4 0 0 2 2 2 3 2 4 0 2 3 4 4 2 0 3 4 3 1 0 2 1 2 2 0 3 4 4 4 1 2 2 3 1 0 0 3 1 4 2 
  0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 
  ./14_bucket_sort  0.01s user 0.01s system 98% cpu 0.015 total
*/