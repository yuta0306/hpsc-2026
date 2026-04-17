#include <cstdio>

int main() {
  int n = 10;
  double dx = 1. / n;
  double pi = 0;
#pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  printf("%17.15f\n",pi);
}

/*
  time ./12_pi
  3.142425985001098
  ./12_pi  0.01s user 0.01s system 13% cpu 0.157 total

  Using OpenMP:
  time ./12_pi
  3.142425985001098
  ./12_pi  0.01s user 0.01s system 88% cpu 0.021 total
*/