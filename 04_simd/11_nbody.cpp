#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xvec = _mm512_loadu_ps(x);
    __m512 yvec = _mm512_loadu_ps(y);
    __m512 mvec = _mm512_loadu_ps(m);
    __m512 rx = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);
    __m512 ry = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);
    __m512 r = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx),
                                               _mm512_mul_ps(ry, ry)));
    __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    __mmask16 mask = _mm512_cmpneq_epi32_mask(idx, _mm512_set1_epi32(i));
    __m512 r3 = _mm512_mul_ps(_mm512_mul_ps(r, r), r);
    fx[i] -= _mm512_mask_reduce_add_ps(mask,
               _mm512_mul_ps(_mm512_mul_ps(rx, mvec), r3));
    fy[i] -= _mm512_mask_reduce_add_ps(mask,
               _mm512_mul_ps(_mm512_mul_ps(ry, mvec), r3));
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
