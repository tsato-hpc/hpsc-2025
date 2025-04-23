#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);

    __m512 xjvec = _mm512_load_ps(x);
    __m512 yjvec = _mm512_load_ps(y);
    __m512 mvec = _mm512_load_ps(m);

    __mmask16 mask = _mm512_cmp_epi32_mask(
	_mm512_set1_epi32(i),
	_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
	_MM_CMPINT_NE
    );

    __m512 rxvec = _mm512_sub_ps(xivec, xjvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yjvec);

    __m512 rvec = _mm512_rsqrt14_ps(
	_mm512_add_ps(
		_mm512_mul_ps(rxvec, rxvec),
		 _mm512_mul_ps(ryvec, ryvec)
	)
    );

    __m512 r3vec = _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec));



    __m512 fxi = _mm512_mul_ps(_mm512_mul_ps(rxvec, mvec), r3vec);
    __m512 fyi = _mm512_mul_ps(_mm512_mul_ps(ryvec, mvec), r3vec);

    fxi = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), fxi);
    fyi = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), fyi);

    fx[i] -= _mm512_reduce_add_ps(fxi);
    fy[i] -= _mm512_reduce_add_ps(fyi);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

