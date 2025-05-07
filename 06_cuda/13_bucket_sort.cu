#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void count_kernel(int *bucket, int *key, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n){
  	atomicAdd(&bucket[key[idx]], 1);
  }
}

__global__ void sort_kernel(int *bucket, int *key, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n){
  	int i = 0;
  	for(int j = 0; j <= idx; i++){
  	  j += bucket[i];
  	}
  	key[idx] = i - 1;
  }
}

int main() {
  int n = 50;
  int range = 5;

  int *key;
  cudaMallocManaged(&key, n * sizeof(int));
  //std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_bucket;
  cudaMalloc(&d_bucket, range * sizeof(int));
  cudaMemset(d_bucket, 0, range*sizeof(int));
  /*
  std::vector<int> bucket(range);
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  */

  const int BLOCK_SIZE = 32;
  int n_blocks = (n + BLOCK_SIZE -1) / BLOCK_SIZE;
  count_kernel<<<n_blocks, BLOCK_SIZE>>>(d_bucket, key, n);
  cudaDeviceSynchronize();
  sort_kernel<<<n_blocks, BLOCK_SIZE>>>(d_bucket, key, n);
  cudaDeviceSynchronize();

/*
  for (int i=0, j=0; i<range; i++){
    for (; bucket[i]>0; bucket[i]--){
      key[j++] = i;
    }
  }
*/
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(d_bucket);
  cudaFree(key);
}
