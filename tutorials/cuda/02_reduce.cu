#include "common.cuh"

// Contract: launch with exactly 128 threads per block.
__global__ void partial_sum(const float* x, float* partial, int n) {
  __shared__ float tile[128];
  int t = threadIdx.x, i = blockIdx.x*blockDim.x+t;
  tile[t] = i<n ? x[i] : 0.0f;
  __syncthreads();
  for (int stride=64; stride>0; stride/=2) {
    if (t<stride) tile[t] += tile[t+stride];
    __syncthreads();
  }
  if (t==0) partial[blockIdx.x] = tile[0];
}

int main() {
  for (int n : {0, 1, 127, 128, 129, 257, 65537}) {
    int blocks=(n+127)/128;
    std::vector<float> x(n), partial(blocks);
    double ref=0, result=0;
    for (int i=0; i<n; ++i) { x[i]=float(i%19-9)/8; ref+=x[i]; }
    DeviceBuffer<float> dx(n), dp(blocks);
    if (n) {
      CUDA(cudaMemcpy(dx.ptr,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
      partial_sum<<<blocks,128>>>(dx.ptr,dp.ptr,n);
      CUDA(cudaGetLastError());
      CUDA(cudaDeviceSynchronize());
      CUDA(cudaMemcpy(partial.data(),dp.ptr,blocks*sizeof(float),cudaMemcpyDeviceToHost));
    }
    for (float value : partial) result+=value;
    if (!std::isfinite(result) || std::abs(result-ref)>1e-5) return 2;
    std::printf("n=%d sum=%.9g error=%.9g PASS\n",n,result,std::abs(result-ref));
  }
}
