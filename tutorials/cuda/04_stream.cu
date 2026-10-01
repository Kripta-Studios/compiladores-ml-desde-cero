#include "common.cuh"

__global__ void square(const float* x,float* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) y[i]=x[i]*x[i];
}

int main() {
  constexpr int n=65537;
  float *input=nullptr,*output=nullptr;
  CUDA(cudaMallocHost(reinterpret_cast<void**>(&input),n*sizeof(float)));
  CUDA(cudaMallocHost(reinterpret_cast<void**>(&output),n*sizeof(float)));
  std::vector<float> ref(n);
  for (int i=0;i<n;++i) { input[i]=float(i%31-15)/16; ref[i]=input[i]*input[i]; }
  DeviceBuffer<float> dx(n),dy(n);
  cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  CUDA(cudaMemcpyAsync(dx.ptr,input,n*sizeof(float),cudaMemcpyHostToDevice,stream));
  square<<<(n+127)/128,128,0,stream>>>(dx.ptr,dy.ptr,n);
  CUDA(cudaGetLastError());
  CUDA(cudaMemcpyAsync(output,dy.ptr,n*sizeof(float),cudaMemcpyDeviceToHost,stream));
  CUDA(cudaStreamSynchronize(stream));
  std::vector<float> got(output,output+n);
  std::printf("n=%d stream_chain max_error=%.9g PASS\n",n,compare(got,ref));
  CUDA(cudaStreamDestroy(stream));
  CUDA(cudaFreeHost(input)); CUDA(cudaFreeHost(output));
}
