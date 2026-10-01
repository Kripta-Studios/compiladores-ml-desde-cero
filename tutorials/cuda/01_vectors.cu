#include "common.cuh"

__global__ void axpby(const float* x, const float* y, float* z, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = 2.0f*x[i] - y[i];
}

int main() {
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
  for (int n : {0, 1, 127, 128, 129, 257, 65537}) {
    std::vector<float> x(n), y(n), z(n), ref(n);
    for (int i=0; i<n; ++i) {
      x[i] = float(i%31-15)/16; y[i] = float(i%17-8)/8;
      ref[i] = 2*x[i]-y[i];
    }
    DeviceBuffer<float> dx(n), dy(n), dz(n);
    if (n) {
      CUDA(cudaMemcpy(dx.ptr, x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
      CUDA(cudaMemcpy(dy.ptr, y.data(), n*sizeof(float), cudaMemcpyHostToDevice));
      axpby<<<(n+127)/128, 128>>>(dx.ptr, dy.ptr, dz.ptr, n);
      CUDA(cudaGetLastError());
      CUDA(cudaDeviceSynchronize());
      CUDA(cudaMemcpy(z.data(), dz.ptr, n*sizeof(float), cudaMemcpyDeviceToHost));
    }
    std::printf("n=%d max_error=%.9g PASS\n", n, compare(z, ref));
  }
}
