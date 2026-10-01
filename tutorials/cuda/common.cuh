#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

inline void checked(cudaError_t status, const char* expr, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "line %d: %s: %s\n", line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}
#define CUDA(expr) checked((expr), #expr, __LINE__)

// Owning storage: copying an owner would cause a double free.
template<class T> struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t count) {
    if (count) CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), count*sizeof(T)));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (ptr) CUDA(cudaFree(ptr)); }
};

inline double compare(const std::vector<float>& got,
                      const std::vector<float>& expected,
                      double atol=1e-5, double rtol=1e-5) {
  if (got.size() != expected.size()) std::exit(2);
  double worst = 0;
  for (size_t i=0; i<got.size(); ++i) {
    double error = std::abs(double(got[i])-expected[i]);
    if (!std::isfinite(got[i]) || error > atol+rtol*std::abs(expected[i])) {
      std::fprintf(stderr, "mismatch at %zu: %.9g vs %.9g\n",
                   i, got[i], expected[i]);
      std::exit(2);
    }
    worst = std::max(worst, error);
  }
  return worst;
}
