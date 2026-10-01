// Complete one-warp WMMA example. Requires a compatible NVIDIA GPU/toolkit.
// Validated on RTX 5070 Ti Laptop (sm_120), CUDA 12.9; see validation reports.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

__global__ void tile16(const half* A, const half* B, float* C) {
    using namespace nvcuda;
    // Every lane of the only warp must execute the same collective operations.
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    half *A=nullptr,*B=nullptr;
    float* C=nullptr;
    try {
        cudaDeviceProp property{};
        check(cudaGetDeviceProperties(&property,0),"device properties");
        if (property.major < 7) {
            std::fprintf(stderr,"This teaching example needs compatible WMMA support.\n");
            return 77;
        }
        std::printf("Device: %s; compute capability %d.%d\n",property.name,property.major,property.minor);
        half ha[256],hb[256]; float hc[256];
        for (int i=0;i<256;i++) {
            ha[i]=__float2half(((i*7)%13-6)*0.25f);
            hb[i]=__float2half(((i*3)%11-5)*0.125f);
        }
        check(cudaMalloc((void**)&A,sizeof(ha)),"allocate A");
        check(cudaMalloc((void**)&B,sizeof(hb)),"allocate B");
        check(cudaMalloc((void**)&C,sizeof(hc)),"allocate C");
        check(cudaMemcpy(A,ha,sizeof(ha),cudaMemcpyHostToDevice),"copy A");
        check(cudaMemcpy(B,hb,sizeof(hb),cudaMemcpyHostToDevice),"copy B");
        tile16<<<1,32>>>(A,B,C);
        check(cudaGetLastError(),"launch");
        check(cudaDeviceSynchronize(),"synchronize");
        check(cudaMemcpy(hc,C,sizeof(hc),cudaMemcpyDeviceToHost),"copy C");
        double error=0;
        for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
            double reference=0;
            for (int k=0;k<16;k++)
                reference+=(double)__half2float(ha[i*16+k])*__half2float(hb[k*16+j]);
            error=std::fmax(error,std::fabs(hc[i*16+j]-reference));
        }
        check(cudaFree(A),"free A"); A=nullptr;
        check(cudaFree(B),"free B"); B=nullptr;
        check(cudaFree(C),"free C"); C=nullptr;
        std::printf("Maximum absolute error: %.9g\n",error);
        return error <= 1e-4 ? 0 : 1;
    } catch(const std::exception& e) {
        if(A) cudaFree(A); if(B) cudaFree(B); if(C) cudaFree(C);
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
}
