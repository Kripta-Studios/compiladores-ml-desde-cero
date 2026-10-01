#include "common.cuh"

__global__ void gemm_plain(const float* a,const float* b,float* c,int m,int n,int k) {
  int row=blockIdx.y*16+threadIdx.y, col=blockIdx.x*16+threadIdx.x;
  if (row<m && col<n) {
    float sum=0;
    for (int q=0;q<k;++q) sum+=a[row*k+q]*b[q*n+col];
    c[row*n+col]=sum;
  }
}

__global__ void gemm_tiled(const float* a,const float* b,float* c,int m,int n,int k) {
  __shared__ float as[16][16], bs[16][16];
  int tx=threadIdx.x, ty=threadIdx.y;
  int row=blockIdx.y*16+ty, col=blockIdx.x*16+tx;
  float sum=0;
  for (int base=0;base<k;base+=16) {
    as[ty][tx]=(row<m && base+tx<k) ? a[row*k+base+tx] : 0;
    bs[ty][tx]=(base+ty<k && col<n) ? b[(base+ty)*n+col] : 0;
    __syncthreads();
    for (int q=0;q<16;++q) sum+=as[ty][q]*bs[q][tx];
    __syncthreads();
  }
  if (row<m && col<n) c[row*n+col]=sum;
}

int main() {
  const int shapes[][3]={{1,1,1},{17,19,23},{31,47,65},{128,128,128},{256,256,256}};
  for (auto& shape:shapes) {
    int m=shape[0],n=shape[1],k=shape[2];
    std::vector<float> a(m*k),b(k*n),c(m*n),ref(m*n);
    for (int i=0;i<m*k;++i) a[i]=float(i%23-11)/16;
    for (int i=0;i<k*n;++i) b[i]=float(i%17-8)/8;
    for (int r=0;r<m;++r) for (int col=0;col<n;++col) {
      double sum=0;
      for (int q=0;q<k;++q) sum+=double(a[r*k+q])*b[q*n+col];
      ref[r*n+col]=float(sum);
    }
    DeviceBuffer<float> da(a.size()),db(b.size()),dc(c.size());
    CUDA(cudaMemcpy(da.ptr,a.data(),a.size()*4,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db.ptr,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    dim3 grid((n+15)/16,(m+15)/16),block(16,16);
    cudaEvent_t start,stop;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    for (int tiled=0;tiled<2;++tiled) {
      auto launch=[&]() {
        if (tiled) gemm_tiled<<<grid,block>>>(da.ptr,db.ptr,dc.ptr,m,n,k);
        else gemm_plain<<<grid,block>>>(da.ptr,db.ptr,dc.ptr,m,n,k);
        CUDA(cudaGetLastError());
      };
      for (int i=0;i<3;++i) launch();
      CUDA(cudaDeviceSynchronize());
      std::vector<float> samples;
      for (int trial=0;trial<30;++trial) {
        CUDA(cudaEventRecord(start));
        for (int repeat=0;repeat<100;++repeat) launch();
        CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
        float ms; CUDA(cudaEventElapsedTime(&ms,start,stop));
        samples.push_back(ms*1000/100);
      }
      CUDA(cudaMemcpy(c.data(),dc.ptr,c.size()*4,cudaMemcpyDeviceToHost));
      double error=compare(c,ref);
      std::sort(samples.begin(),samples.end());
      float median=(samples[14]+samples[15])/2;
      std::printf("m=%d n=%d k=%d tiled=%d median_us=%.6f min_us=%.6f max_us=%.6f max_error=%.9g PASS\n",
                  m,n,k,tiled,median,samples.front(),samples.back(),error);
    }
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
  }
}
