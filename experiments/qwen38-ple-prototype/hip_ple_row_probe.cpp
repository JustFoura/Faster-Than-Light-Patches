#include <hip/hip_runtime.h>
#include "ggml-quants.h"
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,hipGetErrorString(e));return 10;} } while(0)

__device__ __constant__ int8_t iq4nl[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__global__ void dequant_iq4nl(const unsigned char *row,float *dst){
    int i=threadIdx.x;
    if(i>=160)return;
    int block=i/32, within=i%32;
    const unsigned char *b=row+block*18;
    float d=__half2float(*reinterpret_cast<const __half*>(b));
    unsigned char packed=b[2+within%16];
    unsigned int q=within<16?(packed&15):(packed>>4);
    dst[i]=d*(float)iq4nl[q];
}

int main(int argc, char **argv){
    if (argc != 2) { fprintf(stderr,"usage: %s MODEL.gguf\n",argv[0]); return 2; }
    const char *path=argv[1];
    constexpr off_t tensor_off=536121120, page_off=536117248; constexpr size_t in_page=tensor_off-page_off;
    HIP_OK(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP_OK(hipGetDeviceProperties(&prop,0));
    unsigned char *host=nullptr; HIP_OK(hipHostMalloc((void**)&host,4096,hipHostMallocMapped|hipHostMallocCoherent));
    void *gpu_alias=nullptr; HIP_OK(hipHostGetDevicePointer(&gpu_alias,host,0));
    int fd=open(path,O_RDONLY|O_DIRECT|O_CLOEXEC);if(fd<0){perror("open O_DIRECT");return 11;}
    ssize_t got=pread(fd,host,4096,page_off);close(fd);if(got!=4096){perror("pread PLE page");return 12;}
    float cpu[160]; dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl*>(host+in_page),cpu,160);
    float *gpu_out=nullptr; HIP_OK(hipMalloc((void**)&gpu_out,sizeof(cpu)));
    hipLaunchKernelGGL(dequant_iq4nl,dim3(1),dim3(160),0,0,(const unsigned char*)gpu_alias+in_page,gpu_out);
    HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());float gpu[160];HIP_OK(hipMemcpy(gpu,gpu_out,sizeof(gpu),hipMemcpyDeviceToHost));
    float maxerr=0;int mismatches=0;for(int i=0;i<160;i++){float e=std::fabs(cpu[i]-gpu[i]);maxerr=std::max(maxerr,e);mismatches+=e>1e-6f;}
    printf("gpu=%s arch=%s source=O_DIRECT PLE IQ4_NL row0 bytes=%zd row_bytes=90 max_abs_error=%g mismatches=%d/160\n",prop.name,prop.gcnArchName,got,maxerr,mismatches);
    HIP_OK(hipFree(gpu_out));HIP_OK(hipHostFree(host));return mismatches==0?0:13;
}
