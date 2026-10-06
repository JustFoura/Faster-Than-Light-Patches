#include <hip/hip_runtime.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,hipGetErrorString(e));return 10;} } while(0)

__global__ void checksum_page(const unsigned char * bytes, unsigned long long * out) {
    unsigned long long sum=0;
    for (int i=threadIdx.x; i<4096; i+=blockDim.x) sum += (unsigned long long) bytes[i] * (unsigned long long)(i+1);
    atomicAdd(out,sum);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr,"usage: %s MODEL.gguf\n",argv[0]); return 2; }
    const char *path=argv[1];
    HIP_OK(hipSetDevice(0));
    hipDeviceProp_t prop{}; HIP_OK(hipGetDeviceProperties(&prop,0));
    unsigned char *host=nullptr; HIP_OK(hipHostMalloc((void**)&host,4096,hipHostMallocMapped|hipHostMallocCoherent));
    void *dev_alias=nullptr; HIP_OK(hipHostGetDevicePointer(&dev_alias,host,0));
    if (((uintptr_t)host & 4095) || ((uintptr_t)dev_alias & 255)) { fprintf(stderr,"bad alignment host=%p device=%p\n",host,dev_alias); return 11; }
    int fd=open(path,O_RDONLY|O_DIRECT|O_CLOEXEC); if(fd<0){perror("open O_DIRECT");return 12;}
    ssize_t n=pread(fd,host,4096,536117248LL); close(fd); if(n!=4096){perror("pread to HIP pinned buffer");return 13;}
    unsigned long long expected=0;for(int i=0;i<4096;i++)expected+=(unsigned long long)host[i]*(unsigned long long)(i+1);
    unsigned long long *out=nullptr; HIP_OK(hipMalloc((void**)&out,sizeof(*out))); HIP_OK(hipMemset(out,0,sizeof(*out)));
    hipLaunchKernelGGL(checksum_page,dim3(1),dim3(256),0,0,(const unsigned char*)dev_alias,out);
    HIP_OK(hipGetLastError()); HIP_OK(hipDeviceSynchronize());
    unsigned long long got=0; HIP_OK(hipMemcpy(&got,out,sizeof(got),hipMemcpyDeviceToHost));
    printf("gpu=%s arch=%s host_ptr=%p gpu_alias=%p O_DIRECT_bytes=%zd checksum_gpu=%llu checksum_cpu=%llu match=%s\n",prop.name,prop.gcnArchName,host,dev_alias,n,got,expected,got==expected?"yes":"no");
    HIP_OK(hipFree(out)); HIP_OK(hipHostFree(host)); return got==expected?0:14;
}
