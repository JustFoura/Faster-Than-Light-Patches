#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
int main(int argc,char**argv){
 if(argc!=2){fprintf(stderr,"usage: %s MODEL.gguf\n",argv[0]);return 2;}
 const char*p=argv[1];
 const size_t start=536117248, len=28800138240ULL, page=4096, pages=len/page, n=2048;
 int fd=open(p,O_RDONLY|O_CLOEXEC); if(fd<0){perror("open");return 2;} posix_fadvise(fd,start,len,POSIX_FADV_DONTNEED);
 void*map=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,start);if(map==MAP_FAILED){perror("mmap");return 3;}
 unsigned char*v=calloc(pages,1);void*buf=NULL;if(posix_memalign(&buf,page,page))return 4;
 int dfd=open(p,O_RDONLY|O_DIRECT|O_CLOEXEC);if(dfd<0){perror("O_DIRECT open");return 5;}
 srandom(8260926);struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);
 for(size_t i=0;i<n;i++){size_t ix=(size_t)random()%pages;ssize_t got=pread(dfd,buf,page,start+ix*page);if(got!=page){perror("O_DIRECT pread");return 6;}}
 clock_gettime(CLOCK_MONOTONIC,&b);int rc=mincore(map,len,v);if(rc){perror("mincore");return 7;}size_t resident=0;for(size_t i=0;i<pages;i++)resident+=v[i]&1;
 printf("reads=%zu bytes=%zu elapsed=%.3fs resident_pages=%zu/%zu\n",n,n*page,(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9,resident,pages);
 close(dfd);free(buf);free(v);munmap(map,len);close(fd);return 0;
}
