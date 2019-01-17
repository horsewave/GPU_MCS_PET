/*
 * gpuErrchk.cuh
 *
 *  Created on: Jul 21, 2018
 *      Author: bma
 */

#ifndef GPUERRCHK_CUH_
#define GPUERRCHK_CUH_



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//common function: example: gpuErrchk( cudaMalloc(&a_d, size*sizeof(int)) );

//kernel function examle:
//kernel<<<1,1>>>(a);
//gpuErrchk( cudaPeekAtLastError() );
//gpuErrchk( cudaDeviceSynchronize() );
//will firstly check for invalid launch argument, then force the host to wait until the kernel stops
//and checks for an execution error. The synchronisation can be eliminated if you have a subsequent blocking
//API call like this:



#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)



#endif /* GPUERRCHK_CUH_ */
