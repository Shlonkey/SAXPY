
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <time.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t error = call;                                                     \
        if (error != cudaSuccess) {                                                   \
            fprintf(stderr, "********* CUDA Error: %s, File: %s, Line: %d *********n",\
                    cudaGetErrorString(error), __FILE__, __LINE__);                   \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

__global__ void SAXPY(long long* d_a, long long* d_b, int k, long long array_length)
{
    long long start_index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for(long long i = start_index; i < array_length; i+=stride)
        d_a[i] += k * d_b[i];
    return;
}

int FPUsPerSM(cudaDeviceProp device_properties)
{
    int major = device_properties.major;
    int minor = device_properties.minor;
    switch (major) {
    case 2: // Fermi
        if (minor == 1) return 48;
        else return 32;
    case 3: // Kepler
        return 192;
    case 5: // Maxwell
        return 128;
    case 6: // Pascal
        if ((minor == 1) || (minor == 2)) return 128;
        else if (minor == 0) return 64;
    case 7: // Volta and Turing
        if ((minor == 0) || (minor == 5)) return 64;
    case 8: // Ampere
        if (minor == 0) return 64;
        else if (minor == 6) return 128;
        else if (minor == 9) return 128; // ada lovelace
    case 9: // Hopper
        if (minor == 0) return 128;
    }
    return NULL;
}

enum arraytype
{
    SHORT,
    INT,
    LONG,
    LONGLONG,
    FLOAT,
    DOUBLE,
    LONGDOUBLE
};

struct program_run_infomation
{
    enum arraytype type_of_array;
    double mem_usage_fraction;
    bool time;
    unsigned int repetitions;
};

struct program_run_infomation default_program_run_information()
{
    struct program_run_infomation default_run_info = {INT, 0.9, false};
    return default_run_info;
}

int main(int argc, char* argv[])
{
    //argv can contain the following;
    //  arraytype : {S, I, L, LL, F, D, LD}
    //  --memuseagefration : 0.0 - 1.0
    //  --profile=(1-UNINTMAX)
    
    long long size_of_array_to_add = 1000000000;

    clock_t full_time_start, full_time_end, gpu_only_time_start, gpu_only_time_end, mem_cpy_start, mem_cpy_end;

    #ifndef OPTIMIZATION_O3
    printf("Allocating Host Memory\n");
    #endif

    //Asign variable
    long long* a = (long long*)malloc(sizeof(long long) * size_of_array_to_add);
    long long* b = (long long*)malloc(sizeof(long long) * size_of_array_to_add);
    long long* c = (long long*)malloc(sizeof(long long) * size_of_array_to_add);
    int k = 2;

    if (a == NULL || b == NULL || c == NULL) return -1;

    #ifndef OPTIMIZATION_O3
    printf("Assigning Host Memory\n\n");

    for (long long i = 0; i < size_of_array_to_add; i++)
    {
        if (i % 50000000 == 0)
        {
            printf("%lf %% complete\n", 100 * i / (double)size_of_array_to_add);
        }
        a[i] = i;
        b[i] = i;
    }
    #endif

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);

    int number_of_sms = device_properties.multiProcessorCount;
    
    int number_of_fpus_per_sm = FPUsPerSM(device_properties);
    int oversubscription_factor = 100;
    int max_threads_per_sm = device_properties.maxThreadsPerMultiProcessor;
    int max_threads_per_block = device_properties.maxThreadsPerBlock;

    int number_of_blocks = number_of_sms * max_threads_per_sm / max_threads_per_block;

    int number_of_threads_requested = (1 + oversubscription_factor) * number_of_fpus_per_sm;

    int number_of_threads_per_sm = MIN(max_threads_per_block, number_of_threads_requested);

    #ifndef OPTIMIZATION_O3
    printf("\nArray Size : %1.4lf * 10^9\nBlocks : %i\nThreads Per Block : %i\n\n",size_of_array_to_add / (double)1000000000, number_of_blocks, number_of_threads_per_sm);
    #endif

    full_time_start = clock();

    //define device pointers
    long long* d_a;
    long long* d_b;

    #ifndef OPTIMIZATION_O3
    printf("allocating device Memory\n");
    #endif

    //allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(long long) * size_of_array_to_add));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(long long) * size_of_array_to_add));

    #ifndef OPTIMIZATION_O3
    printf("copying %lf GB from Host to Device\n", sizeof(long long) * 2 * size_of_array_to_add / double(1024 * 1024 * 1024));
    #endif

    mem_cpy_start = clock();

    //cpy hist data to device
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(long long) * size_of_array_to_add, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(long long) * size_of_array_to_add, cudaMemcpyHostToDevice));

    mem_cpy_end = clock();

    #ifndef OPTIMIZATION_O3
    printf("%lf GB H->D Mem Cpy took %lf seconds\n\n", sizeof(long long) * 2 * size_of_array_to_add / double(1024 * 1024 * 1024), (double)(mem_cpy_end - mem_cpy_start) / CLOCKS_PER_SEC);
    #endif

    gpu_only_time_start = clock();

    #ifndef OPTIMIZATION_O3
    printf("Launching Kernel\n");
    #endif
    
    //launch kernel
    SAXPY<<<number_of_blocks, number_of_threads_per_sm >>>(d_a, d_b, k, size_of_array_to_add);

    //not strictly needed as 'cudamemcpy' runs on the default stream as does 'Kernel' and hence it waits by default however if another stream was used, it would be mandatory
    CUDA_CHECK(cudaDeviceSynchronize());

    gpu_only_time_end = clock();
    
    #ifndef OPTIMIZATION_O3
    printf("Kernel Complete\n\n");

    printf("copying %lf GB from Device to Host\n", sizeof(long long) * size_of_array_to_add / double(1024 * 1024 * 1024));
    #endif

    mem_cpy_start = clock();

    //read back data
    CUDA_CHECK(cudaMemcpy(c, d_a, sizeof(long long) * size_of_array_to_add, cudaMemcpyDeviceToHost));

    mem_cpy_end = clock();

    #ifndef OPTIMIZATION_O3
    printf("%lf GB D->H Mem Cpy took %lf seconds\n", sizeof(long long) * size_of_array_to_add / double(1024 * 1024 * 1024), (double)(mem_cpy_end - mem_cpy_start) / CLOCKS_PER_SEC);

    printf("Freeing Data from Device\n\n");
    #endif

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    full_time_end = clock();

    CUDA_CHECK(cudaDeviceReset());

    #ifndef OPTIMIZATION_O3
    printf("VALIDATING RESULT\n");
    for (long long i = 0; i < size_of_array_to_add; i++)
    {
        if (c[i] != (long long)(k + 1) * i)
        {
            printf("%lli != %lli\n", c[i], (long long)(k + 1) * i);
            printf("RESULT INVALID\n\n");
            return -1;
        }
    }
    printf("RESULT VALID\n\n");

    printf("Freeing Data from Host\n\n");
    #endif

    free(a);
    free(b);
    free(c);

    double cpu_gpu_time = (double)(full_time_end - full_time_start) / CLOCKS_PER_SEC;
    double gpu_time = (double)(gpu_only_time_end - gpu_only_time_start) / CLOCKS_PER_SEC;
    double cpu_time = cpu_gpu_time - gpu_time;
    
    #ifndef OPTIMIZATION_O3
    printf("CPU-GPU including cudaMalloc, cudaMemcpy and kernel : %lf\nGPU including kernel : %lf\nCPU including cudaMalloc and cudaMemcpy : %lf\nPercentage of time on Computation vs IO : %lf%%\n", cpu_gpu_time, gpu_time, cpu_time, gpu_time * 100 / cpu_time);
    #endif

    int number_of_active_threads_per_sm = MIN(number_of_fpus_per_sm, number_of_threads_requested);
    double percentage_of_fpus_used = 100 * number_of_active_threads_per_sm / (double)number_of_fpus_per_sm;
    double percentage_of_inactive_threads_used = 100 * (number_of_threads_per_sm - number_of_active_threads_per_sm) / (double)(max_threads_per_block - number_of_active_threads_per_sm);
    
    #ifndef OPTIMIZATION_O3
    printf("Number of Active Threads per SM : %i\nNumber of Active and Inactive Threads per SM : %i\nPercentage of FPUs used : %lf%%\nPercentage of Inactive Threads Used : %lf%%\nActive to Inactive Thread Ratio : (%i:%i)\n--------------------------------------------------------------------------------\n", number_of_active_threads_per_sm, number_of_threads_per_sm, percentage_of_fpus_used, percentage_of_inactive_threads_used, number_of_active_threads_per_sm, (number_of_threads_per_sm - number_of_active_threads_per_sm));
    #endif

    return 0;
}