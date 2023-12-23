
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <time.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CLAMP(a, min, max) MIN((MAX((a), (min))),(max))
#define MEMORY_ASSIGNMENT_FEEDBACK_FRACTION 0.1

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t error = call;                                                     \
        if (error != cudaSuccess) {                                                   \
            fprintf(stderr, "********* CUDA Error: %s, File: %s, Line: %d *********n",\
                    cudaGetErrorString(error), __FILE__, __LINE__);                   \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define CHECK_NOT_NEG(call) if(call < 0){fprintf(stderr, "*********Error: %llu, File: %s, Line: %d *********n",call, __FILE__, __LINE__);exit(1);}
#define CALC_TIME(start_time, end_time) ((double)((end_time) - (start_time)) / CLOCKS_PER_SEC)

#ifdef SHORT
    #define ARRAY_VARIABLE_TYPE short
#elif INT
    #define ARRAY_VARIABLE_TYPE int
#elif LONG
    #define ARRAY_VARIABLE_TYPE long
#elif LONGLONG
    #define ARRAY_VARIABLE_TYPE long long
#elif FLOAT
    #define ARRAY_VARIABLE_TYPE float
#elif DOUBLE
    #define ARRAY_VARIABLE_TYPE double
#elif LONGDOUBLE
    #define ARRAY_VARIABLE_TYPE long double
#else
    #define ARRAY_VARIABLE_TYPE short
#endif


__global__ void SAXPY(ARRAY_VARIABLE_TYPE* d_a, ARRAY_VARIABLE_TYPE* d_b, float k, unsigned long long array_length)
{
    unsigned long long start_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    for(unsigned long long i = start_index; i < array_length; i+=stride)
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

struct program_run_infomation
{
    double mem_usage_fraction;
    unsigned int profile;
    unsigned int oversubscription;
};

struct program_run_infomation default_program_run_information()
{
    struct program_run_infomation default_run_info = {0.9, 0, 0};
    return default_run_info;
}

void process_input_flag(char flag, char* assignment, struct program_run_infomation* program_info)
{
    switch(flag)
    {
        case 'm':
            program_info->mem_usage_fraction = atof(assignment);
            return;
        case 'p':
            program_info->profile = MAX(atoi(assignment), 0);
            return;
        case 's':
            program_info->oversubscription = MAX(atoi(assignment), 0);
            return;
    }
    printf("flag '%s' is not a valid flag. try running with the -h flag for a list of VALID flags.",assignment);exit(1);
}

//argv can contain the following;
//  memuseagefration -m : 0.0 - 1.0
//  profile -p : (0-UNINTMAX) --- should run through p times and calclate sd dev mean and such
//  oversubscription -s: (0-INTMAX)
int main(int argc, char* argv[])
{
    struct program_run_infomation run_info = default_program_run_information();

    for(int i = 1; i < argc; i++)
    {
        if(argv[i][0] == '-'){
            if(argv[i][1] != 'h'){process_input_flag(argv[i][1], argv[i+1], &run_info);i++;}
            else {printf("\nValid Flags:\n\t-a : arraytype {S, I, L, LL, F, D, LD}\n\t-m : memusagefraction {0.0-1.0}\n\t-p : profile {0, 1, 2, ...}\n\t-s : oversubscription {0, 1, 2, ...}\n\n");exit(1);}
        }
    }

    size_t size_of_list_element_bytes = sizeof(ARRAY_VARIABLE_TYPE);

    clock_t cpu_mem_alloc_time_start, cpu_mem_alloc_time_end;
    clock_t cpu_data_set_time_start, cpu_data_set_time_end;
    clock_t gpu_mem_alloc_time_start, gpu_mem_alloc_time_end;
    clock_t host_to_device_mem_copy_time_start, host_to_device_mem_copy_time_end;
    clock_t kernel_run_time_start, kernel_run_time_end;
    clock_t device_to_host_mem_copy_time_start, device_to_host_mem_copy_time_end;
    clock_t device_mem_free_time_start, device_mem_free_time_end;
    clock_t data_validation_time_start, data_validation_time_end;
    clock_t host_mem_free_time_start, host_mem_free_time_end;


    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);

    long long global_mem_on_gpu_bytes = device_properties.totalGlobalMem;
    unsigned long long size_of_array_to_add = global_mem_on_gpu_bytes * run_info.mem_usage_fraction / size_of_list_element_bytes;

    int number_of_sms = device_properties.multiProcessorCount;
    int number_of_fpus_per_sm = FPUsPerSM(device_properties);

    int max_threads_per_sm = device_properties.maxThreadsPerMultiProcessor;
    int max_threads_per_block = device_properties.maxThreadsPerBlock;
    int number_of_blocks = number_of_sms * max_threads_per_sm / max_threads_per_block;

    int number_of_threads_requested = (1 + run_info.oversubscription) * number_of_fpus_per_sm;
    int number_of_threads_per_block = MIN(max_threads_per_block, number_of_threads_requested);

    unsigned long long MEMORY_FEEDBACK_INDEX_SPLIT = floor(size_of_array_to_add * MEMORY_ASSIGNMENT_FEEDBACK_FRACTION);

    #ifndef OPTIMIZATION_O3
    printf("Allocating Host Memory\n");
    #endif

    //Asign variable
    if(run_info.profile > 0){cpu_mem_alloc_time_start = clock();}
    ARRAY_VARIABLE_TYPE* a = (ARRAY_VARIABLE_TYPE*)malloc(size_of_list_element_bytes * size_of_array_to_add);
    ARRAY_VARIABLE_TYPE* b = (ARRAY_VARIABLE_TYPE*)malloc(size_of_list_element_bytes * size_of_array_to_add);
    ARRAY_VARIABLE_TYPE* c = (ARRAY_VARIABLE_TYPE*)malloc(size_of_list_element_bytes * size_of_array_to_add);
    int k = 2;

    if (a == NULL || b == NULL || c == NULL){printf("NULL POINTER\na : %p\nb : %p\nc : %p", a, b, c);return -1;}

    #ifndef OPTIMIZATION_O3
    printf("Assigning Host Memory\n\n");
    #endif

    //set host data
    if(run_info.profile > 0){cpu_data_set_time_start = clock();}
    for (unsigned long long i = 0; i < size_of_array_to_add; i++)
    {

        #ifndef OPTIMIZATION_O3
        if (i % MEMORY_FEEDBACK_INDEX_SPLIT == 0)
        {
            printf("%lf %% complete\n", 100 * i / (double)size_of_array_to_add);
        }
        #endif
        a[i] = i;
        b[i] = i;
    }
    if(run_info.profile > 0){cpu_data_set_time_end = clock();}

    #ifndef OPTIMIZATION_O3
    printf("\nArray Size : %1.4lf * 10^9\nBlocks : %i\nThreads Per Block : %i\n\n",size_of_array_to_add / (double)1000000000, number_of_blocks, number_of_threads_per_block);
    #endif

    //define device pointers
    ARRAY_VARIABLE_TYPE* d_a;
    ARRAY_VARIABLE_TYPE* d_b;

    #ifndef OPTIMIZATION_O3
    printf("allocating device Memory\n");
    #endif

    //allocate device memory
    if(run_info.profile > 0){gpu_mem_alloc_time_start = clock();}
    CUDA_CHECK(cudaMalloc(&d_a, size_of_list_element_bytes * size_of_array_to_add));
    CUDA_CHECK(cudaMalloc(&d_b, size_of_list_element_bytes * size_of_array_to_add));
    if(run_info.profile > 0){gpu_mem_alloc_time_end = clock();}

    #ifndef OPTIMIZATION_O3
    printf("copying %lf GB from Host to Device\n", size_of_list_element_bytes * 2 * size_of_array_to_add / double(1024 * 1024 * 1024));
    #endif

    //cpy hist data to device
    if(run_info.profile > 0){host_to_device_mem_copy_time_start = clock();}
    CUDA_CHECK(cudaMemcpy(d_a, a, size_of_list_element_bytes * size_of_array_to_add, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, size_of_list_element_bytes * size_of_array_to_add, cudaMemcpyHostToDevice));
    if(run_info.profile > 0){host_to_device_mem_copy_time_end = clock();}

    #ifndef OPTIMIZATION_O3
    printf("Launching Kernel\n");
    #endif
    
    //launch kernel
    if(run_info.profile > 0){kernel_run_time_start = clock();}
    SAXPY<<<number_of_blocks, number_of_threads_per_block >>>(d_a, d_b, k, size_of_array_to_add);

    //not strictly needed as 'cudamemcpy' runs on the default stream as does 'Kernel' and hence it waits by default however if another stream was used, it would be mandatory
    CUDA_CHECK(cudaDeviceSynchronize());
    if(run_info.profile > 0){kernel_run_time_end = clock();}
    
    #ifndef OPTIMIZATION_O3
    printf("Kernel Complete\n\n");
    printf("copying %lf GB from Device to Host\n", size_of_list_element_bytes * size_of_array_to_add / double(1024 * 1024 * 1024));
    #endif

    //read back data
    if(run_info.profile > 0){device_to_host_mem_copy_time_start = clock();}
    CUDA_CHECK(cudaMemcpy(c, d_a, size_of_list_element_bytes * size_of_array_to_add, cudaMemcpyDeviceToHost));
    if(run_info.profile > 0){device_to_host_mem_copy_time_end = clock();}

    #ifndef OPTIMIZATION_O3
    printf("Freeing Data from Device\n\n");
    #endif

    if(run_info.profile > 0){device_mem_free_time_start = clock();}
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaDeviceReset());
    if(run_info.profile > 0){device_mem_free_time_end = clock();}
    
    //VALIDATE RESULT
    if(run_info.profile > 0){data_validation_time_start = clock();}
    #ifndef OPTIMIZATION_O3
    printf("VALIDATING RESULT\n");
    #endif
    for (unsigned long long i = 0; i < size_of_array_to_add; i++)
    {
        if (c[i] != (ARRAY_VARIABLE_TYPE)((k + 1) * i))
        {
            //can't do this since format strings suck. if needed, make a function to return a format string given a var type
            printf("%i != %i\n", c[i], (ARRAY_VARIABLE_TYPE)((k + 1) * i));
            printf("RESULT INVALID\n\n");
            return -1;
        }
    }
    if(run_info.profile > 0){data_validation_time_end = clock();}
    //END VALIDATE RESULT

    #ifndef OPTIMIZATION_O3
    printf("RESULT VALID\n\n");
    printf("Freeing Data from Host\n\n");
    #endif

    if(run_info.profile > 0){host_mem_free_time_start = clock();} 
    free(a);
    free(b);
    free(c);
    if(run_info.profile > 0){host_mem_free_time_end = clock();} 

    if(run_info.profile > 0)
    {
        int number_of_active_threads_per_sm = MIN(number_of_fpus_per_sm, number_of_threads_requested);
        double percentage_of_fpus_used = 100 * number_of_active_threads_per_sm / (double)number_of_fpus_per_sm;
        double percentage_of_inactive_threads_used = 100 * (number_of_threads_per_block - number_of_active_threads_per_sm) / (double)(max_threads_per_block - number_of_active_threads_per_sm);
        
        printf("Number of Active Threads per SM : %i\nNumber of Active and Inactive Threads per SM : %i\nPercentage of FPUs used : %lf%%\nPercentage of Inactive Threads Used : %lf%%\nActive to Inactive Thread Ratio : (%i:%i)\n", number_of_active_threads_per_sm, number_of_threads_per_block, percentage_of_fpus_used, percentage_of_inactive_threads_used, number_of_active_threads_per_sm, (number_of_threads_per_block - number_of_active_threads_per_sm));
        printf("\n----------TIMINGS----------\n\n");
        printf("");
    }
    return 0;
}