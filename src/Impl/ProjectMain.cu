#include "OpenmpUtil.h"
#include "ParseInput.h"
#include <cuda_runtime.h>

#include "ProjHelperFun.cu.h"
#include "ProjCoreOrig.cu.h"
#include "ProjKernels.cu.h"

int main()
{
    unsigned int OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T;
	const REAL s0 = 0.03, /*strike = 0.03,*/ t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T );

    const int Ps = get_CPU_num_threads();
    const unsigned T2D = 16;
    const unsigned T3D = 8;
    REAL* res = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));

    cudaFree(0);
    {   // Original Program (Sequential CPU Execution)
        cout<<"\n// Running Optimized, Parallel Project Program"<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_optimGPU<T2D,T3D>( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

        // validation and writeback of the result
        bool is_valid = validate   ( res, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res, OUTER_LOOP_COUNT,
                             NUM_X, NUM_Y, NUM_T, false, Ps, elapsed );
    }

    return 0;
}

