#ifndef PROJ_CORE_ORIG
#define PROJ_CORE_ORIG
#include "ProjHelperFun.cu.h"
#include "ProjKernels.cu.h"
#include "Constants.h"

inline void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
) {
    int    i;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}

template<const unsigned T2D,const unsigned T3D>
void   run_optimGPU(
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t,
                const REAL&           alpha,
                const REAL&           nu,
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
) {

    unsigned long int e_all, e_initGrid, e_initOp, e_payoff, e_update, e_eX, e_eY, e_iX, e_triX,
        e_iY, e_triY, e_res, e_cudaMalloc, e_free;
    e_initGrid = 0; e_initOp = 0; e_payoff = 0; e_update = 0;
    e_eX = 0; e_eY = 0; e_iX = 0; e_triX = 0; e_iY = 0;
    e_triY = 0; e_res = 0; e_cudaMalloc = 0;
    struct timeval t_start, t_end, t_diff, t_start_all, t_end_all;
    gettimeofday(&t_start_all, NULL);
    // Generate vector of globs. Initialize grid and operators
    gettimeofday(&t_start, NULL);
    DevicePrivGlobs d_globs(outer, numX, numY, numT);

    deviceInitGrid<T2D>(s0, alpha, nu, t, outer, numX, numY, numT, d_globs);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    e_initGrid += (t_diff.tv_sec*1e6+t_diff.tv_usec);

    gettimeofday(&t_start, NULL);
    deviceInitOperator<T2D>(outer, numX, d_globs.myX, d_globs.myDxx);
    deviceInitOperator<T2D>(outer, numY, d_globs.myY, d_globs.myDyy);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    e_initOp += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    /**
     * setPayoff function (and strike)
     */
    gettimeofday(&t_start, NULL);
    deviceSetPayoff<T2D>(outer, numX, numY, d_globs);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    e_payoff += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    // end setPayoff

    // Arrays for rollback:
    gettimeofday(&t_start, NULL);
    REAL* d_a, *d_b, *d_c, *d_y, *d_yy; // [outer][max(numX,numY)]
    REAL *d_v, *d_u, *d_tu;
    REAL *d_ta, *d_tb, *d_tc;
    int mem_full_size = sizeof(REAL)*outer*numX*numY;
    cudaMalloc((void**) &d_a,  mem_full_size);
    cudaMalloc((void**) &d_b,  mem_full_size);
    cudaMalloc((void**) &d_c,  mem_full_size);
    cudaMalloc((void**) &d_ta, mem_full_size);
    cudaMalloc((void**) &d_tb, mem_full_size);
    cudaMalloc((void**) &d_tc, mem_full_size);
    cudaMalloc((void**) &d_y,  mem_full_size);
    cudaMalloc((void**) &d_yy, mem_full_size);
    cudaMalloc((void**) &d_u,  mem_full_size);
    cudaMalloc((void**) &d_tu, mem_full_size);
    cudaMalloc((void**) &d_v,  mem_full_size);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    e_cudaMalloc += (t_diff.tv_sec*1e6+t_diff.tv_usec);
    REAL * timeline = (REAL*) malloc(sizeof(REAL)*numT);
    cudaMemcpy(timeline,d_globs.myTimeline,sizeof(REAL)*numT,cudaMemcpyDeviceToHost);

    // Value function inserted:
    for(int t = numT-2;t>=0;--t)
    {
        /**
         * updateParams function
         */
        REAL time = timeline[t];
        gettimeofday(&t_start, NULL);
        deviceUpdateParams<T3D>(outer, numX, numY, alpha, beta, nu, time, d_globs);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_update += (t_diff.tv_sec*1e6+t_diff.tv_usec);
        // end updateParams

        /**
         * Rollback function
         */
        REAL dtInv = 1.0/(timeline[t+1]-timeline[t]);

        gettimeofday(&t_start, NULL);
        // explicit X
        explicitX<T3D>(outer, numX, numY, dtInv, d_tu, d_globs);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_eX += (t_diff.tv_sec*1e6+t_diff.tv_usec);

        gettimeofday(&t_start, NULL);
        // explicit Y
        explicitY<T3D>(outer, numX, numY, dtInv, d_v, d_tu, d_globs);
        sgmMatTranspose<T2D>(outer, numX, numY, d_tu, d_u);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_eY += (t_diff.tv_sec*1e6+t_diff.tv_usec);

        gettimeofday(&t_start, NULL);
        // implicit X
        deviceImplicitX<T3D>(outer, numX, numY, dtInv, d_globs, d_ta, d_tb, d_tc);
        sgmMatTranspose<T2D>(outer, numX, numY, d_ta, d_a);
        sgmMatTranspose<T2D>(outer, numX, numY, d_tb, d_b);
        sgmMatTranspose<T2D>(outer, numX, numY, d_tc, d_c);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_iX += (t_diff.tv_sec*1e6+t_diff.tv_usec);

        gettimeofday(&t_start, NULL);
        // tridag X
        deviceTridag<T2D*T2D/2>(d_a,d_b,d_c,d_u,outer*numX*numY,numX,d_u,d_yy);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_triX += (t_diff.tv_sec*1e6+t_diff.tv_usec);

        gettimeofday(&t_start, NULL);
        // implicit Y
        sgmMatTranspose<T2D>(outer, numY, numX, d_u, d_tu);
        deviceImplicitY<T3D>(outer, numX, numY, dtInv, d_globs,
                d_a, d_b, d_c, d_tu, d_v, d_y);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_iY += (t_diff.tv_sec*1e6+t_diff.tv_usec);

        gettimeofday(&t_start, NULL);
        // tridag Y
        deviceTridag<T2D*T2D/2>(d_a,d_b,d_c,d_y,outer*numX*numY,numY,d_globs.myResult,d_yy);
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        e_triY = e_triY + (t_diff.tv_sec*1e6+t_diff.tv_usec);

        if (t == 0) {
            gettimeofday(&t_start, NULL);
            deviceResult<T2D*T2D>(outer,numX,numY,d_globs,res);
            gettimeofday(&t_end, NULL);
            timeval_subtract(&t_diff, &t_end, &t_start);
            e_res += (t_diff.tv_sec*1e6+t_diff.tv_usec);
        }
        // End value function
    }
    gettimeofday(&t_start, NULL);
    free(timeline);
    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_c); cudaFree(d_y);
    cudaFree(d_yy);
    cudaFree(d_ta); cudaFree(d_tb); cudaFree(d_tc);
    cudaFree(d_u); cudaFree(d_tu); cudaFree(d_v);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    e_free = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    gettimeofday(&t_end_all, NULL);
    timeval_subtract(&t_diff, &t_end_all, &t_start_all);
    e_all = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("initGrid:\t%lu\ninitOperator:\t%lu\ncudaMalloc:\t%lu\nsetPayoff:\t%lu\n\
updateParams:\t%lu\nexplicitX:\t%lu\nexplicitY:\t%lu\nimplicitX:\t%lu\n\
tridagX:\t%lu\nimplicitY:\t%lu\ntridagY:\t%lu\nres copy:\t%lu\nclean-up:\t%lu\nAll code:\t%lu\n",
            e_initGrid, e_initOp, e_cudaMalloc, e_payoff, e_update, e_eX, e_eY, e_iX, e_triX,
            e_iY, e_triY, e_res, e_free, e_all);
}

#endif // PROJ_CORE_ORIG
