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
template<const unsigned T>
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
    // Generate vector of globs. Initialize grid and operators onces
    // and make default element of vector
    // Hoisted from "value"
    // CPU:
    PrivGlobs    globs(numX, numY, numT);
    DevicePrivGlobs d_globs(outer, numX, numY, numT);
    initGrid(s0, alpha, nu, t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);
    vector<PrivGlobs> globArr (outer, globs);
    // GPU:
    deviceInitGrid<T>(s0, alpha, nu, t, outer, numX, numY, numT, d_globs); 
    deviceInitOperator<T>(outer, numX, d_globs.myX, d_globs.myDxx);
    deviceInitOperator<T>(outer, numY, d_globs.myY, d_globs.myDyy);
    REAL* line = (REAL*) malloc(sizeof(REAL)*numT);
    REAL* myX = (REAL*) malloc(sizeof(REAL)*numX);
    REAL* myY = (REAL*) malloc(sizeof(REAL)*numY);
    REAL* myDxx = (REAL*) malloc(sizeof(REAL)*numX*4);
    REAL* myDyy = (REAL*) malloc(sizeof(REAL)*numY*4);
    cudaMemcpy(line,d_globs.myTimeline, sizeof(REAL)*numT,cudaMemcpyDeviceToHost);
    cudaMemcpy(myX,d_globs.myX, sizeof(REAL)*numX,cudaMemcpyDeviceToHost);
    cudaMemcpy(myY,d_globs.myY, sizeof(REAL)*numY,cudaMemcpyDeviceToHost);
    cudaMemcpy(myDxx,d_globs.myDxx, sizeof(REAL)*numX*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(myDyy,d_globs.myDyy, sizeof(REAL)*numY*4,cudaMemcpyDeviceToHost);
    bool succes = true;
    printf("timeline:\n");
    for (int i = 0; i < numT; i++) {
        if (abs(line[i]-globs.myTimeline[i]) > 1e-6) {
            printf("WRONG! %i: %f != %f\n",i,line[i],globs.myTimeline[i]);
            succes = false;
            break;
        }
    }
    printf("myX:\n");
    for (int i = 0; i < numX; i++) {
        if (abs(myX[i]-globs.myX[i]) > 1e-6) {
            printf("WRONG! %i: %f != %f\n",i,myX[i],globs.myX[i]);
            succes = false;
            break;
        }
    }
    printf("myY:\n");
    for (int i = 0; i < numY; i++) {
        if (abs(myY[i]-globs.myY[i]) > 1e-6) {
            printf("WRONG! %i: %f != %f\n",i,myY[i],globs.myY[i]);
            succes = false;
            break;
        }
    }
    printf("myDxx:\n");
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < 4; j++) {
        if (abs(myDxx[i*4+j]-globs.myDxx[i][j]) > 1) {
            printf("WRONG! %i,%i: %f != %f\n",i,j,myDxx[i*4+j],globs.myDxx[i][j]);
            succes = false;
            break;
        }
        }
    }
    printf("myDyy:\n");
    for (int i = 0; i < numY; i++) {
        for (int j = 0; j < 4; j++) {
        if (abs(myDyy[i*4+j]-globs.myDyy[i][j]) > 1) {
            printf("WRONG! %i,%i: %f != %f\n",i,j,myDyy[i*4+j],globs.myDyy[i][j]);
            succes = false;
            break;
        }
        }
    }
    if (succes) {
        printf("Globs init well done!\n");
    } else {
        printf("Something something dark side!\n");
    }

    unsigned numZ = max(numX,numY);

    /**
     * setPayoff function (and strike)
     */
    // Make 3D kernel for computing payoff for each outer loop
    // x: globs.myX, y: globs.myY, z: outer
    #pragma omp parallel for default(shared) schedule(static) if(outer>8)
    for(unsigned i = 0; i < outer; ++ i) {
        for(unsigned j=0;j<globArr[i].myX.size();++j)
        {
            for(unsigned k=0;k<globArr[i].myY.size();++k)
            {
                // This could be computed once for each i,j, put in shared memory and
                // read for each k
                // Overhead of computing 0.001*i might be lower than accessing it in memory
                globArr[i].myResult[j][k] = max(globArr[i].myX[j]-0.001*i, (REAL)0.0); // privatized one level in
            }
        }
    }
    // end setPayoff
    // GPU:
    deviceSetPayoff<T>(outer, numX, numY, d_globs);

    REAL* myResult = (REAL*) malloc(sizeof(REAL)*numX*numY);
    cudaMemcpy(myResult,d_globs.myResult,sizeof(REAL)*numX*numY,cudaMemcpyDeviceToHost);
    for(unsigned j = 0;j<numX;++j) {
        for(unsigned k = 0;k<numY;++k) {
            if (abs(globArr[0].myResult[j][k] - myResult[j*numY+k]) > 1e-6) {
                printf("WRONG!\n");
            }
        }
    }

    // Value function inserted:
    for(int t = numT-2;t>=0;--t)
    {
        // Adding CPU parallelization
        /**
         * updateParams function
         */
        REAL time = globs.myTimeline[t];
        // Make 3D kernel for computing update of params for each outer loop
        // x: globs.myX, y: globs.myY, z: outer
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            for(unsigned j=0;j<globArr[i].myX.size();++j) {
                for(unsigned k=0;k<globArr[i].myY.size();++k) {
                    globArr[i].myVarX[j][k] = exp(2.0*(  beta*log(globArr[i].myX[j])
                                + globArr[i].myY[k]
                                - 0.5*nu*nu*time )
                            );
                    globArr[i].myVarY[j][k] = exp(2.0*(  alpha*log(globArr[i].myX[j])
                                + globArr[i].myY[k]
                                - 0.5*nu*nu*time )
                            ); // nu*nu
                }
            }
        }
        // end updateParams

        /**
         * Rollback function
         */
        REAL dtInv = 1.0/(globs.myTimeline[t+1]-globs.myTimeline[t]);
        vector<vector<vector<REAL> > > u(outer,vector<vector<REAL> >(numY, vector<REAL>(numX)));   // [outer][numY][numX]
        vector<vector<vector<REAL> > > v(outer,vector<vector<REAL> >(numX, vector<REAL>(numY)));   // [outer][numX][numY]
        vector<vector<vector<REAL> > > a(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ))), b(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ))), c(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ))), y(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ)));     // [outer][max(numX,numY)]
        vector<vector<REAL> > yy(outer,vector<REAL>(numZ));  // temporary used in tridag  // [outer][max(numX,numY)]

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            unsigned j,k;
            //	explicit x
            for(j=0;j<numX;j++) {
                for(k=0;k<numY;k++) {
                    u[i][k][j] = dtInv*globArr[i].myResult[j][k];

                    if(j > 0) {
                        u[i][k][j] += 0.5*( 0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][0] )
                            * globArr[i].myResult[j-1][k];
                    }
                    u[i][k][j]  +=  0.5*( 0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][1] )
                        * globArr[i].myResult[j][k];
                    if(j < numX-1) {
                        u[i][k][j] += 0.5*( 0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][2] )
                            * globArr[i].myResult[j+1][k];
                    }
                }
            }
        }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            unsigned j, k;
            //	explicit y
            for(k=0;k<numY;k++)
            {
                for(j=0;j<numX;j++) {
                    v[i][j][k] = 0.0;

                    if(k > 0) {
                        v[i][j][k] +=  ( 0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][0] )
                            *  globArr[i].myResult[j][k-1];
                    }
                    v[i][j][k]  +=   ( 0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][1] )
                        *  globArr[i].myResult[j][k];
                    if(k < numY-1) {
                        v[i][j][k] +=  ( 0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][2] )
                            *  globArr[i].myResult[j][k+1];
                    }
                    u[i][k][j] += v[i][j][k];
                }
            }
        }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            unsigned j, k;
            //	implicit x
            for(k=0;k<numY;k++) {
                for(j=0;j<numX;j++) {  // here a, b,c should have size [numX]
                    a[i][k][j] =		 - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][0]);
                    b[i][k][j] = dtInv - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][1]);
                    c[i][k][j] =		 - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][2]);
                }
            }
        }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            unsigned k;
            //	implicit x
            for(k=0;k<numY;k++) {
                // here yy should have size [numY][numX]
                tridag(a[i][k],b[i][k],c[i][k],u[i][k],numX,u[i][k],yy[i]);
            }
        }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++i ) {
            unsigned j, k;

            //	implicit y
            for(j=0;j<numX;j++) {
                for(k=0;k<numY;k++) {  // here a, b, c should have size [numX][numY]
                    a[i][j][k] =		 - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][0]);
                    b[i][j][k] = dtInv - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][1]);
                    c[i][j][k] =		 - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][2]);
                }
            }
        }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++i ) {
            unsigned j,k;
            //	implicit y
            for(j=0;j<numX;j++) {
                for(k=0;k<numY;k++) {
                    y[i][j][k] = dtInv*u[i][k][j] - 0.5*v[i][j][k];
                }
            }
        }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++i ) {
            unsigned j;

            //	implicit y
            for(j=0;j<numX;j++) {
                // here yy should have size [numY]
                tridag(a[i][j],b[i][j],c[i][j],y[i][j],numY,globArr[i].myResult[j],yy[i]);
            }
            // end rollback function

            if (t == 0) {
                res[i] = globArr[i].myResult[globArr[i].myXindex][globArr[i].myYindex];
            }
        }
        // End value function
    }
}

#endif // PROJ_CORE_ORIG
