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

    /*
void globs2gpu(vector<PrivGlobs> &globs, unsigned outer, unsigned numX, unsigned numY, DevicePrivGlobs &d_globs) {
    REAL* local = (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
    for (int i = 0; i < outer; i++) {
        for (int j = 0; j < numX; j++) {
            memcpy(&dst[i*numX*numY+j*numY],&local[i*numX*numY+j*numY], sizeof(REAL)*numY);
            cudaMemcpy(&dst[i*numX*numY+j*numY],&globs[i].[j][0], sizeof(REAL)*numY,cudaMemcpyDeviceToHost);
        }
    }
    for (int i = 0; i < outer; i++) {
        for (int j = 0; j < numX; j++) {
            for (int k = 0; k < numX; k++) {
            }
        }
    }
}
*/
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
    deviceInitGrid<T2D>(s0, alpha, nu, t, outer, numX, numY, numT, d_globs);
    deviceInitOperator<T2D>(outer, numX, d_globs.myX, d_globs.myDxx);
    deviceInitOperator<T2D>(outer, numY, d_globs.myY, d_globs.myDyy);

    // Test code:
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
    free(line); free(myX); free(myY); free(myDxx); free(myDyy);
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
    deviceSetPayoff<T3D>(outer, numX, numY, d_globs);

    REAL* myResult = (REAL*) malloc(sizeof(REAL)*numX*numY);
    cudaMemcpy(myResult,d_globs.myResult,sizeof(REAL)*numX*numY,cudaMemcpyDeviceToHost);
    for(unsigned j = 0;j<numX;++j) {
        for(unsigned k = 0;k<numY;++k) {
            if (abs(globArr[0].myResult[j][k] - myResult[j*numY+k]) > 1e-6) {
                printf("Payoff WRONG! %i,%i: %f != %f\n",j,k,myResult[j*numY+k],globArr[0].myResult[j][k]);
            }
        }
    }

    // Arrays for rollback:
    REAL* d_a, *d_b, *d_c, *d_y, *d_yy; // [outer][max(numX,numY)]
    REAL *d_v, *d_u;
    cudaMalloc((void**) &d_a, sizeof(REAL)*outer*numX*numY);
    cudaMalloc((void**) &d_b, sizeof(REAL)*outer*numX*numY);
    cudaMalloc((void**) &d_c, sizeof(REAL)*outer*numX*numY);
    cudaMalloc((void**) &d_y, sizeof(REAL)*outer*numX*numY);
    cudaMalloc((void**) &d_yy, sizeof(REAL)*outer*numX*numY);
    cudaMalloc((void**) &d_u, sizeof(REAL)*outer*numX*numY);
    cudaMalloc((void**) &d_v, sizeof(REAL)*outer*numY*numX);

    // Value function inserted:
    for(int t = numT-2;t>=0;--t)
    {
        // Adding CPU parallelization
        /**
         * updateParams function
         */
        REAL time = globs.myTimeline[t];
        cpGlob2Gpu(globArr,outer,numX,numY,numT,d_globs);
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
        deviceUpdateParams<T3D>(outer, numX, numY, alpha, beta, nu, time, d_globs);

        REAL* myVarX = (REAL*) malloc(sizeof(REAL)*numX*numY);
        cudaMemcpy(myVarX,d_globs.myVarX,sizeof(REAL)*numX*numY,cudaMemcpyDeviceToHost);
        REAL* myVarY = (REAL*) malloc(sizeof(REAL)*numX*numY);
        cudaMemcpy(myVarY,d_globs.myVarY,sizeof(REAL)*numX*numY,cudaMemcpyDeviceToHost);
        for(unsigned j = 0;j<numX;++j) {
            for(unsigned k = 0;k<numY;++k) {
                if (abs(globArr[0].myVarX[j][k] - myVarX[j*numY+k]) > 1e-1) {
                    printf("Update params WRONG! %i,%i: %f != %f\n",j,k,myVarX[j*numY+k],globArr[0].myVarX[j][k]);
                    succes = false;
                }
            }
        }
        if (!succes) { break; }
        /**
         * Rollback function
         */
        REAL dtInv = 1.0/(globs.myTimeline[t+1]-globs.myTimeline[t]);
        vector<vector<vector<REAL> > > u(outer,vector<vector<REAL> >(numY, vector<REAL>(numX)));   // [outer][numY][numX]
        vector<vector<vector<REAL> > > v(outer,vector<vector<REAL> >(numX, vector<REAL>(numY)));   // [outer][numX][numY]
        vector<vector<vector<REAL> > > a(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ))), b(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ))), c(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ))), y(outer,vector<vector<REAL> >(numZ, vector<REAL>(numZ)));     // [outer][max(numX,numY)]
        vector<vector<REAL> > yy(outer,vector<REAL>(numZ));  // temporary used in tridag  // [outer][max(numX,numY)]

        // GPU version
        cpGlob2Gpu(globArr,outer,numX,numY,numT,d_globs); // made for copying of globs
        explicitX<T3D>(outer, numX, numY, dtInv, d_u, d_globs);
        REAL* h_v = (REAL*) malloc(sizeof(REAL)*outer*numY*numX);
        REAL* h_u = (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
        cudaMemcpy(h_u,d_u, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);

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
        succes = true;
        for (unsigned i=0;i<outer;i++) {
            for(unsigned j=0;j<numX;j++) {
                for(unsigned k=0;k<numY;k++) {
                    if (abs(u[i][k][j] - h_u[i*numY*numX + k*numX + j]) > 1e-5) {
                        printf("ExplicitX: %i,%i,%i. %f != %f\n", i, k, j, h_u[i*numY*numX + k*numX + j], u[i][k][j]);
                        succes = false;
                    }
                }
            }
        }
        if (!succes) { break;  }

        cpGlob2Gpu(globArr,outer,numX,numY,numT,d_globs); // made for copying of globs
        explicitY<T3D>(outer, numX, numY, dtInv, d_v, d_u, d_globs);
        cudaMemcpy(h_v,d_v, sizeof(REAL)*outer*numY*numX,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_u,d_u, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
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
        succes = true;
        for (unsigned i=0;i<outer;i++) {
            for(unsigned j=0;j<numX;j++) {
                for(unsigned k=0;k<numY;k++) {
                    if (abs(u[i][k][j] - h_u[i*numY*numX + k*numX + j]) > 1e-5) {
                        printf("ExplicitY u: %i,%i,%i. %f != %f\n", i, k, j, h_u[i*numY*numX + k*numX + j], u[i][k][j]);
                        succes = false;
                    }
                    if (abs(v[i][j][k] - h_v[i*numY*numX + j*numY+k]) > 1e-6) {
                        printf("ExplicitY v: %i,%i,%i. %f != %f\n", i, k, j, h_v[i*numY*numX + k*numY + j], v[i][j][k]);
                        succes = false;
                    }
                }
            }
        }
        if (!succes) { break;  }

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            unsigned j, k;
            //	implicit x
            for(j=0;j<numX;j++) {  // here a, b,c should have size [numX]
                for(k=0;k<numY;k++) {
                    a[i][k][j] =	   - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][0]);
                    b[i][k][j] = dtInv - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][1]);
                    c[i][k][j] =	   - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][2]);
                }
            }
        }

        cpGlob2Gpu(globArr,outer,numX,numY,numT,d_globs); // made for copying of globs
        deviceImplicitX<T3D>(outer, numX, numY, dtInv, d_globs, d_a, d_b, d_c);

        REAL* a1 = (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
        REAL* b1 = (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
        REAL* c1 = (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
        cudaMemcpy(a1,d_a,sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        cudaMemcpy(b1,d_b,sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        cudaMemcpy(c1,d_c,sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        succes = true;
        printf("Time: %i\n",t);
        for(unsigned i = 0;i<outer;++i) {
            for(unsigned k = 0;k<numY;++k) {
                for(unsigned j = 0;j<numX;++j) {
                    if (abs(a[i][k][j] - a1[i*(numX*numY)+k*numX+j]) > 1e-6) {
                        printf("Implicit X a WRONG! %i,%i,%i: %f != %f\n",i,k,j,a1[i*(numX*numY)+k*numX+j],a[i][k][j]);
                        succes = false;
                    }
                    if (abs(b[i][k][j] - b1[i*(numX*numY)+k*numX+j]) > 1e-6) {
                        printf("Implicit X b WRONG! %i,%i,%i: %f != %f\n",i,k,j,b1[i*(numX*numY)+k*numX+j],b[i][k][j]);
                        succes = false;
                    }
                    if (abs(c[i][k][j] - c1[i*(numX*numY)+k*numX+j]) > 1e-6) {
                        printf("Implicit X c WRONG! %i,%i,%i: %f != %f\n",i,j,k,c1[i*(numX*numY)+k*numX+j],c[i][k][j]);
                        succes = false;
                    }
                }
            }
        }
        if (!succes) { break; }

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
        deviceTridag<T2D*T2D>(d_a,d_b,d_c,d_u,outer*numX*numY,numX,d_u,d_yy);

        // 3D kernel
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++i ) {
            unsigned j, k;

            //	implicit y
            for(j=0;j<numX;j++) {
                for(k=0;k<numY;k++) {  // here a, b, c should have size [numX][numY]
                    a[i][j][k] =        - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][0]);
                    b[i][j][k] = dtInv  - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][1]);
                    c[i][j][k] =		- 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][2]);
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
        cpGlob2Gpu(globArr,outer,numX,numY,numT,d_globs);
        cpCpu2Gpu(a,outer,numX,numY,d_a); // copy a to d_a
        cpCpu2Gpu(b,outer,numX,numY,d_b); // copy b to d_b
        cpCpu2Gpu(c,outer,numX,numY,d_c); // copy c to d_c
        cpCpu2Gpu(u,outer,numY,numX,d_u); // copy u to d_u
        cpCpu2Gpu(v,outer,numX,numY,d_v); // copy v to d_v
        deviceImplicitY<T3D>(outer, numX, numY, dtInv, d_globs,
                d_a, d_b, d_c, d_u, d_v, d_y);

        REAL* y1 = (REAL*) malloc(sizeof(REAL)*outer*numX*numY); // [outer][numX][nymY]
        cudaMemcpy(a1,d_a, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        cudaMemcpy(b1,d_b, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        cudaMemcpy(c1,d_c, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        cudaMemcpy(y1,d_y, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        for (int i = 0; i < outer; i++) {
            for (int j = 0; j < numX; j++) {
                for (int k = 0; k < numY; k++) {
                    if (abs(a[i][j][k] - a1[i*(numX*numY)+j*numY+k]) > 1e-6) {
                        printf("Implicit Y a WRONG! %i,%i,%i: %f != %f\n",i,k,j,a1[i*(numX*numY)+j*numY+k],a[i][j][k]);
                        succes = false;
                    }
                    if (abs(b[i][j][k] - b1[i*(numX*numY)+j*numY+k]) > 1e-6) {
                        printf("Implicit Y b WRONG! %i,%i,%i: %f != %f\n",i,k,j,b1[i*(numX*numY)+j*numY+k],b[i][j][k]);
                        succes = false;
                    }
                    if (abs(c[i][j][k] - c1[i*(numX*numY)+j*numY+k]) > 1e-6) {
                        printf("Implicit Y c WRONG! %i,%i,%i: %f != %f\n",i,k,j,c1[i*(numX*numY)+j*numY+k],c[i][j][k]);
                        succes = false;
                    }
                    if (abs(y[i][j][k] - y1[i*(numX*numY)+j*numY+k]) > 1e-6) {
                        printf("Implicit Y y WRONG! %i,%i,%i: %f != %f\n",i,k,j,y1[i*(numX*numY)+j*numY+k],y[i][j][k]);
                        succes = false;
                    }
                }
            }
        }
        free(a1); free(b1); free(c1); free(y1);

        // 3D kernel
        //#pragma omp parallel for default(shared) schedule(static) if(outer>8)
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
        deviceTridag<T2D*T2D>(d_a,d_b,d_c,d_y,outer*numX*numY,numY,d_globs.myResult,d_yy);

        myResult = (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
        cudaMemcpy(myResult, d_globs.myResult, sizeof(REAL)*outer*numX*numY,cudaMemcpyDeviceToHost);
        for (int i = 0; i < outer; i++) {
            for (int j = 0; j < numX; j++) {
                for (int k = 0; k < numY; k++) {
                    if (abs(globArr[i].myResult[j][k] - myResult[i*numX*numY+j*numY+k]) > 1e-3) {
                        printf("Invalid result after tridag: %i, %i, %i: %f != %f\n",i,j,k,myResult[i*numX*numY+j*numY+k],globArr[i].myResult[j][k]);
                    }
                }
            }
        }
        if (t == 0) {
            deviceResult<T2D*T2D>(outer,numX,numY,d_globs,res);
        }
        free(myResult);
        // End value function
    }
    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_c); cudaFree(d_y);
    cudaFree(d_yy);
    cudaFree(d_u); cudaFree(d_v);
}

#endif // PROJ_CORE_ORIG
