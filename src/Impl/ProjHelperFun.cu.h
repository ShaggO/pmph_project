#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"
#include <cuda_runtime.h>
#include "ProjKernels.cu.h"

using namespace std;

struct DevicePrivGlobs {
    // grid
    REAL* myX; // [numX]
    REAL* myY; // [numY]
    REAL* myTimeline; // [numT]
    unsigned myXindex;
    unsigned myYindex;

    // variable
    REAL* myResult; // [numX*numY]

    // coeffs
    REAL* myVarX; // [numX*numY]
    REAL* myVarY; // [numX*numY]

    // operators
    REAL*   myDxx; // [numX*4]
    REAL*   myDyy; // [numY*4]

    __device__ __host__ DevicePrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }
    __device__ __host__ DevicePrivGlobs( const unsigned int& outer,
                                         const unsigned int& numX,
                                         const unsigned int& numY,
                                         const unsigned int& numT ) {
        cudaMalloc((void**)&this->myX, sizeof(REAL)*outer*numX);
        cudaMalloc((void**)&this->myY, sizeof(REAL)*outer*numY);
        cudaMalloc((void**)&this->myTimeline, sizeof(REAL)*outer*numT);
        cudaMalloc((void**)&this->myResult, sizeof(REAL)*outer*numX*numY);
        cudaMalloc((void**)&this->myVarX, sizeof(REAL)*outer*numX*numY);
        cudaMalloc((void**)&this->myVarY, sizeof(REAL)*outer*numX*numY);
        cudaMalloc((void**)&this->myDxx, sizeof(REAL)*outer*numX*4);
        cudaMalloc((void**)&this->myDyy, sizeof(REAL)*outer*numY*4);
    }

    __device__ __host__ ~DevicePrivGlobs() {
        cudaFree(this->myX);
        cudaFree(this->myY);
        cudaFree(this->myTimeline);
        cudaFree(this->myResult);
        cudaFree(this->myVarX);
        cudaFree(this->myVarY);
        cudaFree(this->myDxx);
        cudaFree(this->myDyy);
    }

} __attribute__ ((aligned (128)));

struct PrivGlobs {

    //	grid
    vector<REAL>        myX;        // [numX]
    vector<REAL>        myY;        // [numY]
    vector<REAL>        myTimeline; // [numT]
    unsigned            myXindex;
    unsigned            myYindex;

    //	variable
    vector<vector<REAL> > myResult; // [numX][numY]

    //	coeffs
    vector<vector<REAL> >   myVarX; // [numX][numY]
    vector<vector<REAL> >   myVarY; // [numX][numY]

    //	operators
    vector<vector<REAL> >   myDxx;  // [numX][4]
    vector<vector<REAL> >   myDyy;  // [numY][4]

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {
        this->  myX.resize(numX);
        this->myDxx.resize(numX);
        for(int k=0; k<numX; k++) {
            this->myDxx[k].resize(4);
        }

        this->  myY.resize(numY);
        this->myDyy.resize(numY);
        for(int k=0; k<numY; k++) {
            this->myDyy[k].resize(4);
        }

        this->myTimeline.resize(numT);

        this->  myVarX.resize(numX);
        this->  myVarY.resize(numX);
        this->myResult.resize(numX);
        for(unsigned i=0;i<numX;++i) {
            this->  myVarX[i].resize(numY);
            this->  myVarY[i].resize(numY);
            this->myResult[i].resize(numY);
        }

    }
} __attribute__ ((aligned (128)));


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs
            );

void initOperator(  const vector<REAL>& x,
                    vector<vector<REAL> >& Dxx
                 );

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
);

void rollback( const unsigned g, PrivGlobs& globs );

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike,
                const REAL t,
                const REAL alpha,
                const REAL nu,
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
            );

template<const unsigned T>
void deviceInitGrid( const REAL s0, const REAL alpha, const REAL nu, const REAL t,
        const unsigned outer, const unsigned numX, const unsigned numY, const unsigned numT, DevicePrivGlobs &globs) {
    const unsigned numZ = max(numX,max(numY,numT));
    const unsigned dimx = ceil(((float) outer) / T);
    const unsigned dimy = ceil(((float) numZ) / T);
    const dim3 block(T,T,1), grid(dimx,dimy,1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);

    globs.myXindex = static_cast<unsigned>(s0/dx) % numX;
    globs.myYindex = static_cast<unsigned>(numY/2.0);

    const REAL myXvar = globs.myXindex*dx-s0;
    const REAL myYvar = globs.myYindex*dy-logAlpha;
    // myTimeline, myXindex and myYindex for each outer
    initGridKernel<T><<<grid, block>>>(outer,numX,numY,numT,
            globs.myTimeline,globs.myX,globs.myY,
            myXvar, myYvar, dx, dy, t);
    cudaThreadSynchronize();
}

template<const unsigned T>
void deviceInitOperator( const unsigned outer, const unsigned num,
    REAL* x, REAL* Dxx) {
    const unsigned dimx = ceil(((float) outer) / T);
    const unsigned dimy = ceil(((float) num) / T);
    const dim3 block(T,T,1), grid(dimx,dimy,1);
    initOperatorKernel<T><<<grid, block>>>(outer,num,x,Dxx);
    cudaThreadSynchronize();
}

template<const unsigned T>
void deviceSetPayoff(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        DevicePrivGlobs &globs
        )
{
    const unsigned dimx = ceil((float) numX / T);
    const unsigned dimy = ceil((float) numY / T);
    const unsigned dimz = ceil((float) outer / T);
    const dim3 block(T,T,T), grid(dimx,dimy,dimz);

    setPayoffKernel<T><<<grid, block>>>(outer, numX, numY, globs.myX, globs.myResult);
    cudaThreadSynchronize();
}

template<const unsigned T>
void deviceUpdateParams(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        REAL alpha, REAL beta, REAL nu, REAL time,
        DevicePrivGlobs &globs
        )
{
    const unsigned dimx = ceil(((float) numX) / T);
    const unsigned dimy = ceil(((float) numY) / T);
    const unsigned dimz = ceil(((float) outer) / T);
    const dim3 block(T,T,T), grid(dimx,dimy,dimz);
    REAL x = 0.5*nu*nu*time; 
    updateParamsKernel<T><<<grid, block>>>(outer, numX, numY, alpha, beta, x, globs.myX, globs.myY, globs.myVarX, globs.myVarY);
}

template<const unsigned T>
void explicitX(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const REAL dtInv,
        REAL* v,
        REAL* u,
        DevicePrivGlobs &globs
        )
{
    const unsigned dimx = ceil((float) numX / T);
    const unsigned dimy = ceil((float) numY / T);
    const unsigned dimz = ceil((float) outer / T);
    const dim3 block(T,T,T), grid(dimx,dimy,dimz);

    explicitXKernel<T><<<grid, block>>>(outer, numX, numY, dtInv, v, u, globs.myVarX, globs.myResult, globs.myDxx);
    cudaThreadSynchronize();
}

template<const unsigned T>
void explicitY(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const REAL dtInv,
        REAL* v,
        REAL* u,
        DevicePrivGlobs &globs
        )
{
    const unsigned dimx = ceil((float) numX / T);
    const unsigned dimy = ceil((float) numY / T);
    const unsigned dimz = ceil((float) outer / T);
    const dim3 block(T,T,T), grid(dimx,dimy,dimz);

    explicitYKernel<T><<<grid, block>>>(outer, numX, numY, dtInv, v, u, globs.myVarY, globs.myResult, globs.myDyy);
    cudaThreadSynchronize();
}

#endif // PROJ_HELPER_FUNS
