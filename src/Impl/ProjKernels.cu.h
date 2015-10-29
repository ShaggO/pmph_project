#ifndef PROJ_KERNELS
#define PROJ_KERNELS
#include <stdlib.h>

template<const unsigned T>
__global__ void initGridKernel( const unsigned outer, const unsigned numX, const unsigned numY, const unsigned numT,
        REAL* timeline, REAL* myX, REAL* myY, const REAL myXvar, const REAL myYvar, const REAL dx, const REAL dy, const REAL t) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int i = blockIdx.x*T + tidx;
    int j = blockIdx.y*T + tidy;
    if (i < outer) {
        if (j < numX) {
            myX[i*numX+j] = j*dx - myXvar;
        }
        if (j < numY) {
            myY[i*numY+j] = j*dy - myYvar;
        }
        if (j < numT) {
            timeline[i*numT+j] = t*j/(numT-1);
        }
    }
}

template<const unsigned T>
__global__ void setPayoffKernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        REAL* myX,
        REAL* myResult
        )
{
    int i = blockIdx.z*T + threadIdx.z; // outer
    int j = blockIdx.x*T + threadIdx.x; // myX.size
    int k = blockIdx.y*T + threadIdx.y; // myY.size
    if (i < outer && j < numX && k < numY) {
        myResult[i * numX*numY + j * numY + k] = max(myX[i * numX + j]-0.001*i, (REAL)0.0);
    }
}

template<const unsigned T>
__global__ void initOperatorKernel(const unsigned outer, const unsigned num, REAL* x, REAL* Dxx) {
    int i = blockIdx.x*T + threadIdx.x;
    int j = blockIdx.y*T + threadIdx.y;
    if (i < outer && j < num) {
        int offset = i*(num*4)+j*4;
        if (j == 0 || j == num-1) {
            Dxx[offset] = 0.0;
            Dxx[offset+1] = 0.0;
            Dxx[offset+2] = 0.0;
            Dxx[offset+3] = 0.0;
        } else {
            REAL dxl = x[j] - x[j-1];
            REAL dxu = x[j+1] - x[j];
            Dxx[offset] = 2.0/dxl/(dxl+dxu);
            Dxx[offset+1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
            Dxx[offset+2] = 2.0/dxu/(dxl+dxu);
            Dxx[offset+3] = 0.0;
        }
    }
}

template<const unsigned T>
__global__ void updateParamsKernel(const unsigned outer, const unsigned numX, const unsigned numY,
        REAL alpha, REAL beta, REAL sConst, 
        REAL* myX,
        REAL* myY,
        REAL* myVarX,
        REAL* myVarY) {
    int i = blockIdx.z*T + threadIdx.z;
    int j = blockIdx.x*T + threadIdx.x;
    int k = blockIdx.y*T + threadIdx.y;
    if (i < outer && j < numX && k < numY) {
        int vIdx = i*(numX*numY)+j*numY+k;
        REAL Xthis = log(myX[i*numX+j]);
        REAL Ythis = myY[i*numY+k];

        myVarX[vIdx] = exp(2.0*(  beta*Xthis + Ythis - sConst ));
        myVarY[vIdx] = exp(2.0*(  alpha*Xthis + Ythis - sConst ));
    }
}

template<const unsigned T>
__global__ void implicitXKernel(const unsigned outer, const unsigned numX, const unsigned numY,
        REAL dtInv,
        REAL* myVarX,
        REAL* myDxx,
        REAL* a,
        REAL* b,
        REAL* c)
{
    int i = blockIdx.z*T + threadIdx.z;
    int j = blockIdx.x*T + threadIdx.x;
    int k = blockIdx.y*T + threadIdx.y;
    if (i < outer && j < numX && k < numY) {
        int idx = i*(numX*numY)+k*numX+j;
        int idxDxx = i*(numX*4)+j*4;
        REAL Xthis = 0.25*myVarX[i*(numX*numY)+j*numY+k];
        a[idx] =       - Xthis*myDxx[idxDxx];
        b[idx] = dtInv - Xthis*myDxx[idxDxx+1];
        c[idx] =       - Xthis*myDxx[idxDxx+2];
    }
}

template<const unsigned T>
__global__ void implicitYKernel(const unsigned outer, const unsigned numX, const unsigned numY,
        REAL dtInv,
        REAL* myVarY,
        REAL* myDyy,
        REAL* a,
        REAL* b,
        REAL* c)
{
    int i = blockIdx.z*T + threadIdx.z;
    int j = blockIdx.x*T + threadIdx.x;
    int k = blockIdx.y*T + threadIdx.y;
    if (i < outer && j < numX && k < numY) {
        int idx = i*(numX*numY)+j*numY+k;
        int idxDyy = i*(numY*4)+k*4;
        REAL Ythis = 0.25*myVarY[i*numX*numY+j*numY+k];
        a[idx] =       - Ythis*myDyy[idxDyy];
        b[idx] = dtInv - Ythis*myDyy[idxDyy+1];
        c[idx] =       - Ythis*myDyy[idxDyy+2];
    }
}

template<const unsigned T>
__global__ void implicitYKernelY(const unsigned outer, const unsigned numX, const unsigned numY,
        REAL dtInv,
        REAL* u,
        REAL* v,
        REAL* y)
{
    int i = blockIdx.z*T + threadIdx.z;
    int j = blockIdx.x*T + threadIdx.x;
    int k = blockIdx.y*T + threadIdx.y;
    if (i < outer && j < numX && k < numY) {
        int idx = i*numX*numY+j*numY+k;
        y[idx] = dtInv*u[i*numX*numY+k*numX+j] - 0.5*v[idx];
    }
}

template<const unsigned T>
__global__ void explicitXKernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const REAL dtInv,
        REAL* u,
        REAL* myVarX,
        REAL* myResult,
        REAL* myDxx
        )
{
    int i = blockIdx.z*T + threadIdx.z; // outer
    int j = blockIdx.x*T + threadIdx.x; // myX.size
    int k = blockIdx.y*T + threadIdx.y; // myY.size


    if (i < outer && j < numX && k < numY) {
        // u[outer][numY][numX]
        int uindex = i*numY*numX + k*numX + j;
        // myVarX [outer][numX][numY]
        int myVarXindex = i*numX*numY + j * numY + k;
        // myResult[outer][numX][numY]
        u[uindex] = dtInv * myResult[myVarXindex];

        // Dxx [outer][numX][4]
        int Dxxindex = i*numX*4 + j*4;
        REAL varX = 0.25*myVarX[myVarXindex];
        if (j > 0) {
            u[uindex] +=    varX*myDxx[Dxxindex]
                                * myResult[i*numX*numY + (j-1)*numY + k];
        }
        u[uindex] +=        varX*myDxx[Dxxindex+1]
                                * myResult[myVarXindex];
        if (j < numX) {
            u[uindex] +=    varX*myDxx[Dxxindex+2]
                                * myResult[i*numX*numY + (j+1)*numY + k];
        }
    }
}

template<const unsigned T>
__global__ void explicitYKernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const REAL dtInv,
        REAL* v,
        REAL* u,
        REAL* myVarY,
        REAL* myResult,
        REAL* myDyy
        )
{
    int i = blockIdx.z*T + threadIdx.z; // outer
    int j = blockIdx.x*T + threadIdx.x; // myX.size
    int k = blockIdx.y*T + threadIdx.y; // myY.size
    if (i < outer && j < numX && k < numY) {
        // v[outer][numX][numY]
        int vindex = i*numX*numY + j*numY + k;
        v[vindex] = 0.0;

        int Dyyindex = i*numY*4 + k*4;
        REAL varY = 0.5*myVarY[vindex];
        if(k > 0) {
            v[vindex] +=    varY*myDyy[Dyyindex]   * myResult[vindex-1];
        }
        v[vindex]  +=       varY*myDyy[Dyyindex+1] * myResult[vindex];
        if(k < numY-1) {
            v[vindex] +=    varY*myDyy[Dyyindex+2] * myResult[vindex+1];
        }
        u[i*numY*numX + k*numX + j] += v[vindex];
    }
}

template<const unsigned T>
__global__ void sgmMatTransposeKernel( REAL* A, REAL* trA, int rowsA, int colsA ) {
    __shared__ REAL tile[T][T+1];
    int gidz = blockIdx.z*blockDim.z*threadIdx.z;
    A += gidz*rowsA*colsA;
    trA += gidz*rowsA*colsA;
    // follows code for matrix transp in x & y
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int j = blockIdx.x*T + tidx, i = blockIdx.y*T+tidy;
    if ( j < colsA && i < rowsA ) {
        tile[tidy][tidx] = A[i*colsA+j];
    }
    __syncthreads();

    i = blockIdx.y*T+tidx;
    j = blockIdx.x*T+tidy;
    if ( j < colsA && i < rowsA ) {
        trA[j*rowsA+i] = tile[tidx][tidy];
    }
}
#endif //PROJ_KERNELS
