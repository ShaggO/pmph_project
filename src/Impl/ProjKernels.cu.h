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

#endif //PROJ_KERNELS
