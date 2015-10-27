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
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int j = blockIdx.x*T + tidx; // myX.size
    int k = blockIdx.y*T + tidy; // myY.size
    int i = blockIdx.z*T + tidz; // outer
    if (i < outer && j < numX && k < numY) {
        myResult[i * numX*numY + j * numX + k] = max(myX[i * numX + j]-0.001*i, (REAL)0.0);
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
#endif //PROJ_KERNELS
