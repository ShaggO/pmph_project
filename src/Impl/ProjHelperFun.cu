#ifndef PROJ_HELPER_FUNS_CU
#define PROJ_HELPER_FUNS_CU

#include "ProjHelperFun.cu.h"
#include "ProjKernels.cu.h"

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/

// Copy triple-nested vector togpu device array
void cpCpu2Gpu(
        vector<vector<vector<REAL > > >& src,
        unsigned numX, unsigned numY, unsigned numZ,
        REAL* dst) {
    int mem_size = sizeof(REAL)*numX*numY*numZ;
    // Allocate local flat array
    REAL* local = (REAL*) malloc(mem_size);
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < numY; j++) {
            // Copy each subarray into the flat array
            memcpy((void*) &local[i*numY*numZ+j*numZ],(void*) &src[i][j][0], sizeof(REAL)*numZ);
        }
    }
    // Copy flat array to device
    cudaMemcpy(dst,local,mem_size,cudaMemcpyHostToDevice);
    free(local);
}

// Copy glob vector to gpu device glob
void cpGlob2Gpu(
        vector<PrivGlobs>& globs,
        unsigned outer,
        unsigned numX,
        unsigned numY,
        unsigned numT,
        DevicePrivGlobs &d_globs) {
    // Allocate local flat array
    REAL* myX =         (REAL*) malloc(sizeof(REAL)*outer*numX);
    REAL* myY =         (REAL*) malloc(sizeof(REAL)*outer*numY);
    REAL* myTimeline =  (REAL*) malloc(sizeof(REAL)*outer*numT);
    REAL* myResult =    (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
    REAL* myVarX =      (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
    REAL* myVarY =      (REAL*) malloc(sizeof(REAL)*outer*numX*numY);
    REAL* myDxx =       (REAL*) malloc(sizeof(REAL)*outer*numX*4);
    REAL* myDyy =       (REAL*) malloc(sizeof(REAL)*outer*numY*4);

    // Copy each subarray into the flat arrays
    for (int i = 0; i < outer; i++) {
        memcpy(&myX[i*numX],&globs[i].myX[0],sizeof(REAL)*numX);
        memcpy(&myY[i*numY],&globs[i].myY[0],sizeof(REAL)*numY);
        memcpy(&myTimeline[i*numT],&globs[i].myTimeline[0],sizeof(REAL)*numT);
        for (int j = 0; j < numX; j++) {
            memcpy(&myResult[i*numX*numY+j*numY],&globs[i].myResult[j][0], sizeof(REAL)*numY);
            memcpy(&myVarX[i*numX*numY+j*numY],&globs[i].myVarX[j][0], sizeof(REAL)*numY);
            memcpy(&myVarY[i*numX*numY+j*numY],&globs[i].myVarY[j][0], sizeof(REAL)*numY);
            memcpy(&myDxx[i*numX*4+j*4],&globs[i].myDxx[j][0],sizeof(REAL)*4);
        }
        for (int j = 0; j < numY; j++) {
            memcpy(&myDyy[i*numY*4+j*4],&globs[i].myDyy[j][0],sizeof(REAL)*4);
        }
    }

    // Copy flat array to device
    cudaMemcpy(d_globs.myX,myX,sizeof(REAL)*outer*numX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myY,myY,sizeof(REAL)*outer*numY,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myTimeline,myTimeline,sizeof(REAL)*outer*numT,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myResult,myResult,sizeof(REAL)*outer*numX*numY,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myVarX,myVarX,sizeof(REAL)*outer*numX*numY,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myVarY,myVarY,sizeof(REAL)*outer*numX*numY,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myDxx,myDxx,sizeof(REAL)*outer*numX*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_globs.myDyy,myDyy,sizeof(REAL)*outer*numY*4,cudaMemcpyHostToDevice);
    d_globs.myXindex = globs[0].myXindex;
    d_globs.myYindex = globs[0].myYindex;

    // Clean up
    free(myX);      free(myY);      free(myTimeline);
    free(myResult); free(myVarX);   free(myVarY);
    free(myDxx);    free(myDyy);
}
/**
 * Fills in:
 *   globs.myTimeline  of size [0..numT-1]
 *   globs.myX         of size [0..numX-1]
 *   globs.myY         of size [0..numY-1]
 * and also sets
 *   globs.myXindex and globs.myYindex (both scalars)
 */
void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t,
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs
) {
    for(unsigned i=0;i<numT;++i)
        globs.myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    globs.myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(unsigned i=0;i<numX;++i)
        globs.myX[i] = i*dx - globs.myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    globs.myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned i=0;i<numY;++i)
        globs.myY[i] = i*dy - globs.myYindex*dy + logAlpha;
}

/**
 * Fills in:
 *    Dx  [0..n-1][0..3] and
 *    Dxx [0..n-1][0..3]
 * Based on the values of x,
 * Where x's size is n.
 */
void initOperator(  const vector<REAL>& x,
                    vector<vector<REAL> >& Dxx
) {
	const unsigned n = x.size();

	REAL dxl, dxu;

	//	lower boundary
	//dxl		 =  0.0;
	//dxu		 =  x[1] - x[0];

	Dxx[0][0] =  0.0;
	Dxx[0][1] =  0.0;
	Dxx[0][2] =  0.0;
    Dxx[0][3] =  0.0;

	//	standard case
	for(unsigned i=1;i<n-1;i++)
	{
		dxl      = x[i]   - x[i-1];
		dxu      = x[i+1] - x[i];

		Dxx[i][0] =  2.0/dxl/(dxl+dxu);
		Dxx[i][1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
		Dxx[i][2] =  2.0/dxu/(dxl+dxu);
        Dxx[i][3] =  0.0;
	}

	//	upper boundary
	//dxl		   =  x[n-1] - x[n-2];
	//dxu		   =  0.0;

	Dxx[n-1][0] = 0.0;
	Dxx[n-1][1] = 0.0;
	Dxx[n-1][2] = 0.0;
    Dxx[n-1][3] = 0.0;
}
#endif //PROJ_HELPER_FUN
