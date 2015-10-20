#include "ProjHelperFun.h"
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
    int    i, offset;
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

void   run_OrigCPU(
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
    PrivGlobs    globs(numX, numY, numT);
    initGrid(s0, alpha, nu, t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);
    vector<PrivGlobs> globArr (outer, globs);

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
                globArr[i].myResult[j][k] = max(globArr[i].myX[j]-0.001*i, (REAL)0.0); // privatized one level in
            }
        }
    }
    // end setPayoff
    // Value function inserted:
    for(int t = numT-2;t>=0;--t)
    {
        // Adding CPU parallelization
        /**
         * updateParams function
         */
        // Make 3D kernel for computing update of params for each outer loop
        // x: globs.myX, y: globs.myY, z: outer
        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            for(unsigned j=0;j<globArr[i].myX.size();++j) {
                for(unsigned k=0;k<globArr[i].myY.size();++k) {
                    globArr[i].myVarX[j][k] = exp(2.0*(  beta*log(globArr[i].myX[j])
                                + globArr[i].myY[k]
                                - 0.5*nu*nu*globArr[i].myTimeline[t] )
                            );
                    globArr[i].myVarY[j][k] = exp(2.0*(  alpha*log(globArr[i].myX[j])
                                + globArr[i].myY[k]
                                - 0.5*nu*nu*globArr[i].myTimeline[t] )
                            ); // nu*nu
                }
            }
        }
        // end updateParams

        REAL dtInv = 1.0/(globs.myTimeline[t+1]-globs.myTimeline[t]);
        vector<vector<vector<REAL> > > u(outer,vector<vector<REAL> >(numY, vector<REAL>(numX)));   // [outer][numY][numX]
        vector<vector<vector<REAL> > > v(outer,vector<vector<REAL> >(numX, vector<REAL>(numY)));   // [outer][numX][numY]
        vector<vector<REAL> > a(outer,vector<REAL>(numZ)), b(outer,vector<REAL>(numZ)), c(outer,vector<REAL>(numZ)), y(outer,vector<REAL>(numZ));     // [outer][max(numX,numY)]
        vector<vector<REAL> > yy(outer,vector<REAL>(numZ));  // temporary used in tridag  // [outer][max(numX,numY)]

        #pragma omp parallel for default(shared) schedule(static) if(outer>8)
        for( unsigned i = 0; i < outer; ++ i ) {
            /**
             * Rollback function
             */
            unsigned j, k;
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

            //	implicit x
            for(k=0;k<numY;k++) {
                for(j=0;j<numX;j++) {  // here a, b,c should have size [numX]
                    a[i][j] =		 - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][0]);
                    b[i][j] = dtInv - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][1]);
                    c[i][j] =		 - 0.5*(0.5*globArr[i].myVarX[j][k]*globArr[i].myDxx[j][2]);
                }
                // here yy should have size [numX]
                tridag(a[i],b[i],c[i],u[i][k],numX,u[i][k],yy[i]);
            }

            //	implicit y
            for(j=0;j<numX;j++) {
                for(k=0;k<numY;k++) {  // here a, b, c should have size [numY]
                    a[i][k] =		 - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][0]);
                    b[i][k] = dtInv - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][1]);
                    c[i][k] =		 - 0.5*(0.5*globArr[i].myVarY[j][k]*globArr[i].myDyy[k][2]);
                }

                for(k=0;k<numY;k++)
                    y[i][k] = dtInv*u[i][k][j] - 0.5*v[i][j][k];

                // here yy should have size [numY]
                tridag(a[i],b[i],c[i],y[i],numY,globArr[i].myResult[j],yy[i]);
            }
            // end rollback

            if (t == 0) {
                res[i] = globArr[i].myResult[globArr[i].myXindex][globArr[i].myYindex];
            }
        }
        // End value function
    }
}

//#endif // PROJ_CORE_ORIG
