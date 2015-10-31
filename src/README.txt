To compile and run:
    $ make clean; make; make run_small
                                _medium
                                _large

Folder `OrigImpl' contains the original implementation:
    -- `ProjectMain.cpp'   contains the main function
    -- `ProjCoreOrig.cpp'  contains the core functions
                                (to parallelize)
    -- `ProjHelperFun.cpp' contains the functions that compute
                                the input parameters, and
                                (can be parallelize as well)

Folder `include' contains
    -- `ParserC.h'     implements a simple parser
    -- `ParseInput.h'  reads the input/output data
                        and provides validation.
    -- `OpenmpUtil.h'  some OpenMP-related helpers.
    -- `Constants.h'   currently only sets up REAL
                        to either double or float
                        based on the compile-time
                        parameter WITH_FLOATS.

    -- `CudaUtilProj.cu.h' provides stubs for calling
                        transposition and inclusive
                        (segmented) scan.
    -- `TestCudaUtil.cu'  a simple tester for
                        transposition and scan.

Folder `OpenMpImpl' contains the OpenMP version
    -- `ProjectMain.cpp'   contains the main function
    -- `ProjCoreOrig.cpp'  contains the core functions
                                (to parallelize)
    -- `ProjHelperFun.cpp' contains the functions that compute
                                the input parameters, and
                                (can be parallelize as well)

Folder `NaiveImpl' contains the Naïve CUDA version
    -- `ProjectMain.cu'   contains the main function
    -- `ProjCoreOrig.cu.h'  contains the core functions
    -- `ProjHelperFun.cu.h' contains the functions that compute
                                the input parameters
    -- `ProjHelperFun.cu' contains the Tridag kernel and some
                                helper functions
    -- `ProjKernels.cu.h' contains most of the CUDA kernels
    -- `TestCudaTridag.cu' contains a test function for the
                                Tridag kernel
    -- `TridagKernel.cu.h' contains helper functions for the
                                Tridag kernel
    -- `TridagPar.cu.h' contains a CPU version of the Tridag
                                kernel

Folder `Impl' contains the Optimized CUDA version
    -- `ProjectMain.cu'   contains the main function
    -- `ProjCoreOrig.cu.h'  contains the core functions
    -- `ProjHelperFun.cu.h' contains the functions that compute
                                the input parameters
    -- `ProjHelperFun.cu' contains the Tridag kernel and some
                                helper functions
    -- `ProjKernels.cu.h' contains most of the CUDA kernels
    -- `TestCudaTridag.cu' contains a test function for the
                                Tridag kernel
    -- `TridagKernel.cu.h' contains helper functions for the
                                Tridag kernel
    -- `TridagPar.cu.h' contains a CPU version of the Tridag
                                kernel
