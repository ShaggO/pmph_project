CXX        = nvcc
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL
CXXFLAGS   = -O3 -DWITH_FLOATS=1

INCLUDES    += -I ../include
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =ProjectMain.cu ProjHelperFun.cu
HELPERS     =ProjHelperFun.cu.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ProjCoreOrig.cu.h ProjKernels.cu.h TridagKernel.cu.h
EXECUTABLE  =runproject


default: clean compile run_small

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(SOURCES_CPP)


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE)
