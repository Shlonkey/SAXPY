SOURCE = source
BIN = bin

fast:
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy_fast -O3 -D OPTIMIZATION_O3

debug:
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy_debug -O0 -g -G

clean:
	del /F /Q $(BIN)\* vc140.pdb
	