SOURCE = source
BIN = bin

fast:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3

debug:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G

clean:
	del /F /Q $(BIN)\* vc140.pdb
	