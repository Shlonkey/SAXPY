SOURCE = source
BIN = bin

fast_short:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D SHORT

debug_short:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D SHORT

fast_int:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D INT

debug_int:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D INT

fast_long:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D LONG

debug_long:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D LONG

fast_long_long:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D LONGLONG

debug_long_long:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D LONGLONG

fast_float:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D FLOAT

debug_float:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D FLOAT

fast_double:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D DOUBLE

debug_double:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D DOUBLE

fast_long_double:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O3 -D OPTIMIZATION_O3 -D LONGDOUBLE

debug_long_double:
	del /F /Q $(BIN)\* vc140.pdb
	nvcc $(SOURCE)\saxpy.cu -o $(BIN)\saxpy -O0 -g -G -D LONGDOUBLE

clean:
	del /F /Q $(BIN)\* vc140.pdb

help:
	echo "Availible Options	:	{fast, debug}_{short, int, long, long_long, float, double, long_double}"
	