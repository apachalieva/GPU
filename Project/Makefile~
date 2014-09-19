main: main.cu aux.cu aux.h helper.c helper.h init.c init.h boundary_val.c boundary_val.h uvp.c uvp.h visual.c visual.h sor.c sor.h Makefile
	nvcc -o main main.cu aux.cu helper.c init.c boundary_val.c uvp.c visual.c sor.c --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

