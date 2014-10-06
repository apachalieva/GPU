main: main.cu aux.cu aux.h functionalities.cu functionalities.h CFD/cfd.c CFD/cfd.h CFD/helper.c CFD/helper.h CFD/visual.c CFD/visual.h CFD/functions.c CFD/functions.h Makefile
	nvcc -o main main.cu aux.cu functionalities.cu CFD/cfd.c CFD/helper.c CFD/visual.c CFD/functions.c --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

