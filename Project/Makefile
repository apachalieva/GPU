main: main.cu aux.cu aux.h functionalities.cu functionalities.h CFD/cfd.c CFD/cfd.h CFD/helper.c CFD/helper.h CFD/init.c CFD/init.h CFD/boundary_val.c CFD/boundary_val.h CFD/uvp.c CFD/uvp.h CFD/visual.c CFD/visual.h CFD/sor.c CFD/sor.h Makefile
	nvcc -o main main.cu aux.cu functionalities.cu CFD/cfd.c CFD/helper.c CFD/init.c CFD/boundary_val.c CFD/uvp.c CFD/visual.c CFD/sor.c --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

