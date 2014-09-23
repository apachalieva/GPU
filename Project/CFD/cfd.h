
#include <stdio.h>
#include <string.h>

#define PARAMF "cavity.dat"
#define VISUAF "visual/sim"

#define OBSTACLE 0
#define FLUID 1
#define INFLOW 2

int cfd( int argc, char** args, float *imgU, float *imgV, int *imgDomain, float *initBU, float *initBV, int imax, int jmax, int iter );