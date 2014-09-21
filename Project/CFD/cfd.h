

/* includefiles */
//extern "C"{
//     #include "helper.h"
//     #include "visual.h"
//     #include "init.h"
//     #include "uvp.h"
//     #include "boundary_val.h"
//     #include "sor.h"
//}
#include <stdio.h>
#include <string.h>

#define PARAMF "cavity.dat"
#define VISUAF "visual/sim"

#define OBSTACLE 0
#define FLUID 1
#define INFLOW 2

int cfd(int argc, char** args, float *imgU, float *imgV, int *imgDomain);