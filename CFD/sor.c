#include "sor.h"
#include "helper.h"
#include <math.h>

void sor(
  double omg,
  double dx,
  double dy,
  int    imax,
  int    jmax,
  const int fluid_cells,
  double **P,
  double **RS,
  int    **Flag,
  double *res,
  double dp
) {
  
  int i,j;
  double rloc;
  double coeff = omg/(2.0*(1.0/(dx*dx)+1.0/(dy*dy)));

  /* SOR iteration */
  for(i = 1; i <= imax; i++) {
      for(j = 1; j<=jmax; j++) {
		  /* SOR computation limited to the fluid cells ( C_F - fluid cell )*/
		  if( Flag[ i ][ j ] == C_F ){
			P[i][j] = (1.0-omg)*P[i][j] + coeff*(( P[i+1][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]+P[i][j-1])/(dy*dy) - RS[i][j]);
		  }
      }
  }


  /* compute the residual */
  rloc = 0;
  for(i = 1; i <= imax; i++) {
      for(j = 1; j <= jmax; j++) {
	  /* Residual computation limited to the fluid cells ( C_F - fluid cell ) */
	  if( Flag[ i ][ j ] == C_F ){
	      rloc += ( (P[i+1][j]-2.0*P[i][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]-2.0*P[i][j]+P[i][j-1])/(dy*dy) - RS[i][j])*
		          ( (P[i+1][j]-2.0*P[i][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]-2.0*P[i][j]+P[i][j-1])/(dy*dy) - RS[i][j]);
	  }
    }
  }
  /* Residual devided only by the number of fluid cells instead of imax*jmax */
  rloc = rloc/((double)fluid_cells);
  rloc = sqrt(rloc);
  /* set residual */
  *res = rloc;
}

