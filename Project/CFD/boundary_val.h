#ifndef __RANDWERTE_H__
#define __RANDWERTE_H__

/**
 * The boundary values of the problem are set.
 */
void findBoundaryvalues( int    imax,
			  int    jmax,
			  double **U,
			  double **V,
			  float  *imgU,
			  float  *imgV,
			  int    **Flag
			);

void setBoundaryvalues( int    imax,
			 int    jmax,
			 double **U,
			 double **V,
			 double **boundU, 
			 double **boundV,
			 int    **Flag
			);

#endif
