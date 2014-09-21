/*
 * boundary_val.c
 *
 *  Created on: May 11, 2014
 *      Author: mauro
 */
#include <stdio.h>
#include <string.h>
#include "boundary_val.h"
#include "helper.h"

void boundaryvalues( int    imax,
		     int    jmax,
		     double **U,
		     double **V, 
		     float  *imgU, 
		     float  *imgV, 
		     int    **Flag
		   )
{
	int i, j;
	
	for( i = 0; i <= imax; i++ )
	  for( j = 0; j <= jmax; j++){
	    if( Flag[i][j] == 2 ){
	   // if( imgDomain[ i+j*imax ] == 2 ){
		U[i][j] = imgU[ i+j*imax ];
		V[i][j] = imgV[ i+j*imax ];
	  }
      }
}




