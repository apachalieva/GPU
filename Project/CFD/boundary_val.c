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

void findBoundaryvalues( int    imax,
			  int    jmax,
			  double **U,
			  double **V, 
			  float  *imgU, 
			  float  *imgV, 
			  int    **Flag
			)
{
    int i, j;
	
    for( i = 0; i <= imax; i++ ){
      for( j = 0; j <= jmax; j++ ){
	if( Flag[i][j] == 2 ){
	// if( imgDomain[ i+j*imax ] == 2 ){
	  U[i][j] = imgU[ i+j*imax ];
	  V[i][j] = imgV[ i+j*imax ];
	}
      }
    }
}

void setBoundaryvalues( int imax, 
			 int jmax, 
			 double **boundU, 
			 double **boundV, 
			 double **U, 
			 double **V, 
			 int **Flag 
		      )
{
    int i, j;
	
    for( i = 0; i <= imax; i++ ){
      for( j = 0; j <= jmax; j++ ){
	if( Flag[i][j] == 2 ){
	// if( imgDomain[ i+j*imax ] == 2 ){
	  U[i][j] = boundU[i][j];
	  V[i][j] = boundV[i][j];
	}
      }
    }
}




