#include "helper.h"
#include "init.h"

int read_parameters( const char *szFileName,       /* name of the file 			*/
		     double *Re,                   /* reynolds number   		*/
		     double *UI,                   /* velocity x-direction 		*/
		     double *VI,                   /* velocity y-direction 		*/
                     double *PI,                   /* pressure 				*/
                     double *GX,                   /* gravitation x-direction 		*/
                     double *GY,                   /* gravitation y-direction 		*/
                     double *t_end,                /* end time 				*/
                     double *xlength,              /* length of the domain x-dir.	*/
                     double *ylength,              /* length of the domain y-dir.	*/
                     double *dt,                   /* time step 			*/
                     double *dx,                   /* length of a cell x-dir. 		*/
                     double *dy,                   /* length of a cell y-dir. 		*/
                     int    imax,                 /* number of cells x-direction	*/
                     int    jmax,                 /* number of cells y-direction	*/
                     double *alpha,                /* uppwind differencing factor	*/
                     double *omg,                  /* relaxation factor 		*/
                     double *tau,                  /* safety factor for time step	*/
                     int    *itermax,              /* max. number of iterations  	*/
		                                   /* for pressure per time step	*/
                     double *eps,                  /* accuracy bound for pressure	*/
                     double *dt_value,             /* time for output 			*/
                     double *dp		   /* dp/dx gradient of pressure 	*/
		  )
{
   READ_DOUBLE( szFileName, *Re    );
   READ_DOUBLE( szFileName, *t_end );
   READ_DOUBLE( szFileName, *dt    );

   READ_DOUBLE( szFileName, *omg   );
   READ_DOUBLE( szFileName, *eps   );
   READ_DOUBLE( szFileName, *tau   );
   READ_DOUBLE( szFileName, *alpha );

   READ_INT   ( szFileName, *itermax );
   READ_DOUBLE( szFileName, *dt_value );

   READ_DOUBLE( szFileName, *UI );
   READ_DOUBLE( szFileName, *VI );
   READ_DOUBLE( szFileName, *GX );
   READ_DOUBLE( szFileName, *GY );
   READ_DOUBLE( szFileName, *PI );

   READ_DOUBLE( szFileName, *dp );
   
   *xlength = (double)(imax)/10;
   *ylength = (double)(jmax)/10;
   
   *dx = *xlength / (double)(imax);
   *dy = *ylength / (double)(jmax);

   return 1;
}


void init_uv( int imax, int jmax, double **U, double **V, float *imgU, float *imgV, int **Flag, int iter ){
  int i, j;
  for( i = 0; i <= imax; i++ ){
    for( j = 0; j <= jmax; j++ ){
      if( Flag[i][j] == 1 ){
	if( iter == 0 ){
	    U[i][j] = 0.0;
	    V[i][j] = 0.0;
	}
	else{
	   U[i][j] = imgU[i+j*imax];
	   V[i][j] = imgV[i+j*imax];
	}
      }
    }
  }
}


void init_flag( const int imax, const int jmax, int **Flag, int *imgDomain ){
	
  int i, j;
  for( i = 0; i <= imax; i++ ){
    for( j = 0; j <= jmax; j++ ){
      Flag[i][j] = imgDomain[i+j*imax];
    }
  }
}
