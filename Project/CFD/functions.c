#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "helper.h"
#include "functions.h"

//__________________________________________________________________//
//								      //
//		CFD Initialization Functions			      //
//								      //
//__________________________________________________________________//

int read_parameters( const char *szFileName,       /* name of the file 			*/
		     double *Re,                   /* reynolds number   			*/
                     double *PI,                   /* pressure 				*/
                     double *GX,                   /* gravitation x-direction 		*/
                     double *GY,                   /* gravitation y-direction 		*/
                     double *t_end,                /* end time 				*/
                     double *xlength,              /* length of the domain x-dir.		*/
                     double *ylength,              /* length of the domain y-dir.		*/
                     double *dt,                   /* time step 				*/
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
	   U[i][j] = 3.0*imgU[i+j*imax]; // TODO 3.0?
	   V[i][j] = 3.0*imgV[i+j*imax]; // TODO 3.0?
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

//__________________________________________________________________//
//								      //
//		CFD Boundary Values Function			      //
//								      //
//__________________________________________________________________//
void boundaryvalues( int    imax,
		     int    jmax,
		     double **U,
		     double **V, 
		     float  *initBU, 
		     float  *initBV, 
		     int    **Flag
		   )
{
    int i, j;
	
    for( i = 0; i <= imax; i++ ){
      for( j = 0; j <= jmax; j++ ){
	if( Flag[i][j] == 2 ){
	// if( imgDomain[ i+j*imax ] == 2 ){
	  U[i][j] = initBU[ i+j*imax ];
	  V[i][j] = initBV[ i+j*imax ];
	}
      }
    }
}

//__________________________________________________________________//
//								      //
//			CFD UVP Functions 			      //
//								      //
//__________________________________________________________________//

#define SQ(a) ((a)*(a))

/* central difference approximation of the second derivative in x */
inline double d2dx(double **m, int i, int j, double dx){
	return (m[i+1][j] - 2*m[i][j] + m[i-1][j]) / (SQ(dx));
}

/* central difference approximation of the second derivative in y */
inline double d2dy(double **m, int i, int j, double dy){
	return (m[i][j+1] - 2*m[i][j] + m[i][j-1]) / (SQ(dy));
}

/* forward difference approximation of the first derivative in x */
inline double ddx(double **m, int i, int j, double dx){
	return (m[i+1][j] - m[i][j]) / dx;
}

/* approximation of the first derivative of the square of u in x */
inline double du2dx(double **m, int i, int j, double dx, double alpha){
	return (
			SQ(m[i][j]+m[i+1][j]) - SQ(m[i-1][j]+m[i][j])
			+ alpha * ( fabs(m[i][j]+m[i+1][j]) * (m[i][j]-m[i+1][j]) -  fabs(m[i-1][j]+m[i][j]) * (m[i-1][j]-m[i][j]) )
	                       )/dx/4.0;
}

/* approximation of the first derivative of the square of v in y */
inline double dv2dy(double **m, int i, int j, double dy, double alpha){
	return (
			SQ(m[i][j]+m[i][j+1]) - SQ(m[i][j-1]+m[i][j])
			+ alpha * ( fabs(m[i][j]+m[i][j+1]) * (m[i][j]-m[i][j+1]) -  fabs(m[i][j-1]+m[i][j]) * (m[i][j-1]-m[i][j]))
	                       )/dy/4.0;
}

inline double duvdx(double ** u, double **v, int i, int j, double dx, double alpha){
	return (
			(u[i][j]+u[i][j+1]) * (v[i][j]+v[i+1][j]) - (u[i-1][j]+u[i-1][j+1]) * (v[i-1][j]+v[i][j])
			+ alpha * ( fabs(u[i][j]+u[i][j+1]) * (v[i][j]-v[i+1][j]) - fabs(u[i-1][j]+u[i-1][j+1]) * (v[i-1][j]-v[i][j]) )
	                       )/dx/4.0;
}

inline double duvdy(double **u, double **v, int i, int j, double dy, double alpha){
	return (
			(u[i][j]+u[i][j+1]) * (v[i][j]+v[i+1][j]) - (u[i][j-1]+u[i][j]) * (v[i][j-1]+v[i+1][j-1])
			+ alpha * ( fabs(v[i][j]+v[i+1][j]) * (u[i][j]-u[i][j+1]) - fabs(v[i][j-1]+v[i+1][j-1]) * (u[i][j-1]-u[i][j]) )
	                       )/dy/4.0;
}

void calculate_fg(
  double Re,
  double GX,
  double GY,
  double alpha,
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **F,
  double **G, 
  int **Flag
){
	int i,j;
	
	/* boundary conditions */
	for(j=1; j<=jmax; j++){
		F[ 0 ][ j ] = U[ 0 ][ j ];
		F[ imax ][ j ] = U[ imax ][ j ];
	}

	for(i=1; i<=imax; i++){
		G[i][0] = V[i][0];
		G[i][jmax] = V[i][jmax];
	}
	
	/* inner values */
	for(i=1; i<=imax-1; i++)
		for(j=1; j<=jmax; j++)
			if(Flag[i][j]==C_F && Flag[i+1][j]==C_F)
				F[i][j] = U[i][j] + dt * (
						(d2dx(U,i,j,dx) + d2dy(U,i,j,dy))/Re - du2dx(U, i, j, dx, alpha) - duvdy(U,V,i,j,dy, alpha) + GX
						);

	for(i=1; i<=imax; i++)
		for(j=1; j<=jmax-1; j++)
			if(Flag[i][j]==C_F && Flag[i][j+1]==C_F)
				G[i][j] = V[i][j] + dt * (
						(d2dx(V,i,j,dx) + d2dy(V,i,j,dy))/Re - duvdx(U, V, i, j, dx, alpha) - dv2dy(V,i,j,dy, alpha) + GY
						);

}


void calculate_rs(
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **F,
  double **G,
  double **RS,
  int **Flag
){
	int i,j;

	for(i=1; i<=imax; i++)
		for(j=1; j<=jmax; j++)
			if( IS_FLUID(Flag[i][j]) )
				RS[i][j] = 1/dt*((F[i][j]-F[i-1][j])/dx + (G[i][j]-G[i][j-1])/dy) ;

}

void calculate_dt(
  double Re,
  double tau,
  double *dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V
){
	*dt = tau * fmin( fmin( Re/2/(1/(dx*dx) + 1/(dy*dy)), dx/fmatrix_max( U, 0, imax+1, 0, jmax+1 ) ),
			   dy/fmatrix_max(V,0,imax+1,0,jmax+1)
			);
}


void calculate_uv(
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **F,
  double **G,
  double **P,
  int **Flag
){
	int i,j;

	for(i=1; i<=imax-1; i++)
		for(j=1; j<=jmax; j++)
			if(Flag[i][j]==C_F && Flag[i+1][j]==C_F)
				U[i][j] = F[i][j] - dt/dx*(P[i+1][j]-P[i][j]);

	for(i=1; i<=imax; i++)
		for(j=1; j<=jmax-1; j++)
			if(Flag[i][j]==C_F && Flag[i][j+1]==C_F)
				V[i][j] = G[i][j] - dt/dy*(P[i][j+1]-P[i][j]);
}

//__________________________________________________________________//
//								      //
//			CFD SOR Functions			      //
//								      //
//__________________________________________________________________//

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