#ifndef __FUNCTIONS_H_
#define __FUNCTIONS_H_


//__________________________________________________________________//
//								      //
//		CFD Initialization Functions			      //
//								      //
//__________________________________________________________________//

int read_parameters( const char *szFileName,       /* name of the file 		*/
                     double *Re,                   /* reynolds number   	 	*/
                     double *PI,                   /* pressure 			*/
                     double *GX,                   /* gravitation x-direction 	*/
                     double *GY,                   /* gravitation y-direction 	*/
                     double *t_end,                /* end time 			*/
                     double *xlength,              /* length of the domain x-dir.	*/
                     double *ylength,              /* length of the domain y-dir.	*/
                     double *dt,                   /* time step 			*/
                     double *dx,                   /* length of a cell x-dir. 	*/
                     double *dy,                   /* length of a cell y-dir. 	*/
                     int    imax,                  /* number of cells x-direction	*/
                     int    jmax,                  /* number of cells y-direction	*/
                     double *alpha,                /* uppwind differencing factor	*/
                     double *omg,                  /* relaxation factor 	 	*/
                     double *tau,                  /* safety factor for time step	*/
                     int    *itermax,              /* max. number of iterations  	*/
		                                    /* for pressure per time step 	*/
                     double *eps,                  /* accuracy bound for pressure	*/
                     double *dt_value,             /* time for output 		*/
                     double *dp		    /* dp/dx gradient of pressure 	*/
		   );

void init_uv( int imax, 
	      int jmax, 
	      double **U, 
	      double **V, 
	      float *imgU, 
	      float *imgV, 
	      int **Flag, 
	      int iter
	    );

void init_flag( const int  imax, 
		 const int  jmax, 
		 int        **Flag, 
		 int        *imgDomain 
	      );
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
		   );
//__________________________________________________________________//
//								      //
//			CFD UVP Functions 			      //
//								      //
//__________________________________________________________________//

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
);


/**
 * This operation computes the right hand side of the pressure poisson equation.
 * The right hand side is computed according to the formula
 *
 * @f$ rs = \frac{1}{\delta t} \left( \frac{F^{(n)}_{i,j}-F^{(n)}_{i-1,j}}{\delta x} + \frac{G^{(n)}_{i,j}-G^{(n)}_{i,j-1}}{\delta y} \right)  @f$
 *
 */
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
);


/**
 * Determines the maximal time step size. The time step size is restricted
 * accordin to the CFL theorem. So the final time step size formula is given
 * by
 *
 * @f$ {\delta t} := \tau \, \min\left( \frac{Re}{2}\left(\frac{1}{{\delta x}^2} + \frac{1}{{\delta y}^2}\right)^{-1},  \frac{{\delta x}}{|u_{max}|},\frac{{\delta y}}{|v_{max}|} \right) @f$
 *
 */
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
);


/**
 * Calculates the new velocity values according to the formula
 *
 * @f$ u_{i,j}^{(n+1)}  =  F_{i,j}^{(n)} - \frac{\delta t}{\delta x} (p_{i+1,j}^{(n+1)} - p_{i,j}^{(n+1)}) @f$
 * @f$ v_{i,j}^{(n+1)}  =  G_{i,j}^{(n)} - \frac{\delta t}{\delta y} (p_{i,j+1}^{(n+1)} - p_{i,j}^{(n+1)}) @f$
 *
 * As always the index range is
 *
 * @f$ i=1,\ldots,imax-1, \quad j=1,\ldots,jmax @f$
 * @f$ i=1,\ldots,imax, \quad j=1,\ldots,jmax-1 @f$
 *
 * @image html calculate_uv.jpg
 */
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
);

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
);
#endif