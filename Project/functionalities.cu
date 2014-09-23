//______________________________________________________________//
//	      functionalities.cu includes 			//
//		all the CUDA functions 				//
//______________________________________________________________//

#include "functionalities.h"



__global__ void global_vorticity( float *imgU, float *imgV, float *imgVorticity, int w, int h, int nc, int n )
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y, ch;	

	float dVdx, dUdy;

	//ch = ind / w*h;
	ch = 0;
	y = ( ind - ch*w*h ) / w;
	x = ( ind - ch*w*h ) % w;

	if( ind < n )
	{ 

		//v1[ind] = cuda_diff_x(imgIn[ind+1], imgIn[ind], x, w);
		//v2[ind] = cuda_diff_y(imgIn[ind+w], imgIn[ind], y, h);

		dVdx = (1./32.)*(3*imgV[max(min(w-1, x+1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] + 10*imgV[max(min(w-1, x+1), 0) + w*max(min(h-1,y),0) + ch*w*h] + 3*imgV[max(min(w-1, x+1), 0) + w*max(min(h-1,y-1),0) + ch*w*h] - 3*imgV[max(min(w-1, x-1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] - 10*imgV[max(min(w-1, x-1), 0) + w*max(min(h-1,y),0) + ch*w*h] - 3*imgV[max(min(w-1, x-1), 0) + w*max(min(h-1,y-1),0) + ch*w*h]);

		dUdy = (1./32.)*(3*imgU[max(min(w-1, x+1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] + 10*imgU[max(min(w-1, x), 0) + w*max(min(h-1,y+1),0) + ch*w*h] + 3*imgU[max(min(w-1, x-1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] - 3*imgU[max(min(w-1, x+1), 0) + w*max(min(h-1,y-1),0) + ch*w*h] - 10*imgU[max(min(w-1, x), 0) + w*max(min(h-1,y-1),0) + ch*w*h] - 3*imgU[max(min(w-1, x-1), 0) + w*max(min(h-1,y-1),0) + ch*w*h]);

		imgVorticity[ind] = dVdx - dUdy;


	}
}


__global__ void global_solve_Poisson (float *imgOut, float *imgIn, float *initVorticity, float *rhs, int *imgDomain, int w, int h, int nc, int n, float sor_theta, int redOrBlack)
{
	
	float dh = 1.0;
	float f;
	
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y, ch;
	//ch = ind / w*h;
	ch = 0;
	
	x = ( ind - ch*w*h ) % w;
 	y = ( ind - ch*w*h ) / w;

	if ( ind<n ) 
	{ 	
	    bool isActive = ((x<w && y<h) && (((x+y)%2)==redOrBlack));
	    //bool isActive = (x<w && y<h); //&& (((x+y)%2)==redOrBlack));
	    

	    if ( (isActive) && (imgDomain[x + (size_t)w*y] == 1) )
	    {

		    float u0  = imgIn[ind];
		    float upx = (x+1<w?  imgIn[x+1 + (size_t)w*(y  ) + w*h*ch] : u0);
		    float umx = (x-1>=0? imgIn[x-1 + (size_t)w*(y  ) + w*h*ch] : u0);
		    float upy = (y+1<h?  imgIn[x   + (size_t)w*(y+1) + w*h*ch] : u0);
		    float umy = (y-1>=0? imgIn[x   + (size_t)w*(y-1) + w*h*ch] : u0);

		if (imgDomain[ind] == 1)
		{
			if ((imgDomain[ind+1] == 1) && (imgDomain[ind-1] == 1) && (imgDomain[ind+w] == 1) && (imgDomain[ind-w] == 1))
			{
				f = -dh*dh*rhs[ind];
			}
			else
			{
				f = -dh*dh*initVorticity[ind];
			}
		}
		else
		{
			f = 0.0f;
		}			    
			// TODO: Think about the sign!!!
		    float val = ( f + (upx + umx + upy + umy) ) / 4.0;
		    //float val = ((upx + umx + upy + umy) ) / 4.0;
		    val = val + sor_theta*(val-u0);

		    imgOut[ind] = val;
	    }


	}

}


__device__ float cuda_diff_x( float a, float b, int x, int w )
{
	if( x+1 < w ) return (a - b);
	else return 0.0f;
}

__device__ float cuda_diff_y( float a, float b, int y, int h )
{
	if( y+1 < h ) return (a - b);
	else return 0.0f;
}

__global__ void global_grad( float *imgIn, float *v1, float *v2, int w, int h, int nc, int n )
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y, ch;	

	//ch = ind / w*h;
	ch = 0;
	y = ( ind - ch*w*h ) / w;
	x = ( ind - ch*w*h ) % w;

	if( ind < n )
	{ 

		//v1[ind] = cuda_diff_x(imgIn[ind+1], imgIn[ind], x, w);
		//v2[ind] = cuda_diff_y(imgIn[ind+w], imgIn[ind], y, h);

		v1[ind] = (1./32.)*(3*imgIn[max(min(w-1, x+1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] + 10*imgIn[max(min(w-1, x+1), 0) + w*max(min(h-1,y),0) + ch*w*h] + 3*imgIn[max(min(w-1, x+1), 0) + w*max(min(h-1,y-1),0) + ch*w*h] - 3*imgIn[max(min(w-1, x-1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] - 10*imgIn[max(min(w-1, x-1), 0) + w*max(min(h-1,y),0) + ch*w*h] - 3*imgIn[max(min(w-1, x-1), 0) + w*max(min(h-1,y-1),0) + ch*w*h]);

		v2[ind] = (1./32.)*(3*imgIn[max(min(w-1, x+1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] + 10*imgIn[max(min(w-1, x), 0) + w*max(min(h-1,y+1),0) + ch*w*h] + 3*imgIn[max(min(w-1, x-1), 0) + w*max(min(h-1,y+1),0) + ch*w*h] - 3*imgIn[max(min(w-1, x+1), 0) + w*max(min(h-1,y-1),0) + ch*w*h] - 10*imgIn[max(min(w-1, x), 0) + w*max(min(h-1,y-1),0) + ch*w*h] - 3*imgIn[max(min(w-1, x-1), 0) + w*max(min(h-1,y-1),0) + ch*w*h]);
	}
}


__device__ float cuda_div_x( float a, float b, int x, int w )
{
		if( ( x+1 < w ) && ( x > 0 ) ) return ( a - b );
		else if( x+1 < w ) return ( a - 0 );
		else if( x > 0 ) return ( 0 - b );
		else return 0.0f;
}


__device__ float cuda_div_y( float a, float b, int y, int h )
{
		if( ( y+1 < h ) && ( y > 0 ) ) return ( a - b );
		else if ( y+1 < h ) return ( a - 0 );
		else if ( y > 0 ) return ( 0 - b );
		else return 0.0f;
}

__global__ void global_div( float *v1, float *v2, float *imgOut, int w, int h, int nc, int n )
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	int x, y, ch;
	ch = ind / w*h;
	
	x = ( ind - ch*w*h ) % w;
 	y = ( ind - ch*w*h ) / w;

	if( ( ind<n ) && ( ind-w >= 0) && ( ind-1 >= 0 ) ) 
	{ 	
		imgOut[ind] = cuda_div_x( v1[ind], v1[ind-1], x, w ) + cuda_div_y( v2[ind], v2[ind-w], y, h );
	}
}


__global__ void global_norm( float *imgIn, float *imgOut, int w, int h, int n )
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	if( ind < n )
	{ 
		imgOut[ind] = imgIn[ind]*imgIn[ind];
		//imgOut[ind] += imgIn[ind+w*h]*imgIn[ind+w*h];
		//imgOut[ind] += imgIn[ind+2*w*h]*imgIn[ind+2*w*h];
		imgOut[ind] = sqrtf(imgOut[ind]);
	}
}

__device__ int check_color( float *c, float r, float g, float b )
{
	float eps = 0.0001;

	if( ( fabsf( r-c[0] ) < eps) && ( fabsf( g-c[1] ) < eps) && ( fabsf( b-c[2] ) < eps ) ) return 1;
	else return 0;	
}

/*
__global__ void global_detect_domain( float *imgIn, int *imgDomain, int w, int h, int n )
{
	float c[3] = {1.0f, 0.0f, 0.0f};
	// For looping around a pixel
	int neighbour[8]={ 1, -1, w, -w, -w-1, -w+1, w-1, w+1 };
	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	if( ind < n )
	{
		if( check_color( c, imgIn[ind], imgIn[ind+w*h], imgIn[ind+2*w*h] ) )
		{
			imgDomain[ind] = FLUID;
			for( int i = 0; i < 8; i++ )
			{
				//TODO: Check if ind+neighbour[i] is in the domain!
				if( check_color( c, imgIn[ind+neighbour[i]], imgIn[ind+w*h+neighbour[i]], imgIn[ind+2*w*h+neighbour[i]] ) != 1 )
				{
					imgDomain[ind+neighbour[i]] = INFLOW;
				}
			}
		}
		else imgDomain[ind] = OBSTACLE;
	}
}
*/

__global__ void global_detect_domain( float *imgMask, int *imgDomain, int w, int h, int n )
{
	float c = 1.0;
	// For looping around a pixel
	int neighbour[8]={ 1, -1, w, -w, -w-1, -w+1, w-1, w+1 };
	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	float eps = 0.0001;

	if( ind < n )
	{
		if ( fabsf( imgMask[ind]-c ) < eps )
		{
			imgDomain[ind] = FLUID;
			for( int i = 0; i < 8; i++ )
			{
				//TODO: Check if ind+neighbour[i] is in the domain!
				if ( fabsf( imgMask[ind+neighbour[i]]-c ) > eps )
				{
					imgDomain[ind+neighbour[i]] = INFLOW;
				}
			}
		}
		else imgDomain[ind] = OBSTACLE;
	}
}

