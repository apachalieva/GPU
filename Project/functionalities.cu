//______________________________________________________________//
//	      functionalities.cu includes 			//
//		all the CUDA functions 				//
//______________________________________________________________//

#include "functionalities.h"

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

	ch = ind / w*h;
	y = ( ind - ch*w*h ) / w;
	x = ( ind - ch*w*h ) % w;

	if( ind < n )
	{ 
		v1[ind] = cuda_diff_x(imgIn[ind+1], imgIn[ind], x, w);
		v2[ind] = cuda_diff_y(imgIn[ind+w], imgIn[ind], y, h);
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
		imgOut[ind] += imgIn[ind+w*h]*imgIn[ind+w*h];
		imgOut[ind] += imgIn[ind+2*w*h]*imgIn[ind+2*w*h];
		imgOut[ind] = sqrtf(imgOut[ind]);
	}
}

__device__ int check_color( float *c, float r, float g, float b )
{
	float eps = 0.0001;

	if( ( fabsf( r-c[0] ) < eps) && ( fabsf( g-c[1] ) < eps) && ( fabsf( b-c[2] ) < eps ) ) return 1;
	else return 0;	
}

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

