//______________________________________________________________//
//								  //
//	      functionalities.cu includes 		   	  //
//		all the CUDA functions 				  //
//______________________________________________________________//

#include "functionalities.h"




//================Functions for the inpainting ===============//


__global__ void global_vorticity( float *imgU, float *imgV, float *imgVorticity, int w, int h, int nc, int n, int iter )
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
		// ( iter & 1 ) chanigin the sign depending on the iteration number
		/* if ( iter & 1 )
		{		
			imgVorticity[ind] = ( dVdx - dUdy );
		}
		else
		{
			imgVorticity[ind] = -( dVdx - dUdy );
		} */
		imgVorticity[ind] = ( dVdx - dUdy );


	}
}


__global__ void global_solve_Poisson (float *imgOut, float *imgIn, float *initVorticity, float *rhs, int *imgDomain, int w, int h, int nc, int n, float sor_theta, int redOrBlack)
{
	
	float dh = 0.5;
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
				f = dh*dh*rhs[ind];
			}
			else
			{
				f = dh*dh*initVorticity[ind];
				//f = -dh*dh*rhs[ind];
			}
		}
		else
		{
			f = 0.0f;
		}			    
		// TODO: Think about the sign!!!
		float val = -( f - (upx + umx + upy + umy) ) / 4.0;
		//float val = ((upx + umx + upy + umy) ) / 4.0;
		val = sor_theta*val + (1.0-sor_theta)*u0;

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

//========================= Functions for anisotropic diffusion ==============================

__global__ void update_aniso_diff(float *imgIn, float *divergence, int *imgDomain, float *imgOut, float timestep, int w, int h, int nc, int n)
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if (ind<n)
	{
		if (imgDomain[x + (size_t)w*y] == 1)
		{ 
			imgOut[ind] = imgIn[ind] + timestep*divergence[ind];
		}
		else
		{
			imgOut[ind] = imgIn[ind];
		}
	}

}

__host__ __device__ float g_dash(float s)
{
	float eps = 0.01;
	//return 1.0f;
	return (1.0f/max(eps, s));
	//return expf(-s*s/eps)/eps;
}

__global__ void global_diffusivity(float *v1, float *v2, float *diffusivity, int w, int h, int nc, int n)
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if (ind<n)
	{ 
		diffusivity[ind] = g_dash(sqrtf( v1[ind]*v1[ind] + v2[ind]*v2[ind]));
	}

}

__global__ void mult_scal_vec(float *scal_field, float *vec_field, int w, int h, int nc, int n)
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if (ind<n)
	{ 
		vec_field[ind] = scal_field[ind]*vec_field[ind];
		//vec_field[ind+w*h] = scal_field[ind]*vec_field[ind+w*h];
		//vec_field[ind+2*w*h] = scal_field[ind]*vec_field[ind+2*w*h];
	}

}  


__device__ float aniso_cuda_diff_x(float a, float b, int x, int w)
{

	if (x+1<w)
	{
		return (a - b);
	}
	else
	{
		return 0.0f;
	}
	
}

__device__ float aniso_cuda_diff_y(float a, float b, int y, int h)
{

	if (y+1<h)
	{
		return (a - b);
	}
	else
	{
		return 0.0f;
	}
	
}

__global__ void aniso_global_grad(float *imgIn, float *v1, float *v2, int w, int h, int nc, int n)
{

	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;	

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if (ind<n)
	{ 

		v1[ind] = aniso_cuda_diff_x(imgIn[ind+1], imgIn[ind], x, w);
		v2[ind] = aniso_cuda_diff_y(imgIn[ind+w], imgIn[ind], y, h);
	}
}


__device__ float aniso_cuda_div_x(float a, float b, int x, int w)
{
		if ((x+1<w) && (x>0))
		{
			return (a - b);
		}
		else if (x+1<w)
		{
			return (a - 0);
		}
		else if (x>0)
		{
			return (0 - b);
		}
		else
		{
			return 0.;
		}
}


__device__ float aniso_cuda_div_y(float a, float b, int y, int h)
{
		if ((y+1<h) && (y>0))
		{
			return (a - b);
		}
		else if (y+1<h)
		{
			return (a - 0);
		}
		else if (y>0)
		{
			return (0 - b);
		}
		else
		{
			return 0.;
		}
}

__global__ void aniso_global_div(float *v1, float *v2, float *imgOut, int w, int h, int nc, int n)
{

	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if ((ind<n) && (ind-w>=0) && (ind-1>=0)) 
	{ 	
		imgOut[ind] = cuda_div_x(v1[ind], v1[ind-1], x, w) + cuda_div_y(v2[ind], v2[ind-w], y, h);
	}

}

__global__ void aniso_global_norm(float *imgIn, float *imgOut, int w, int h, int n)
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	if (ind<n)
	{ 
		imgOut[ind] = imgIn[ind]*imgIn[ind];
		//imgOut[ind] += imgIn[ind+w*h]*imgIn[ind+w*h];
		//imgOut[ind] += imgIn[ind+2*w*h]*imgIn[ind+2*w*h];
		imgOut[ind] = sqrtf(imgOut[ind]);
	}
}

/*
void aniso_diff(float *imgIn, int *imgDomain, float *imgOut, int w, int h, int nc, float tau, int N)
{

	float *v1 = new float[(size_t)w*h*nc];
	float *v2 = new float[(size_t)w*h*nc];
	float *divergence = new float[(size_t)w*h*nc];
	float *diffusivity = new float[(size_t)w*h];
	float *imgOutGray = new float[(size_t)w*h];

	int n = w*h*nc;

	for (int i=0; i<N; i++)
	{
		// allocate GPU memory
		float *gpu_In, *gpu_v1, *gpu_v2, *gpu_Out, *gpu_Out_Gray;
		int *gpu_Domain;

		cudaMalloc(&gpu_In, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v1, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v2, n*sizeof(float));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel
		dim3 block = dim3(128,1,1);

		dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
		aniso_global_grad <<<grid,block>>> (gpu_In, gpu_v1, gpu_v2, w, h, nc, n);

		// copy result back to host (CPU) memory
		cudaMemcpy(v1, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		cudaMemcpy(v2, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_In);
		CUDA_CHECK;
		cudaFree(gpu_v1);
		CUDA_CHECK;
		cudaFree(gpu_v2);
		CUDA_CHECK;


		// Calculate diffusivity and multiply by gradient

		// allocate GPU memory
		float *gpu_diffusivity;

		cudaMalloc(&gpu_v1, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v2, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_diffusivity, w*h*sizeof(float));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_diffusivity, diffusivity, w*h*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel
		global_diffusivity <<<grid,block>>> (gpu_v1, gpu_v2, gpu_diffusivity, w, h, nc, w*h);
		mult_scal_vec <<<grid,block>>> (gpu_diffusivity, gpu_v1, w, h, nc, w*h);
		mult_scal_vec <<<grid,block>>> (gpu_diffusivity, gpu_v2, w, h, nc, w*h);

		// copy result back to host (CPU) memory
		cudaMemcpy(diffusivity, gpu_diffusivity, w*h * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		// copy result back to host (CPU) memory
		cudaMemcpy(v1, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		cudaMemcpy(v2, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;


		// free device (GPU) memory
		cudaFree(gpu_diffusivity);
		CUDA_CHECK;
		cudaFree(gpu_v1);
		CUDA_CHECK;
		cudaFree(gpu_v2);
		CUDA_CHECK;





		// Calculate divergence of a gradient

		// allocate GPU memory
		float *gpu_divergence;

		cudaMalloc(&gpu_v1, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v2, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_divergence, n*sizeof(float));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_divergence, divergence, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel
		//dim3 block = dim3(128,1,1);

		//dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
		aniso_global_div <<<grid,block>>> (gpu_v1, gpu_v2, gpu_divergence, w, h, nc, n);

		// copy result back to host (CPU) memory
		cudaMemcpy(divergence, gpu_divergence, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_v1);
		CUDA_CHECK;
		cudaFree(gpu_v2);
		CUDA_CHECK;
		cudaFree(gpu_divergence);
		CUDA_CHECK;

		// Update image

		// allocate GPU memory
		//float *gpu_In, *gpu_Out_Gray;

		cudaMalloc(&gpu_In, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_Out, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_divergence, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_Domain, n*sizeof(int));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_divergence, divergence, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_Domain, imgDomain, n*sizeof(int), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel

		update_aniso_diff <<<grid,block>>> (gpu_In, gpu_divergence, gpu_Domain, gpu_Out, tau, w, h, nc, n);

		// copy result back to host (CPU) memory
		cudaMemcpy(imgOut, gpu_Out, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		cudaMemcpy(imgIn, gpu_Out, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_In);
		CUDA_CHECK;
		cudaFree(gpu_divergence);
		CUDA_CHECK;
		cudaFree(gpu_Domain);
		CUDA_CHECK;
		cudaFree(gpu_Out);
		CUDA_CHECK;
	}

    delete[] v1;
    delete[] v2;
    delete[] diffusivity;
    delete[] divergence;

}
*/

void aniso_diff(float *imgIn, int *imgDomain, float *imgOut, int w, int h, int nc, float tau, int N, dim3 grid, dim3 block)
{

	float *v1 = new float[(size_t)w*h*nc];
	float *v2 = new float[(size_t)w*h*nc];
	float *divergence = new float[(size_t)w*h*nc];
	float *diffusivity = new float[(size_t)w*h];
	float *imgOutGray = new float[(size_t)w*h];

	int n = w*h*nc;

	for (int i=0; i<N; i++)
	{
		// allocate GPU memory
		float *gpu_In, *gpu_v1, *gpu_v2, *gpu_Out, *gpu_Out_Gray;
		int *gpu_Domain;

		cudaMalloc(&gpu_In, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v1, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v2, n*sizeof(float));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel
		//dim3 block = dim3(128,1,1);

		//dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
		aniso_global_grad <<<grid,block>>> (gpu_In, gpu_v1, gpu_v2, w, h, nc, n);

		// copy result back to host (CPU) memory
		cudaMemcpy(v1, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		cudaMemcpy(v2, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_In);
		CUDA_CHECK;
		cudaFree(gpu_v1);
		CUDA_CHECK;
		cudaFree(gpu_v2);
		CUDA_CHECK;


		// Calculate diffusivity and multiply by gradient

		// allocate GPU memory
		float *gpu_diffusivity;

		cudaMalloc(&gpu_v1, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v2, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_diffusivity, w*h*sizeof(float));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_diffusivity, diffusivity, w*h*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel
		global_diffusivity <<<grid,block>>> (gpu_v1, gpu_v2, gpu_diffusivity, w, h, nc, w*h);
		mult_scal_vec <<<grid,block>>> (gpu_diffusivity, gpu_v1, w, h, nc, w*h);
		mult_scal_vec <<<grid,block>>> (gpu_diffusivity, gpu_v2, w, h, nc, w*h);

		// copy result back to host (CPU) memory
		cudaMemcpy(diffusivity, gpu_diffusivity, w*h * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		// copy result back to host (CPU) memory
		cudaMemcpy(v1, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		cudaMemcpy(v2, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;


		// free device (GPU) memory
		cudaFree(gpu_diffusivity);
		CUDA_CHECK;
		cudaFree(gpu_v1);
		CUDA_CHECK;
		cudaFree(gpu_v2);
		CUDA_CHECK;

		// Calculate divergence of a gradient

		// allocate GPU memory
		float *gpu_divergence;

		cudaMalloc(&gpu_v1, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_v2, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_divergence, n*sizeof(float));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_divergence, divergence, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel
		//dim3 block = dim3(128,1,1);

		//dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);
		aniso_global_div <<<grid,block>>> (gpu_v1, gpu_v2, gpu_divergence, w, h, nc, n);

		// copy result back to host (CPU) memory
		cudaMemcpy(divergence, gpu_divergence, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_v1);
		CUDA_CHECK;
		cudaFree(gpu_v2);
		CUDA_CHECK;
		cudaFree(gpu_divergence);
		CUDA_CHECK;

		// Update image

		// allocate GPU memory
		//float *gpu_In, *gpu_Out_Gray;

		cudaMalloc(&gpu_In, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_Out, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_divergence, n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&gpu_Domain, n*sizeof(int));
		CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_divergence, divergence, n*sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(gpu_Domain, imgDomain, n*sizeof(int), cudaMemcpyHostToDevice);
		CUDA_CHECK;

		// launch kernel

		update_aniso_diff <<<grid,block>>> (gpu_In, gpu_divergence, gpu_Domain, gpu_Out, tau, w, h, nc, n);

		// copy result back to host (CPU) memory
		cudaMemcpy(imgOut, gpu_Out, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;
		cudaMemcpy(imgIn, gpu_Out, n * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_In);
		CUDA_CHECK;
		cudaFree(gpu_divergence);
		CUDA_CHECK;
		cudaFree(gpu_Domain);
		CUDA_CHECK;
		cudaFree(gpu_Out);
		CUDA_CHECK;
	}

    delete[] v1;
    delete[] v2;
    delete[] diffusivity;
    delete[] divergence;

}


