// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2014, September 8 - October 10
// ###
// ###
// ### Maria Klodt, Jan Stuehmer, Mohamed Souiai, Thomas Moellenhoff
// ###
// ###

// ###
// ###
// ### TODO: For every student of your group, please provide here:
// ###
// ### name, email, login username (for example p123)
// ###
// ###


#include "aux.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA


__device__ float cuda_diff_x(float a, float b, int x, int w)
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

__device__ float cuda_diff_y(float a, float b, int y, int h)
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

__global__ void global_grad(float *imgIn, float *v1, float *v2, int w, int h, int nc, int n)
{

	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;	

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if (ind<n)
	{ 

		v1[ind] = cuda_diff_x(imgIn[ind+1], imgIn[ind], x, w);
		v2[ind] = cuda_diff_y(imgIn[ind+w], imgIn[ind], y, h);

	}

	

}


__device__ float cuda_div_x(float a, float b, int x, int w)
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


__device__ float cuda_div_y(float a, float b, int y, int h)
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

__global__ void global_div(float *v1, float *v2, float *imgOut, int w, int h, int nc, int n)
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




__global__ void global_norm(float *imgIn, float *imgOut, int w, int h, int n)
{
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	if (ind<n)
	{ 
		imgOut[ind] = imgIn[ind]*imgIn[ind];
		imgOut[ind] += imgIn[ind+w*h]*imgIn[ind+w*h];
		imgOut[ind] += imgIn[ind+2*w*h]*imgIn[ind+2*w*h];
		imgOut[ind] = sqrtf(imgOut[ind]);
	}
}

__device__ int check_color(float *c, float r, float g, float b)
{

	float  eps = 0.0001;


	if ( (fabsf(r-c[0])<eps) && (fabsf(g-c[1])<eps) &&  (fabsf(b-c[2])<eps) )
	{
		return 1;
	}
	else
	{
		return 0;
	}	
}

__global__ void global_detect_domain(float *imgIn, float *imgDomain, int w, int h, int n)
{

	float c[3] = {1.0f, 0.0f, 0.0f};
	
	// For looping around a pixel
	int neighbour[8]={1, -1, w, -w, -w-1, -w+1, w-1, w+1};

	int ind = threadIdx.x + blockDim.x * blockIdx.x;

	int x, y, ch;

	ch = (int)(ind) / (int)(w*h);
	y = (ind - ch*w*h) / (int)w;
	x = (ind - ch*w*h) % (int)w;

	if (ind<n)
	{

		if (check_color(c, imgIn[ind], imgIn[ind+w*h], imgIn[ind+2*w*h]))
		{
			imgDomain[ind] = 1.0f;
			for (int i=0; i<8; i++)
			{
				//TODO: Check if ind+neighbour[i] is in the domain!
				if ( check_color(c, imgIn[ind+neighbour[i]], imgIn[ind+w*h+neighbour[i]], imgIn[ind+2*w*h+neighbour[i]]) != 1 )
				{
					imgDomain[ind+neighbour[i]] = 0.5f;
				}
			}
		}
		else
		{
			imgDomain[ind] = 0.0f;
		}
	}
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> <gamma> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    




    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    //cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t)w*h*nc];
	float *v1 = new float[(size_t)w*h*nc];
	float *v2 = new float[(size_t)w*h*nc];
	float *imgVorticity = new float[(size_t)w*h*mOut.channels()];
	float *imgDomain = new float[(size_t)w*h];
	// TODO: Temporarly we consider just a grayscale inpainting
	float *U = new float[(size_t)w*h];
	float *V = new float[(size_t)w*h];


    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);


    Timer timer; timer.start();
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###

	int n = w*h*nc, n2=w*h;

	// Calculate gradient

	// allocate GPU memory
	float *gpu_In, *gpu_v1, *gpu_v2, *gpu_Out, *gpu_Vorticity, *gpu_Domain, *gpu_U, *gpu_V;

	cudaMalloc(&gpu_In, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_v1, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_v2, n*sizeof(float));
	CUDA_CHECK;
	// TODO: Temporarly we consider just a grayscale inpainting
	cudaMalloc(&gpu_U, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_V, w*h*sizeof(float));
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
	global_grad <<<grid,block>>> (gpu_In, gpu_v1, gpu_v2, w, h, nc, n);
	global_norm <<<grid,block>>> (gpu_v1, gpu_V, w, h, w*h);
	global_norm <<<grid,block>>> (gpu_v2, gpu_U, w, h, w*h);

	// copy result back to host (CPU) memory
	cudaMemcpy(v1, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;
	cudaMemcpy(v2, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;
	cudaMemcpy(U, gpu_U, w*h * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;
	cudaMemcpy(V, gpu_V, w*h * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_In);
	CUDA_CHECK;
	cudaFree(gpu_v1);
	CUDA_CHECK;
	cudaFree(gpu_v2);
	CUDA_CHECK;
	cudaFree(gpu_U);
	CUDA_CHECK;
	cudaFree(gpu_V);
	CUDA_CHECK;

	// Invert the V values according t: V = -dI/dx
	// TODO: Temporarly we consider just a grayscale inpainting 
	for (int i=0; i<w*h; i++)
	{
		V[i] = -V[i];
	}

	
	// Calculate divergence of a gradient

	cudaMalloc(&gpu_v1, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_v2, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_Out, n*sizeof(float));
	CUDA_CHECK;

	// copy host memory to device
	cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(gpu_Out, imgOut, n*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// launch kernel
	global_div <<<grid,block>>> (gpu_v1, gpu_v2, gpu_Out, w, h, nc, n);

	// copy result back to host (CPU) memory
	cudaMemcpy(imgOut, gpu_Out, n * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_v1);
	CUDA_CHECK;
	cudaFree(gpu_v2);
	CUDA_CHECK;
	cudaFree(gpu_Out);
	CUDA_CHECK;


	// Calculate norm	
	// allocate GPU memory

	cudaMalloc(&gpu_In, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_Vorticity, n2*sizeof(float));
	CUDA_CHECK;

	// copy host memory to device
	cudaMemcpy(gpu_In, imgOut, n*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(gpu_Vorticity, imgVorticity, n2*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// launch kernel
	global_norm <<<grid,block>>> (gpu_In, gpu_Vorticity, w, h, n2);

	// copy result back to host (CPU) memory
	cudaMemcpy(imgVorticity, gpu_Vorticity, n2 * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_In);
	CUDA_CHECK;
	cudaFree(gpu_Vorticity);
	CUDA_CHECK;



	// Calculate the inpainting domain	
	// allocate GPU memory

	cudaMalloc(&gpu_In, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&gpu_Domain, w*h*sizeof(float));
	CUDA_CHECK;

	// copy host memory to device
	cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// launch kernel
	global_detect_domain <<<grid,block>>> (gpu_In, gpu_Domain, w, h, w*h);

	// copy result back to host (CPU) memory
	cudaMemcpy(imgDomain, gpu_Domain, w*h * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_In);
	CUDA_CHECK;
	cudaFree(gpu_Domain);
	CUDA_CHECK;



    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;



    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, v2);
    showImage("Output1", mOut, 100+w+40, 100);


    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgVorticity;
    delete[] imgDomain;
    delete[] v1;
    delete[] v2;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



