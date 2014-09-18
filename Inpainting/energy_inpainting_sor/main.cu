// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2013/2014, March 3 - April 4
// ###
// ###

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "aux.h"
using namespace std;

// uncomment to use the camera
//#define CAMERA



__host__ __device__
float func_diffusivity (float x, float eps)
{
    return 1/max(x,eps);
    //return expf(-x*x/eps)/2/eps;
}



__global__
void cuda_minimizeEnergy_sor_step (float *uOut, const float *uIn, const float *diffusivity, const bool *mask_toInpaint, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    bool isActive = ((x<w && y<h) && (((x+y)%2)==redOrBlack));
    
    if (isActive) isActive = isActive && mask_toInpaint[x + (size_t)w*y];

    if (isActive)
    {
        float d0  = diffusivity[x + (size_t)w*y];
        float d0x = (x+1<w?  d0 : 0);
        float d0y = (y+1<h?  d0 : 0);
        float dmx = (x-1>=0? diffusivity[x-1 + (size_t)w*(y  )] : d0);
        float dmy = (y-1>=0? diffusivity[x   + (size_t)w*(y-1)] : d0);

        size_t nOmega = (size_t)w*h;
        for(int c=0; c<nc; c++)
        {
            size_t pt = x + (size_t)w*y + nOmega*c;

            float u0  = uIn[pt];
            float upx = (x+1<w?  uIn[x+1 + (size_t)w*(y  ) + nOmega*c] : u0);
            float umx = (x-1>=0? uIn[x-1 + (size_t)w*(y  ) + nOmega*c] : u0);
            float upy = (y+1<h?  uIn[x   + (size_t)w*(y+1) + nOmega*c] : u0);
            float umy = (y-1>=0? uIn[x   + (size_t)w*(y-1) + nOmega*c] : u0);
            
            float val = (d0x*upx + dmx*umx + d0y*upy + dmy*umy) / (d0x + dmx + d0y + dmy);
            val = val + sor_theta*(val-u0);

            uOut[pt] = val;
        }
    }
}


__global__
void cuda_minimizeEnergy_jacobi_step (float *uOut, const float *uIn, const float *diffusivity, const bool *mask_toInpaint, int w, int h, int nc, float lambda)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    bool isActive = (x<w && y<h);

    if (isActive) isActive = isActive && mask_toInpaint[x + (size_t)w*y];

    if (isActive)
    {
        float d0  = diffusivity[x + (size_t)w*y];
        float d0x = (x+1<w?  d0 : 0);
        float d0y = (y+1<h?  d0 : 0);
        float dmx = (x-1>=0? diffusivity[x-1 + (size_t)w*(y  )] : d0);
        float dmy = (y-1>=0? diffusivity[x   + (size_t)w*(y-1)] : d0);

        size_t nOmega = (size_t)w*h;
        for(int c=0; c<nc; c++)
        {
            size_t pt = x + (size_t)w*y + nOmega*c;

            float u0  = uIn[pt];
            float upx = (x+1<w?  uIn[x+1 + (size_t)w*(y  ) + nOmega*c] : u0);
            float umx = (x-1>=0? uIn[x-1 + (size_t)w*(y  ) + nOmega*c] : u0);
            float upy = (y+1<h?  uIn[x   + (size_t)w*(y+1) + nOmega*c] : u0);
            float umy = (y-1>=0? uIn[x   + (size_t)w*(y-1) + nOmega*c] : u0);
            
            float val = (d0x*upx + dmx*umx + d0y*upy + dmy*umy) / (d0x + dmx + d0y + dmy);

            uOut[x + (size_t)w*y + nOmega*c] = val;
        }
    }
}


__global__
void cuda_diffusivity_kernel (float *diffusivity, const float *u, int w, int h, int nc, float epsilon)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    bool isActive = (x<w && y<h);

    if (isActive)
    {
        size_t nOmega = (size_t)w*h;
        float a = 0;
        for(int c=0; c<nc; c++)
        {
            size_t pt = x + (size_t)w*y + nOmega*c;
            float u0  = u[pt];
            float upx = (x+1<w?  u[x+1 + (size_t)w*(y  ) + nOmega*c] : u0);
            float upy = (y+1<h?  u[x   + (size_t)w*(y+1) + nOmega*c] : u0);
            float dx = upx-u0;
            float dy = upy-u0;
            a += dx*dx + dy*dy;
        }
        a = sqrtf(a);
        a = func_diffusivity(a,epsilon);
        diffusivity[x + (size_t)w*y] = a;
    }
}


__device__
bool isEqual(float3 a, float3 b)
{
    return a.x==b.x && a.y==b.y && a.z==b.z;
}


__global__
void cuda_getMask_kernel (bool *mask_toInpaint, float *img, int w, int h, int nc, float3 colorInpaint)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    bool isActive = (x<w && y<h);

    if (isActive)
    {
        size_t nOmega = (size_t)w*h;
        float3 color;
        color.x = img[x + (size_t)w*y + nOmega*0];
        color.y = img[x + (size_t)w*y + nOmega*1];
        color.z = img[x + (size_t)w*y + nOmega*2];
        bool toInpaint = isEqual(color,colorInpaint);
        mask_toInpaint[x + (size_t)w*y] = toInpaint;
        if (toInpaint)
        {
            color = make_float3(0.5f,0.5f,0.5f);
            img[x + (size_t)w*y + nOmega*0] = color.x;
            img[x + (size_t)w*y + nOmega*1] = color.y;
            img[x + (size_t)w*y + nOmega*2] = color.z;
        }
    }
}


int main(int argc,char **argv)
{
    // ### Reading command line parameters:
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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);

    // ### Define your own parameters here as needed    
    int iter = 100;
    getParam("iter", iter, argc, argv);
    cout << "iter: " << iter << endl;

    float epsilon = 0.01;
    getParam("epsilon", epsilon, argc, argv);
    cout << "epsilon: " << epsilon << endl;

    float lambda = 1;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda: " << lambda << endl;

    float theta = 0.9;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

    float noise = 0.1;
    getParam("noise", noise, argc, argv);
    cout << "noise: " << theta << endl;
    


    // ### Init camera / Load input image
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
    cout << "Image: " << w << " x " << h << endl;

    // ### Set the output image format
    // Let mOut have the same number of channels as the input image (e.g. for the "invert image" or the "convolution" exercise)
    // To let mOut be a color image with 3 channels: CV_32FC3 instead of mIn.type() (e.g. for "visualization of the laplacian" exercise)
    // To let mOut be a grayscale image: use CV_32FC1 instead of mIn.type() (e.g. for the "visualization of the gradient absolute value" exercise)
    // ###
    // ###
    // ### TODO: Change the output image format as needed by the exercise (CV_32FC1 for grayscale, CV_32FC3 for color, mIn.type() for same as input)
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // color, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // grayscale, 1 layer
    //
    // If you want to display other images, define them here as needed, e.g. the opencv image for the convolution kernel


    // ### Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above depending on the exercise (1 for grayscale, 3 for color, nc for general)
    //
    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frames every 30 milliseconds:
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

    // ### Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "opencv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    


    // ###
    // ### Notes:
    // ### 1. Input CPU image imgIn has nc channels. Do not assume nc=3, write the computation for a general nc
    // ### 2. Output CPU image imgOut has 1, 3, or nc channels, depending on how you defined it above. Use may assume 3 channels only if you have used CV_F32FC3.
    // ### 3. Images are layered: access imgIn(x,y,channel c) as imgIn[x + (size_t)w*y + nOmega*c],  where: size_t nOmega = (size_t)w*h;
    // ###
    // ### 4. Allocate arrays as necessary and remember to free them
    // ### 5. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###
    // ### 6. Use the Timer class to measure the run time:
    // ###    Timer timer; timer.start();
    // ###    ...
    // ###    timer.end();  float t = timer.get();  // elapsed time in seconds
    // ###    cout << "time: " << t*1000 << " ms" << endl;
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###

    {
        size_t nOmega = (size_t)w*h;
        size_t n = nOmega*nc;
        dim3 block = dim3(32, 8, 1);
        dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);

        // fixed input image
        float *d_imgData;
        cudaMalloc (&d_imgData,  n*sizeof(float));  CUDA_CHECK;
        cudaMemcpy (d_imgData, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;

        // used to compute the iterations
        float *d_imgIn;
        cudaMalloc (&d_imgIn,  n*sizeof(float));  CUDA_CHECK;
        cudaMemcpy (d_imgIn, d_imgData, n*sizeof(float), cudaMemcpyDeviceToDevice);  CUDA_CHECK;

        bool *d_mask_toInpaint;
        cudaMalloc (&d_mask_toInpaint, nOmega*sizeof(bool));  CUDA_CHECK;
        float3 colorInpaint = make_float3(0,1,0);
        cuda_getMask_kernel <<<grid,block>>> (d_mask_toInpaint, d_imgIn, w, h, nc, colorInpaint);  CUDA_CHECK;


        float *d_imgOut;
        cudaMalloc (&d_imgOut,  n*sizeof(float));  CUDA_CHECK;

        float *d_diffusivity;
        cudaMalloc( &d_diffusivity, nOmega*sizeof(float));  CUDA_CHECK;


        Timer timer;
        timer.start();
        float *a_in = d_imgIn;
        float *a_out = d_imgOut;
        for(int i=0; i<iter; i++)
        {
            cuda_diffusivity_kernel <<<grid,block>>> (d_diffusivity, a_in, w, h, nc, epsilon);  CUDA_CHECK;
            
            // if jacobi
            //cuda_minimizeEnergy_jacobi_step <<<grid,block>>> (a_out, a_in, d_diffusivity, w, h, nc, lambda);  CUDA_CHECK;
            //swap(a_in, a_out);  // the output is always in "a_in" after this
            
            // if SOR
            cuda_minimizeEnergy_sor_step <<<grid,block>>> (a_in, a_in, d_diffusivity, d_mask_toInpaint, w, h, nc, lambda, theta, 0);  CUDA_CHECK;
            cuda_minimizeEnergy_sor_step <<<grid,block>>> (a_in, a_in, d_diffusivity, d_mask_toInpaint, w, h, nc, lambda, theta, 1);  CUDA_CHECK;
        }
        timer.end();
        float t = timer.get();


        cudaMemcpy(imgOut, a_in, n*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;
        cudaFree(d_imgIn);  CUDA_CHECK;
        cudaFree(d_imgOut);  CUDA_CHECK;
        cudaFree(d_diffusivity);  CUDA_CHECK;
        cudaFree(d_mask_toInpaint);  CUDA_CHECK;
        
        cout << "time: " << t*1000 << " ms" << endl;
    }




    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // proceed similarly for other output images, e.g. the convolution kernel:

#ifdef CAMERA
    // End of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif



    // ### Free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);
    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



