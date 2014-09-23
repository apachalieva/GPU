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
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###


#include "aux.h"
#include <cstdlib>
#include <iostream>
using std::stringstream;
using std::cerr;
using std::cout;
using std::endl;
using std::string;




// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(std::string param, bool &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc) || argv[i+1][0]=='-') { var = true; return true; }
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}




// opencv helpers
void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn)
{
    convert_layered_to_interleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}


void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn)
{
    convert_interleaved_to_layered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}



void showImage(string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}




// adding Gaussian noise
float noise(float sigma)
{
    float x1 = (float)rand()/RAND_MAX;
    float x2 = (float)rand()/RAND_MAX;
    return sigma * sqrtf(-2*log(std::max(x1,0.000001f)))*cosf(2*M_PI*x2);
}
void addNoise(cv::Mat &m, float sigma)
{
    float *data = (float*)m.data;
    int w = m.cols;
    int h = m.rows;
    int nc = m.channels();
    size_t n = (size_t)w*h*nc;
    for(size_t i=0; i<n; i++)
    {
        data[i] += noise(sigma);
    }
}




// cuda error checking
string prev_file = "";
int prev_line = 0;
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}

// Color transformation
// source : http://linuxtv.org/downloads/v4l-dvb-apis/colorspaces.html
int clamp (double x)
{
    int r = x;      /* round to nearest */
	
    if (r < 0)         return 0;
    else if (r > 1)  return 1;
    else               return r;
}

void forward_color_transf( float *imgRGB, float *imgChrom, int w, int h, int nc )
{
    //int ER, EG, EB;         /* gamma corrected RGB input [0;255] */
    //int Y1, Cb, Cr;         /* output [0;255] */

    double r, g, b;         /* temporaries */
    double y1, pb, pr;
    for( int i = 0; i < w; i++ ){
      for( int j = 0; j < h; j++ ){
	//for( int channel; channel < nc; channel++ ){
	  r = imgRGB[ i + j*w + w*h*0 ];	// 	  r = ER / 255.0;
	  g = imgRGB[ i + j*w + w*h*1 ]; 	// 	  g = EG / 255.0;
	  b = imgRGB[ i + j*w + w*h*2 ];	// 	  b = EB / 255.0;

	  y1  =  0.299  * r + 0.587 * g + 0.114  * b;
	  pb  = -0.169  * r - 0.331 * g + 0.5    * b;
	  pr  =  0.5    * r - 0.419 * g - 0.081  * b;

	  imgChrom[i + j*w + w*h*0] = clamp ( ( 219 / 255.0 ) * y1 + ( 16 / 255.0 ));		// 	  Y1 = clamp (219 * y1 + 16);
	  imgChrom[i + j*w + w*h*1] = clamp ( ( 224 / 255.0 ) * pb + ( 128 / 255.0 ));		// 	  Cb = clamp (224 * pb + 128);
	  imgChrom[i + j*w + w*h*2] = clamp ( ( 224 / 255.0 ) * pr + ( 128 / 255.0 ));	// 	  Cr = clamp (224 * pr + 128);

	  /* or shorter */

	  // y1 = 0.299 * ER + 0.587 * EG + 0.114 * EB;
	  // 
	  // Y1 = clamp ( (219 / 255.0)                    *       y1  + 16);
	  // Cb = clamp (((224 / 255.0) / (2 - 2 * 0.114)) * (EB - y1) + 128);
	  // Cr = clamp (((224 / 255.0) / (2 - 2 * 0.299)) * (ER - y1) + 128);
	//}
      }
    }
}     

//Inverse Transformation
void inverse_color_transf( float *imgChrom, float *imgRGB, int w, int h, int nc )
{
    //int Y1, Cb, Cr;         /* gamma pre-corrected input [0;255] */
    //int ER, EG, EB;         /* output [0;255] */

    double r, g, b;         /* temporaries */
    double y1, pb, pr;
    for( int i = 0; i < w; i++ ){
      for( int j = 0; j < h; j++ ){
	y1 = ( imgChrom[i + j*w + w*h*0] - ( 16 / 255.0 ))  / ( 219 / 255.0 );	//     y1 = (Y1 - 16)  / 219.0;
	pb = ( imgChrom[i + j*w + w*h*0] - ( 128 / 255.0 )) / ( 224 / 255.0 );	//     pb = (Cb - 128) / 224.0;
	pr = ( imgChrom[i + j*w + w*h*0] - ( 128 / 255.0 )) / ( 224 / 255.0 );	//     pr = (Cr - 128) / 224.0;

	r = 1.0 * y1 + 0     * pb + 1.402 * pr;
	g = 1.0 * y1 - 0.344 * pb - 0.714 * pr;
	b = 1.0 * y1 + 1.772 * pb + 0     * pr;

	imgRGB[ i + j*w + w*h*0 ] = clamp (r);	//     ER = clamp (r * 255); /* [ok? one should prob. limit y1,pb,pr] */
	imgRGB[ i + j*w + w*h*1 ] = clamp (g);	//     EG = clamp (g * 255);
	imgRGB[ i + j*w + w*h*2 ] = clamp (b);	//     EB = clamp (b * 255);
      }
    }
}