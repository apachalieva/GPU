// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2013/2014, March 3 - April 4
// ###
// ###
// ### Evgeny Strekalovskiy, Maria Klodt, Jan Stuehmer, Mohamed Souiai
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###


#ifndef AUX_H
#define AUX_H

#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <sstream>



// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc)) continue;
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}





// opencv helpers
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn);
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn);
void normalize_for_display(float *aOut, const float *aIn, int n);
void showImage(std::string title, const cv::Mat &mat, int x, int y);
void addNoise(cv::Mat &m, float sigma);



// timer
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{

	}
	void start()
	{
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};





// cuda error checking
void cuda_check(std::string file, size_t line);
#define CUDA_CHECK do { cuda_check(__FILE__,__LINE__); } while (false)


// cuda get value for pitched array
template<typename T>
T& pitch_value(T *a, int x, int y, size_t pitch)
{
    return *((T*)((char*)a + x*sizeof(T) + pitch*y));
}



#endif  // AUX_H
