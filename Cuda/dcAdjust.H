
#include <opencv2/gpu/devmem2d.hpp>

struct myfloat2
{
	float x;
	float y;
};

struct myfloat3
{
	float x;
	float y;
	float z;
};

#ifdef EXPORT_SALIENTGREEN
#define CLASSEXPORT __declspec(dllexport)
#else
#define CLASSEXPORT __declspec(dllimport)
#endif

CLASSEXPORT void labgPadFFTSplitWrapper( 
	cv::gpu::DevMem2D_<myfloat3> const img, 
	cv::gpu::DevMem2D_<myfloat2> l, 
	cv::gpu::DevMem2D_<myfloat2> a, 
	cv::gpu::DevMem2D_<myfloat2> b, 
	cv::gpu::DevMem2D_<myfloat2> g,
	int paddedWidth,
	int paddedHeight,
	int kernelWidth,
	int kernelHeight);

CLASSEXPORT void unpadWrapper( 
	cv::gpu::DevMem2D_<myfloat2> const input, 
	cv::gpu::DevMem2D_<float> output);

CLASSEXPORT void addFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

CLASSEXPORT void addRealFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

CLASSEXPORT void addAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );

CLASSEXPORT void subFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

CLASSEXPORT void mulFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c );

CLASSEXPORT void mulValueFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, float value,
		cv::gpu::DevMem2D_<myfloat2> c );

//! Adjusts the dc component of some matrix
/*! @param[in] mat Device memory for the matrix (float)
 		@param[in] height height of the matrix
		@param[in] width width of the matrix
		@param[in] value the value to add */
CLASSEXPORT void dcAdjustWrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );


CLASSEXPORT void absFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat );
CLASSEXPORT void absAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat );

CLASSEXPORT void maxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value );

CLASSEXPORT void minMaxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const mat, double & theMin, double & theMax );

CLASSEXPORT void minMaxHelperWrapper( cv::gpu::DevMem2D_<float> const mat, float thresh, float & sum, int & count );
