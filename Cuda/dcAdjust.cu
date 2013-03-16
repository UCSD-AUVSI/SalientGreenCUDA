
#include <Cuda/dcAdjust.H>

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>


static inline int divUp( int total, int grain )
{
	return ( total + grain - 1 ) / grain;
}


// this kernel computes, per-block, the min/max
// of a block-sized portion of the input
// using a block-wide reduction
__global__ void blockMinMax( cv::gpu::DevMem2D_<myfloat2> input,
                          myfloat2 *perBlockResults,
                          const size_t numBlocks,
						  bool isFirstPass)
{
  extern __shared__ myfloat2 sdata[];

  //unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  const int threadIdxIdx = threadIdx.y * blockDim.x + threadIdx.x;
  const int blockDimOverall = blockDim.x * blockDim.y;

  const int i = y*blockDim.x*gridDim.x + x;//threadIdx.y * blockDim.x + threadIdx.x; // thread number within everything

  // load input into __shared__ memory
  myfloat2 val;
  val.x = 101010000000000000.0f; // min
  val.y = -101010000000000000.0f; // max
  if (isFirstPass)
  {
	  if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
	  {
		val.x = input.ptr( y )[x].x; // store real component into min
		val.y = input.ptr( y )[x].x; // store real component into max
	  }
  }
  else
	  if (i >= 0 && i < numBlocks)
		  val = perBlockResults[i]; // get min/max of block from last launch
  
  sdata[threadIdxIdx] = val;
  __syncthreads();

  // contiguous range pattern
  for(int offset = blockDimOverall / 2;
      offset > 0;
      offset >>= 1)
  {
	  if(threadIdxIdx < offset)
	  {
		  // compute a partial min/max upstream to our own
		  sdata[threadIdxIdx].x = min(sdata[threadIdxIdx].x,  sdata[threadIdxIdx + offset].x);
		  sdata[threadIdxIdx].y = max(sdata[threadIdxIdx].y,  sdata[threadIdxIdx + offset].y);
	  }
	  
	  // wait until all threads in the block have
	  // updated their partial sums
	  __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdxIdx == 0 )
  {
	  perBlockResults[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0]; // block number within grid
  }
}

// calc min/max of real component
void minMaxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const mat, double & theMin, double & theMax )
{
	// http://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/sum_reduction.cu
	const int numThreads = 16;
	dim3 threads( numThreads, numThreads, 1 );
	dim3 grids( divUp( mat.cols, threads.x ), divUp( mat.rows, threads.y ), 1 );

	size_t numElements = mat.rows*mat.cols;
	size_t blockSize = threads.x * threads.y;
	size_t numBlocks = grids.x * grids.y;

	myfloat2 *dPartialMinMax = 0;
	cudaMalloc((void**)&dPartialMinMax, sizeof(myfloat2) * (numBlocks + 1));

	// launch one kernel to compute, per-block, a partial sum
	blockMinMax<<<grids,threads,blockSize * sizeof(myfloat2)>>>(mat, dPartialMinMax, numBlocks, true);

	for (int i = max(mat.rows, mat.cols)/numThreads; i > 0; i = i / numThreads)
	{
		numBlocks = grids.x * grids.y;
		// launch a kernel to compute intermediate min/max of the partial min/maxs
		// dPartialMinMax shrinks by a factor of 16 in each dimension
		// this loop exits once dPartialMax has shrunk to one element
		grids.x = divUp(grids.x, threads.x);
		grids.y = divUp(grids.y, threads.y);
		blockMinMax<<<grids, threads, blockSize * sizeof(myfloat2)>>>(mat, dPartialMinMax, numBlocks, false);
	}

	// get result off device memory
	myfloat2 result;
	cudaMemcpy(&result, dPartialMinMax, sizeof(myfloat2), cudaMemcpyDeviceToHost);
	theMin = result.x;
	theMax = result.y;

	// free memory
	cudaFree(dPartialMinMax);
}

// WARNING: you must multiply l by 116.0f. I don't do this here because it causes the weirdest 
//			fucking bug ever. The other three channels get changed if I make l too big.
__global__ void labgPadFFTSplit( 
	cv::gpu::DevMem2D_<myfloat3> const img, 
	cv::gpu::DevMem2D_<myfloat2> l, 
	cv::gpu::DevMem2D_<myfloat2> a, 
	cv::gpu::DevMem2D_<myfloat2> b, 
	cv::gpu::DevMem2D_<myfloat2> g,
	int paddedWidth,
	int paddedHeight,
	int kernelWidth,
	int kernelHeight )
{
	// x,y are for destination matrices
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= 0 && x < paddedWidth && y >= 0 && y < paddedHeight)
	{
		// calc input pixel
		int inX, inY;
		if (x < img.cols)
			inX = x;
		else if (paddedWidth - x < kernelWidth)
		{
			inX = paddedWidth - x;
		}
		else
		{
			inX = img.cols - 1;
		}

		if (y < img.rows)
			inY = y;
		else if (paddedHeight - y < kernelHeight)
		{
			inY = paddedHeight - y;
		}
		else
		{
			inY = img.rows - 1;
		}

		// get r,g,b value from img
		float redInput,greenInput,blueInput;
		myfloat3 px = img.ptr( inY )[inX];
		blueInput = px.x;
		greenInput = px.y;
		redInput = px.z;

		float lPx=0.0f,aPx=0.0f,bPx=0.0f,gPx=0.0f;

		// calc grayscale value
		gPx = (redInput+greenInput+blueInput)/3.0f;

		// calc lab
		float xColor, yColor, zColor;

		if (redInput > 0.04045f) {
			redInput = pow(( ( redInput + 0.055f ) / 1.055f), 2.4f);
		} else	{
			redInput = redInput / 12.92f;
		}
		if (greenInput > 0.04045f){
			greenInput = pow(( ( greenInput + 0.055f ) / 1.055f), 2.4f);
		} else	{
			greenInput = greenInput / 12.92f;
		}
		if (blueInput > 0.04045f) {
			blueInput = pow(( ( blueInput + 0.055f ) / 1.055f), 2.4f);
		} else	{
			blueInput = blueInput / 12.92f;	
		}

		redInput = redInput * 100.0f;
		greenInput = greenInput * 100.0f;
		blueInput = blueInput * 100.0f;	
		
		
		xColor = (redInput * 0.4124f) + (greenInput * 0.3576f) + (blueInput * 0.1805f);
		yColor = (redInput * 0.2126f) + (greenInput * 0.7152f) + (blueInput * 0.0722f);
		zColor = (redInput * 0.0193f) + (greenInput * 0.1192f) + (blueInput * 0.9505f);		
				
		xColor = xColor / 95.047f;
		yColor = yColor / 100.000f;
		zColor = zColor / 108.883f;

		// we have xyz, now to convert to lab
		if (xColor > 0.008856f) {
			xColor = pow(xColor, 1.0f/3.0f); //1/3 == 1?? but 0.3333 is okay 
		} else		{	
			xColor = (7.787f * xColor) + (16.0f/116.0f); //0.137931034482759 = 16/116
		}
		if (yColor > 0.008856f){
			yColor = pow(yColor, 1.0f/3.0f);
		} else	{	
			yColor = (7.787f * yColor) + (16.0f/116.0f);
		}
		if (zColor > 0.008856f){ 
			zColor = pow(zColor, 1.0f/3.0f);
		} else	{	
			zColor = (7.787f * zColor) + (16.0f/116.0f); 
		}
		
		// calculate and save lab values into the float4
		lPx = yColor - 16.0f / 116.0f ;//lPx = (116.0f * yColor) - 16.0f; // multiply causes problems, do it later
		aPx = 500.0f * (xColor - yColor);
		bPx = 200.0f * (yColor - zColor);

		// save lab,g into labg[x,y].x
		l.ptr( y )[x].x = lPx;
		a.ptr( y )[x].x = aPx;
		b.ptr( y )[x].x = bPx;
		g.ptr( y )[x].x = gPx;

		// set lab,g[x,y].y to 0
		l.ptr( y )[x].y = 0.0f;
		a.ptr( y )[x].y = 0.0f;
		b.ptr( y )[x].y = 0.0f;
		g.ptr( y )[x].y = 0.0f;
		
	}
}

__global__ void unpad( 
	cv::gpu::DevMem2D_<myfloat2> const input, 
	cv::gpu::DevMem2D_<float> output)
{
	// x,y are for destination matrices
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= 0 && x < output.cols && y >= 0 && y < output.rows) // output is smaller
	{
		output.ptr( y )[x] = input.ptr( y )[x].x;
	}
}

__global__ void minMaxHelper( cv::gpu::DevMem2D_<float> const mat, float thresh, float * sum, int * count )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x > 0 && x < mat.cols - 1 && y > 0 && y < mat.rows - 1 )
	{
		const float value = mat.ptr( y )[x];
		const float neighborLeft = mat.ptr( y )[x - 1];
		const float neighborUp = mat.ptr( y - 1 )[x];
		const float neighborDown = mat.ptr( y + 1 )[x];
		const float neighborRight = mat.ptr( y )[x + 1];

		if( value >= thresh &&
				value > neighborLeft &&
				value > neighborUp &&
				value > neighborDown &&
				value >= neighborRight )
		{
			*sum += value;
			*count = *count + 1;
		}
	}
}
__global__ void absFC2( cv::gpu::DevMem2D_<myfloat2> mat )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x = fabsf( mat.ptr( y )[x].x );
		mat.ptr( y )[x].y = fabsf( mat.ptr( y )[x].y );
	}
}

__global__ void absAndZeroFC2( cv::gpu::DevMem2D_<myfloat2> mat )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x = fabsf( mat.ptr( y )[x].x );
		mat.ptr( y )[x].y = 0.0f;
	}
}

__global__ void maxFC2( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x = fmaxf( mat.ptr( y )[x].x, value );
		mat.ptr( y )[x].y = fmaxf( mat.ptr( y )[x].y, value );
	}
}

//! Adds to the DC offset for a matrix
/*! @param[in] mat The CUDA device memory for the matrix
	  @param[in] value The value to add to the DC component */
__global__ void dcAdjust( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	mat.ptr( 0 )[0].x += value;
}

//! computes c = a + b
__global__ void addFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x + b.ptr( y )[x].x;
		c.ptr( y )[x].y = a.ptr( y )[x].y + b.ptr( y )[x].y;
	}
}

__global__ void addAndZeroFC2( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < mat.cols && y < mat.rows )
	{
		mat.ptr( y )[x].x += value; fmaxf( mat.ptr( y )[x].x, value );
		mat.ptr( y )[x].y = 0.0f;
	}
}

__global__ void addRealFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x + b.ptr( y )[x].x;
		c.ptr( y )[x].y = 0.0f;
	}
}

//! computes c = a - b
__global__ void subFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x - b.ptr( y )[x].x;
		c.ptr( y )[x].y = a.ptr( y )[x].y - b.ptr( y )[x].y;
	}

}
//! computes c = a * b
__global__ void mulFC2( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x * b.ptr( y )[x].x;
		c.ptr( y )[x].y = a.ptr( y )[x].y * b.ptr( y )[x].y;
	}

}

//! computes c = a * b
__global__ void mulValueFC2( cv::gpu::DevMem2D_<myfloat2> const a, float const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( x < a.cols && y < a.rows )
	{
		c.ptr( y )[x].x = a.ptr( y )[x].x * b;
		c.ptr( y )[x].y = a.ptr( y )[x].y * b;
	}

}


void labgPadFFTSplitWrapper( 
	cv::gpu::DevMem2D_<myfloat3> const img, 
	cv::gpu::DevMem2D_<myfloat2> l, 
	cv::gpu::DevMem2D_<myfloat2> a, 
	cv::gpu::DevMem2D_<myfloat2> b, 
	cv::gpu::DevMem2D_<myfloat2> g,
	int paddedWidth,
	int paddedHeight,
	int kernelWidth,
	int kernelHeight )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( paddedWidth, threads.x );
	grids.y = divUp( paddedHeight, threads.y );

	labgPadFFTSplit<<< grids, threads >>>( img, l, a, b, g, paddedWidth, paddedHeight, kernelWidth, kernelHeight );

	// next line is because we don't multiply l by 116.0f in the kernel. see code for details
	mulValueFC2Wrapper(l, 116.0f, l);
}

void unpadWrapper( 
	cv::gpu::DevMem2D_<myfloat2> const input, 
	cv::gpu::DevMem2D_<float> output)
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( input.cols, threads.x );
	grids.y = divUp( input.rows, threads.y );

	unpad<<< grids, threads >>>( input, output );
}

void addRealFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	addRealFC2<<< grids, threads >>>( a, b, c );
}

void addFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	addFC2<<< grids, threads >>>( a, b, c );
}

void addAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	addAndZeroFC2<<< grids, threads >>>( mat, value );
}
void subFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	subFC2<<< grids, threads >>>( a, b, c );
}

void mulFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, cv::gpu::DevMem2D_<myfloat2> const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	mulFC2<<< grids, threads >>>( a, b, c );
}

void mulValueFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> const a, float const b,
		cv::gpu::DevMem2D_<myfloat2> c )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( a.cols, threads.x );
	grids.y = divUp( a.rows, threads.y );

	mulValueFC2<<< grids, threads >>>( a, b, c );
}

void dcAdjustWrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	dcAdjust<<< grids, threads >>>( mat, value );
}


void absFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	absFC2<<< grids, threads >>>( mat );
}

void absAndZeroFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	absAndZeroFC2<<< grids, threads >>>( mat );
}

void maxFC2Wrapper( cv::gpu::DevMem2D_<myfloat2> mat, float value )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	maxFC2<<< grids, threads >>>( mat, value );
}


void minMaxHelperWrapper( cv::gpu::DevMem2D_<float> const mat, float thresh, float & sum, int & count )
{
	dim3 threads( 16, 16, 1 );
	dim3 grids( 1, 1, 1 );

	grids.x = divUp( mat.cols, threads.x );
	grids.y = divUp( mat.rows, threads.y );

	float * dSum;
	int * dCount;
	cudaMalloc(&dSum, sizeof(float));
	cudaMalloc(&dCount, sizeof(int));

	minMaxHelper<<< grids, threads >>>( mat, thresh, dSum, dCount );
	cudaMemcpy( &sum, dSum, sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( &count, dCount, sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree( dCount );
	cudaFree( dSum );
}
