/*
 * Salient Green
 * Copyright (C) 2011 Shane Grant
 * wgrant@usc.edu
 *
 * Salient Green is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3.0
 * of the License, or (at your option) any later version.
 *
 * Salient Green is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Salient Green; if not, see
 * <http://www.gnu.org/licenses/> */

#include <SalientGreenGPU.H>
#include <Filters/LowPass.H>
#include <Filters/Utility.H>
#include <Cuda/dcAdjust.H>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <ctime>

#if defined(__WIN32__)
#include <windows.h>
#endif

void sg::SalientGreenGPU::Results::show()
{
  cv::Mat temp;
  cv::resize( lResponse, temp, labSaliency.size() );
  cv::imshow( "lResponse", temp / 255.0f );
  cv::resize( aResponse, temp, labSaliency.size() );
  cv::imshow( "aResponse", temp / 255.0f );
  cv::resize( bResponse, temp, labSaliency.size() );
  cv::imshow( "bResponse", temp / 255.0f );
  cv::resize( oResponse, temp, labSaliency.size() );
  cv::imshow( "oResponse", temp / 255.0f );

  cv::imshow( "labSaliency", labSaliency / 255.0f );
}

void sg::SalientGreenGPU::Results::save(
  std::string const & prefix, std::string const & filetype )
{
  cv::Mat temp;
  cv::resize( lResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "lResponse." + filetype, temp );
  cv::resize( aResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "aResponse." + filetype, temp );
  cv::resize( bResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "bResponse." + filetype, temp );
  cv::resize( oResponse, temp, labSaliency.size() );
  cv::imwrite( prefix + "oResponse." + filetype, temp );

  cv::imwrite( prefix + "saliencyLAB." + filetype, labSaliency );
}

sg::SalientGreenGPU::SalientGreenGPU() //: itsInputImage()
{
	allocateGPUBuffers();
}

void sg::SalientGreenGPU::release()
{
  itsImgGpu.release();
  for (int i=0; i < 4; ++i)
    itsLabgGpu[i].release();
  itsLabgGpu.clear();
  itsBufferAGpu.release();
  itsBufferBGpu.release();
  itsAccumulatorGpu.release();
  itsSaliencyGpu.release();
  
  itsDoGBank.release();
  //itsLogGabor.release();
  itsNormalizeIterSpatial.release();
  /*for (int i = 0; i < itsLabResponse.size(); i++)
		itsLabResponse.at(i).release();

	itsOResponse.release();

	for (int i = 0; i < itsLabFFT.size(); i++)
		itsLabFFT.at(i).release();

	
	itsGrayscaleFFT.release();
  itsOBuffer.release();

	for (int i = 0; i < itsSplitBufferFrequency.size(); i++)
		itsSplitBufferFrequency.at(i).release();

	for (int i = 0; i < itsSplitBufferSpatial.size(); i++)
		itsSplitBufferSpatial.at(i).release();

  itsSaliencyLab.release();
  itsGpuBuffer.release();

  itsGpuBufferSpatial.release();
  

	for (int i = 0; i < itsGpuBufferFrequency.size(); i++)
		itsGpuBufferFrequency.at(i).release();
  
	itsDoGBank.release();
  itsDoGBankFrequency.release();
  itsLogGabor.release();
  itsNormalizeIterSpatial.release();
  itsNormalizeIterFrequency.release();*/
}

int millis()
{
	return timeGetTime();
}

time_t startTime, lastTime;
void sg::SalientGreenGPU::startTimer()
{
	startTime = lastTime = millis();
}

time_t sg::SalientGreenGPU::tick()
{
	auto current = millis();
	auto elapsed = current - lastTime;
	lastTime = current;
	return elapsed;
}

void sg::SalientGreenGPU::runTests(cv::Mat const & inputImage)
{
	cv::Mat grayscaleImage, compare,image;
  // convert to float, leave 0 to 255 range
  if( inputImage.type() != CV_32FC3 )
    inputImage.convertTo( image, CV_32FC3 );
  else
	image = inputImage;


  auto prevSize = itsImageSize;
  itsImageSize = image.size();
  cvtColor(image,grayscaleImage,CV_BGR2GRAY);
  imwrite("C:\\test\\gray.jpg",grayscaleImage);

	release();
	prepareFiltersGPU();
	allocateGPUBuffers( );

  ////// STEP 1 - Upload to GPU
  itsImgGpu.upload(image);


  labgPadFFTSplit(itsImgGpu, itsLabgGpu);
  itsAccumulatorGpu.setTo(0.0f);
  try{
	  cv::gpu::dft( itsLabgGpu[3], itsLabgGpu[3], itsLabgGpu[3].size());
	  // doDoGGPU(itsLabgGpu[3],itsBufferAGpu,itsLabgGpu[3]);

	  doDoGGPU(itsLabgGpu[3], itsBufferAGpu, itsAccumulatorGpu);
	  // cv::gpu::dft( itsLabgGpu[3], itsLabgGpu[3], itsLabgGpu[3].size(), cv::DFT_INVERSE);
  } catch (cv::Exception& e ){
	  const char* err_msg = e.what();
	  printf(err_msg);
  }
  // absAndZeroFC2Wrapper( itsLabgGpu[3] );
  // normalizeGPU( itsLabgGpu[3], 255.0f );
  unpadFFT(itsAccumulatorGpu, itsSaliencyGpu);

  Results res( itsImageSize ); 
  //cv::Mat dest(itsImageSize, CV_32FC1);
  try {
	  itsSaliencyGpu.download(res.labSaliency);
  } catch( cv::Exception& e ) { const char* err_msg = e.what(); throw e; }

  compare = grayscaleImage - res.labSaliency;
  cv::imwrite("c:\\test\\difference.jpg",compare);
  cv::imwrite("c:\\test\\otherimage.jpg",res.labSaliency);
  
  
	
	/*const int imgSize = 7000;
	cv::Mat testMat(cv::Size(imgSize,imgSize), CV_32FC2);
	testMat.setTo(0.0f);

	testMat.at<cv::Vec2f>(imgSize-1,0)[0] = 1.0f;
	testMat.at<cv::Vec2f>(imgSize-1,imgSize-1)[0] = -1.0f;

	cv::gpu::GpuMat testGpuMat(cv::Size(imgSize,imgSize), CV_32FC2);
	testGpuMat.setTo(0.0f);
	testGpuMat.upload(testMat);

	double mi, ma;
	minMaxFC2Wrapper(testGpuMat, mi, ma);

	
	cv::Mat confirmMat(cv::Size(imgSize,imgSize), CV_32FC2);
	confirmMat.setTo(0.0f);
	testGpuMat.download(confirmMat);

	printf("DEBUG size(%d,%d) confirm(%.1f, %.1f) min/max(%.1f, %.1f)\n", 
		testGpuMat.size().width, testGpuMat.size().height, 
		confirmMat.at<cv::Vec2f>(0,0)[0], confirmMat.at<cv::Vec2f>(0,1)[0],
		mi, ma);*/

}

sg::SalientGreenGPU::Results sg::SalientGreenGPU::computeSaliencyGPU( cv::Mat const & inputImage,
    labWeights const * lw )
{
	// NOTE: don't do multiple dft's in parallel. Do them synchronously, separately.
	// otherwise we get weird bugs.

	// runTests(inputImage);
	//startTimer(); std::cout<< "starting compute saliency" << std::endl;
 //std::cout << "Image Size: " << inputImage.size().width << "," << inputImage.size().height << std::endl;
 	
  // NOTES: if lw && sw == NULL, this won't work

  cv::Mat image;

  // convert to float, leave 0 to 255 range
  if( inputImage.type() != CV_32FC3 )
    inputImage.convertTo( image, CV_32FC3 );
  else
	image = inputImage;

  auto prevSize = itsImageSize;
  itsImageSize = image.size();

  
  // generate new filters if image size changed
  if( prevSize.width != itsImageSize.width || prevSize.height != itsImageSize.height )
  {
		release();
	//try { // LEWIS
		prepareFiltersGPU();
		allocateGPUBuffers( );
	//} catch( cv::Exception& e ) { const char* err_msg = e.what(); throw e; }
  }

  ////// STEP 1 - Upload to GPU
  itsImgGpu.upload(image);

  ////// STEP 2 - LABG conversion, padFFT, split into channels
  labgPadFFTSplit(itsImgGpu, itsLabgGpu);

  ////// STEP 3 - DFT on each channel
  for (int i = 0; i < 4; ++i)
    cv::gpu::dft( itsLabgGpu[i], itsLabgGpu[i], itsLabgGpu[i].size());

  ////// STEP 4 - Gabor transform
  itsAccumulatorGpu.setTo(0.0f); 
  // TODO: replace previous line with gabor

  ////// STEP 5 - DoG on l,a,b
  doDoGGPU(itsLabgGpu[0], itsBufferAGpu, itsAccumulatorGpu);
  doDoGGPU(itsLabgGpu[1], itsBufferAGpu, itsAccumulatorGpu);
  doDoGGPU(itsLabgGpu[2], itsBufferAGpu, itsAccumulatorGpu);

  ////// STEP 6 - normalize accumulator
  normalizeGPU( itsAccumulatorGpu, 255.0f );

  ////// STEP 7 - unpad and copy into saliency
  unpadFFT(itsAccumulatorGpu, itsSaliencyGpu);

  ////// STEP 8 - download saliency to cpu
  Results res( itsImageSize ); 

  try {
	  itsSaliencyGpu.download(res.labSaliency);
  } catch( cv::Exception& e ) { const char* err_msg = e.what(); throw e; }

  return res;
  /*std::cout <<  tick() << " - alloc results\n";
  // Compute saliency
  Results res( itsInputImage.size() );
  cv::Mat saliency;

  cv::Mat grayscale;
  cv::Mat grayscaleFFT;

  std::vector<cv::Mat> lab( 3 );
  std::vector<cv::Mat> labFFT( 3 );

  // get grayscale
  cv::cvtColor( itsInputImage, grayscale, CV_BGR2GRAY ); // LEWIS: removed
  
  std::cout << tick() << " - do lab conversion\n";

  
  cv::gpu::Stream stream;
  stream.enqueueUpload( itsInputImage, itsGpuBuffer3 );
  stream.waitForCompletion();

  // get lab
	{
		std::vector<cv::gpu::GpuMat> saliencyGPUChannels;
		saliencyGPUChannels.push_back(itsLabResponse[0]);
		saliencyGPUChannels.push_back(itsLabResponse[1]);
		saliencyGPUChannels.push_back(itsLabResponse[2]);
		cv::gpu::GpuMat saliencyGPU;
		cv::gpu::cvtColor( itsGpuBuffer3, saliencyGPU, CV_BGR2Lab );
		cv::gpu::split( saliencyGPU, saliencyGPUChannels );
	}

  // Prepare to take FFT
  std::cout <<  tick() << " - Pad images\n";

  grayscaleFFT = padImageFFT( grayscale, itsFFTSizeFrequency ); // LEWIS: removed
 
	// Lab FFTs
  {
    itsLabFFT[0] = padImageFFTGPU( itsLabResponse[0], itsMaxFilterSize, itsFFTSizeSpatial );
    itsLabFFT[1] = padImageFFTGPU( itsLabResponse[1], itsMaxFilterSize, itsFFTSizeSpatial );
    itsLabFFT[2] = padImageFFTGPU( itsLabResponse[2], itsMaxFilterSize, itsFFTSizeSpatial );
  }

  // Upload data to the GPU
  std::cout <<  tick() << " - Uploading images to GPU\n";

 //  cv::gpu::Stream stream;

 //  // stream.enqueueUpload( grayscaleFFT, itsGrayscaleFFT ); // LEWIS: removed
	// for( size_t i = 0; i < 3; ++i )
 //  {
 //    stream.enqueueUpload( labFFT[i], itsLabFFT[i] );
 //  }

 //  stream.waitForCompletion();

  // FFT images
  std::cout <<  tick() << " - Taking image DFTs\n";
  cv::gpu::dft( itsGrayscaleFFT, itsGrayscaleFFT, itsGrayscaleFFT.size() ); // LEWIS: removed
  
	for (int i = 0; i < itsLabFFT.size(); i++)
      cv::gpu::dft( itsLabFFT.at(i), itsLabFFT.at(i), itsLabFFT.at(i).size() );

  //itsGrayscaleFFT = itsLabFFT[0]; // LEWIS: use luminosity for gabor

  // Compute saliency!
  // gabor
  { 
    std::cout <<  tick() << " - Gabor \n";
	//try { // LEWIS DEBUG
		doGaborGPU( itsGrayscaleFFT, itsOBuffer ); // LEWIS: turn this line back on later ... gabor needs to be tested
	//} catch( cv::Exception& e ) { const char* err_msg = e.what(); throw e; }// LEWIS
  }

  // lab color
  {
    std::cout <<  tick() << " - DoG\n";
    std::cout << "L channel...\n";
    doDoGGPU( itsLabFFT[0], itsLabFFT[0] );
    std::cout << "A channel...\n";
    doDoGGPU( itsLabFFT[1], itsLabFFT[1] );
    std::cout << "B channel...\n";
    doDoGGPU( itsLabFFT[2], itsLabFFT[2] );
  }

  std::cout <<  tick() << " - Normalizing\n";
  // gabor
  {
    itsNormalizeIterFrequency( itsOBuffer, itsOResponse, itsSplitBufferFrequency, 1 );
  }

  // lab
	{
    for( size_t i = 0; i < 3; ++i )
      itsNormalizeIterSpatial( itsLabFFT[i], itsLabResponse[i], itsSplitBufferSpatial, 2 );
  }

  // Accumulate maps on GPU
  std::cout <<  tick() << " - Accumulating maps\n";

  // lab
	{
    itsSaliencyLab.setTo( 0.0 );
    cv::gpu::multiply( itsLabResponse[0], lw->l, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

    cv::gpu::multiply( itsLabResponse[1], lw->a, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

    cv::gpu::multiply( itsLabResponse[2], lw->b, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

    cv::gpu::multiply( itsOResponse, lw->o, itsGpuBuffer );
    cv::gpu::add( itsSaliencyLab, itsGpuBuffer, itsSaliencyLab );

		//cv::Mat temp( itsSaliencyLab.size(), itsSaliencyLab.type() );
		// temp = itsSaliencyLab;
    //itsSaliencyLab.download(temp); // LEWIS
		// itsLabFFT[0] = padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial );
    //itsLabFFT[0].upload(padImageFFT( temp, itsMaxFilterSize, itsFFTSizeSpatial )); // LEWIS
    
    itsNormalizeIterSpatial( itsLabFFT[0], itsSaliencyLab, itsSplitBufferSpatial, 0 );
  }

  // Download everything
  //{
  //  stream.enqueueDownload( itsLabResponse[0], res.lResponse );
  //  stream.enqueueDownload( itsLabResponse[1], res.aResponse );
  //  stream.enqueueDownload( itsLabResponse[2], res.bResponse );
  //}
	
  std::cout <<  tick() << " - download images\n";
	stream.enqueueDownload( itsSaliencyLab, res.labSaliency );
  //stream.enqueueDownload( itsOResponse, res.oResponse );
	// LEWIS: you only need to download saliency here when you want this
	// going as fast as possible, the others are just so you can look at them
	// and debug things for now

  stream.waitForCompletion();
  
  std::cout <<  tick() << " - done: " << time(NULL) - startTime << "\n";
  
  return res;*/
}

void sg::SalientGreenGPU::allocateGPUBuffers()
{
  std::cout << "Allocating memory on the GPU.\n";

  // itsImageSize, itsFFTSizeSpatial

  // allocate img_gpu - 3 channel, real, unpadded
  itsImgGpu = cv::gpu::GpuMat( itsImageSize, CV_32FC3);

  // labg - vector of 4 mat, 1 channel, complex, padded
  itsLabgGpu.clear();
  for (int i=0; i < 4; ++i)
    itsLabgGpu.push_back(cv::gpu::GpuMat(itsFFTSizeSpatial, CV_32FC2));

  // bufferA, bufferB - 1 channel, padded, complex
  itsBufferAGpu = cv::gpu::GpuMat(itsFFTSizeSpatial, CV_32FC2);
  //itsBufferBGpu = cv::gpu::GpuMat(itsFFTSizeSpatial, CV_32FC2);

  // accumulator - 1 channel, complex, padded
  itsAccumulatorGpu = cv::gpu::GpuMat(itsFFTSizeSpatial, CV_32FC2);

  // saliency - 1 channel, real, unpadded
  itsSaliencyGpu = cv::gpu::GpuMat( itsImageSize, CV_32FC1);
  
  printf("itsSaliencyGpu.size(): (%d,%d)\n", itsSaliencyGpu.size().width, itsSaliencyGpu.size().height); 
  /*// We force a release of all prior stuff just in case OpenCV isn't
  // deallocating things when we want more space on the GPU

	// LAB response is same size as input, one channel floating point
  for (int i = 0; i < itsLabResponse.size(); i++)
		itsLabResponse.at(i) = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );

  // O and S response is same size as input, one channel floating point
  itsOResponse = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );

	// FFT lab are sized according to fftsizespatial, two channels
  for (int i = 0; i < itsLabFFT.size(); i++)
		itsLabFFT.at(i) = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC2 );

  // FFT for grayscale is sized according to fftsizeFrequency, two channels
  itsGrayscaleFFT = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );
  itsOBuffer = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );

  // Buffer used for splitting images
  itsSplitBufferFrequency.resize( 2 );
  for (int i = 0; i < itsSplitBufferFrequency.size(); i++)
    itsSplitBufferFrequency.at(i) = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC1 );

  itsSplitBufferSpatial.resize( 2 );
  for (int i = 0; i < itsSplitBufferSpatial.size(); i++)
    itsSplitBufferSpatial.at(i) = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC1 );

  itsSaliencyLab = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  itsGpuBuffer = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC1 );
  itsGpuBuffer3 = cv::gpu::GpuMat( itsInputImage.size(), CV_32FC3 );

  itsGpuBufferSpatial = cv::gpu::GpuMat( itsFFTSizeSpatial, CV_32FC2 );
  for (int i = 0; i < itsGpuBufferFrequency.size(); i++)
    itsGpuBufferFrequency.at(i) = cv::gpu::GpuMat( itsFFTSizeFrequency, CV_32FC2 );*/
}

void sg::SalientGreenGPU::prepareFiltersGPU()
{
  std::cout << "Preparing all filters.\n";

  // Figure out FFT size for spatial based
  int maxhw = static_cast<int>( std::min( itsImageSize.height / 2.0 - 1, itsImageSize.width / 2.0 - 1 ) );
  int largest = DoGCenterSurround::largestFilterSize( maxhw );

  itsMaxFilterSize = cv::Size( largest, largest );

  itsFFTSizeSpatial = getPaddedSize( itsImageSize,
                                     itsMaxFilterSize );

  // Figure out FFT size for frequency based kernels
  itsFFTSizeFrequency = cv::Size(getPaddedSize( itsImageSize )); // LEWIS 1024,512);

  // Create DoG filters
  std::cout << "MaxHW: " << maxhw << "\nLargest: " << largest << 
	           "\nItsMaxFilterSize: (" << itsMaxFilterSize.width <<"," << itsMaxFilterSize.height << ")\nItsFFTSizeSpatial: ("<< 
			   itsFFTSizeSpatial.width <<"," << itsFFTSizeSpatial.height << ")\nItsFFTSizeFrequency: " << 
			   itsFFTSizeFrequency.width <<"," << itsFFTSizeFrequency.height << std::endl;
  itsDoGBank = DoGCenterSurround( maxhw, itsFFTSizeSpatial );
  //itsDoGBankFrequency = DoGCenterSurround( maxhw, itsFFTSizeFrequency );

  // TODO: fix gabor filter to be spatial, not frequency
  // Create Gabor filters
  //cv::Mat lp = lowpass( itsFFTSizeFrequency.width, itsFFTSizeFrequency.height, 0.4, 10 );
  //itsLogGabor = LogGabor( itsFFTSizeFrequency.width, itsFFTSizeFrequency.height, lp );//,
			//6, 6.0, 3.0, 0.55, 2.0 );
  //itsLogGabor.addFilters();

  // Create iterative normalizer
  itsNormalizeIterSpatial = NormalizeIterative( itsImageSize, itsFFTSizeSpatial );
  //itsNormalizeIterFrequency = NormalizeIterative( itsInputImage.size(), itsFFTSizeFrequency );

  std::cout << "Filter creation finished.\n";
}



//! Do LAB+Grayscale conversion, split into channels, and pad for FFT
void sg::SalientGreenGPU::labgPadFFTSplit( cv::gpu::GpuMat const & image, std::vector<cv::gpu::GpuMat> outputChannels)
{
	outputChannels[0].setTo(0.0f);
	outputChannels[1].setTo(0.0f);
	outputChannels[2].setTo(0.0f);
	outputChannels[3].setTo(0.0f);
	// call kernel that does LAB+grayscale conversion
	// put result (padded) into channels
	labgPadFFTSplitWrapper(image, outputChannels[0], outputChannels[1], outputChannels[2], 
							outputChannels[3], itsFFTSizeSpatial.width, itsFFTSizeSpatial.height, 
							itsMaxFilterSize.width, itsMaxFilterSize.height);
}


void sg::SalientGreenGPU::unpadFFT( cv::gpu::GpuMat const & input,  cv::gpu::GpuMat & output)
{
  unpadWrapper(input, output);
}

void sg::SalientGreenGPU::doGaborGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & gabor )
{
  //itsLogGabor.getEdgeResponses( fftImage, itsGpuBufferFrequency[0], itsSplitBufferFrequency, itsDoGBankFrequency );

  //cv::gpu::dft( itsGpuBufferFrequency[0], gabor, gabor.size(), cv::DFT_INVERSE );
  //absFC2Wrapper( gabor );
}

void sg::SalientGreenGPU::doDoGGPU( cv::gpu::GpuMat const & fftImage, cv::gpu::GpuMat & buffer, cv::gpu::GpuMat & output )
{
  itsDoGBank( fftImage, buffer );

  cv::gpu::dft( buffer, buffer, buffer.size(), cv::DFT_INVERSE );
  absAndZeroFC2Wrapper( buffer );
  normalizeGPU( buffer, 255.0f );
  addRealFC2Wrapper(output, buffer, output);
}
