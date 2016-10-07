#pragma once

#include <memory>
#include <string>

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

class ImageProcessor {
public:
	virtual cv::Mat binarization(const cv::Mat &src) = 0;
protected:
	virtual cv::Mat histoSmooth(const cv::Mat &src, int histSize, const int smoothDegree) = 0;
	virtual cv::Mat inversion(const cv::Mat &src) = 0;
};

class ImageProcessorImpl : public ImageProcessor {
public:
	virtual cv::Mat binarization(const cv::Mat &src);
protected:
	virtual cv::Mat histoSmooth(const cv::Mat &src, const int histSize, const int smoothDegree);
	virtual cv::Mat inversion(const cv::Mat &src);
};