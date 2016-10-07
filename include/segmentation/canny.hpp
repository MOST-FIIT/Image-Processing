#pragma once

#include <memory>
#include <string>

#include "opencv2\imgproc.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2/core/core.hpp"
#include <vector>

class Canny {
public:
	virtual cv::Mat canny(const cv::Mat &src) = 0;
protected:
	virtual cv::Mat gauss(const cv::Mat &src, cv::Size filterSize) = 0;
	virtual cv::Mat sobel(const cv::Mat &src, cv::Mat &dst) = 0;
	virtual cv::Mat nonMaxSuppression(const cv::Mat &src, const cv::Mat &angles) = 0;
	virtual cv::Mat doubleTresholding(const cv::Mat &src, float lowT, float hiT) = 0;
	virtual cv::Mat tracingEdges(const cv::Mat &src) = 0;
};

class CannyImpl : public Canny {
public:
	virtual cv::Mat canny(const cv::Mat &src);
protected:
	virtual cv::Mat gauss(const cv::Mat &src, cv::Size filterSize);
	virtual cv::Mat sobel(const cv::Mat &src, cv::Mat &dst);
	virtual cv::Mat nonMaxSuppression(const cv::Mat &src, const cv::Mat &angles);
	virtual cv::Mat doubleTresholding(const cv::Mat &src, float lowT, float hiT);
	virtual cv::Mat tracingEdges(const cv::Mat &src);
private:
	int checkNeighbour(cv::Mat &img, cv::Point c);
};