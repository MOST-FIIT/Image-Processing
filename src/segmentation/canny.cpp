#include "canny.hpp"
using namespace cv;

Mat CannyImpl::canny(const Mat &src) {
	Mat src_cpy;
	src.copyTo(src_cpy);
	
	Mat src_gray;
	if (src_cpy.channels() == 3)
		cvtColor(src_cpy, src_gray, COLOR_BGR2GRAY);
	else
		src_cpy.copyTo(src_gray);

	Mat blured = gauss(src_gray, Size(9, 9));
	Mat gradients;
	Mat angles = sobel(blured, gradients);
	Mat suppressed = nonMaxSuppression(gradients, angles);
	Mat treasholded = doubleTresholding(suppressed, 40, 60);
	Mat traced = tracingEdges(treasholded);
	return traced;
}

Mat CannyImpl::gauss(const Mat &src, Size filterSize) {
	Mat dst;
	GaussianBlur(src, dst, filterSize, 1.4f, 1.4f);
	return dst;
}

Mat CannyImpl::sobel(const Mat &src, Mat &dst) {
	src.copyTo(dst);
	Mat angles = Mat(src.rows, src.cols, CV_8UC1);
	int rows = src.rows;
	int cols = src.cols;

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++) {
			uchar p2 = src.at<uchar>(y - 1, x);
			uchar p3 = src.at<uchar>(y - 1, x + 1);
			uchar p4 = src.at<uchar>(y, x + 1);
			uchar p5 = src.at<uchar>(y + 1, x + 1);
			uchar p6 = src.at<uchar>(y + 1, x);
			uchar p7 = src.at<uchar>(y + 1, x - 1);
			uchar p8 = src.at<uchar>(y, x - 1);
			uchar p9 = src.at<uchar>(y - 1, x - 1);

			int Gx = (p3 + 2 * p4 + p5) - (p9 + 2 * p8 + p7);
			int Gy = (p9 + 2 * p2 + p3) - (p7 + 2 * p6 + p5);

			float G = sqrt((float)Gx*Gx + Gy*Gy);
			
			float dir = (atan2((float)Gy, Gx) / CV_PI) * 180.0f;
			if (((dir < 22.5) && (dir >= -22.5)) || (dir >= 157.5) || (dir < -157.5))
				dir = 0;
			if (((dir >= 22.5) && (dir < 67.5)) || ((dir < -112.5) && (dir >= -157.5)))
				dir = 45;
			if (((dir >= 67.5) && (dir < 112.5)) || ((dir < -67.5) && (dir >= -112.5)))
				dir = 90;
			if (((dir >= 112.5) && (dir < 157.5)) || ((dir < -22.5) && (dir >= -67.5)))
				dir = 135;

			dst.at<uchar>(y, x) = G;
			angles.at<uchar>(y, x) = dir;
		}

	return angles;
}

Mat CannyImpl::nonMaxSuppression(const Mat &src, const cv::Mat &angles) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	
	int rows = src.rows;
	int cols = src.cols;

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++) {
			Point c = Point(x, y);
			Point c2 = Point(x, y - 1);
			Point c3 = Point(x + 1, y - 1);
			Point c4 = Point(x + 1, y);
			Point c5 = Point(x + 1, y + 1);
			Point c6 = Point(x, y + 1);
			Point c7 = Point(x - 1, y + 1);
			Point c8 = Point(x - 1, y);
			Point c9 = Point(x - 1, y - 1);

			if (((angles.at<uchar>(c) == 0)		&& (src.at<uchar>(c) >= src.at<uchar>(c8)) && (src.at<uchar>(y, x) >= src.at<uchar>(c4))) ||	// 0 deg
				((angles.at<uchar>(c) == 45)	&& (src.at<uchar>(c) >= src.at<uchar>(c3)) && (src.at<uchar>(y, x) >= src.at<uchar>(c7))) ||	// 45 deg
				((angles.at<uchar>(c) == 90)	&& (src.at<uchar>(c) >= src.at<uchar>(c2)) && (src.at<uchar>(y, x) >= src.at<uchar>(c6))) ||	// 90 deg
				((angles.at<uchar>(c) == 135)	&& (src.at<uchar>(c) >= src.at<uchar>(c9)) && (src.at<uchar>(y, x) >= src.at<uchar>(c5))))		// 135 deg
				dst.at<uchar>(c) = src.at<uchar>(c);
			else
				dst.at<uchar>(c) = 0;
		}

	return dst;
}

Mat CannyImpl::doubleTresholding(const Mat &src, float lowT, float hiT) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	int rows = src.rows;
	int cols = src.cols;

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++) {
			uchar pix = src.at<uchar>(y, x);
			if (pix < lowT)
				pix = 0;
			else if (pix >= lowT && pix < hiT)
				pix = 127;
			else if (pix >= hiT)
				pix = 255;

			dst.at<uchar>(y, x) = pix;
		}
	return dst;
}

Mat CannyImpl::tracingEdges(const Mat &src) {
	Mat dst;
	src.copyTo(dst);
	int rows = src.rows;
	int cols = src.cols;

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++) {
			Point c = Point(x, y);
			if (dst.at<uchar>(c) == 255) {
				Point c2 = Point(x, y - 1);
				Point c3 = Point(x + 1, y - 1);
				Point c4 = Point(x + 1, y);
				Point c5 = Point(x + 1, y + 1);
				Point c6 = Point(x, y + 1);
				Point c7 = Point(x - 1, y + 1);
				Point c8 = Point(x - 1, y);
				Point c9 = Point(x - 1, y - 1);
			
				checkNeighbour(dst, c2);
				checkNeighbour(dst, c3);
				checkNeighbour(dst, c4);
				checkNeighbour(dst, c5);
				checkNeighbour(dst, c6);
				checkNeighbour(dst, c7);
				checkNeighbour(dst, c8);
				checkNeighbour(dst, c9);
			}
		}

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++) {
			if (dst.at<uchar>(y, x) == 127)
				dst.at<uchar>(y, x) = 0;		
		}

	return dst;
}

int CannyImpl::checkNeighbour(cv::Mat &img, Point c) {
	if (!c.inside(Rect(0, 0, img.cols, img.rows)) || img.at<uchar>(c) != 127)
		return 0;
	img.at<uchar>(c) = 255;

	Point c2 = Point(c.x, c.y - 1);
	Point c3 = Point(c.x + 1, c.y - 1);
	Point c4 = Point(c.x + 1, c.y);
	Point c5 = Point(c.x + 1, c.y + 1);
	Point c6 = Point(c.x, c.y + 1);
	Point c7 = Point(c.x - 1, c.y + 1);
	Point c8 = Point(c.x - 1, c.y);
	Point c9 = Point(c.x - 1, c.y - 1);

	checkNeighbour(img, c2);
	checkNeighbour(img, c3);
	checkNeighbour(img, c4);
	checkNeighbour(img, c5);
	checkNeighbour(img, c6);
	checkNeighbour(img, c7);
	checkNeighbour(img, c8);
	checkNeighbour(img, c9);

	return 0;
}