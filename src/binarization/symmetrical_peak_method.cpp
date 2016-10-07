#include "symmetrical_peak_method.hpp"

using namespace cv;

Mat ImageProcessorImpl::binarization(const Mat &src) {
	Mat src_cpy;
	src.copyTo(src_cpy);

	//преобразуем изображение к ч\б
	Mat src_gray;
	if (src_cpy.channels() == 3)
		cvtColor(src_cpy, src_gray, COLOR_BGR2GRAY);
	else
		src_cpy.copyTo(src_gray);


	//считаем гистограмму интенсивностей
	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRanges[] = { range };
	calcHist(&src_gray, 1, 0, Mat(), hist, 1, &histSize, histRanges);
		
	//задаем ширину и высоту дл€ картинки с гистограммой
	int histWidth = 768, histHeight = 480;
	//ширина одного столбца на картинке
	int binWidth = cvRound((double)histWidth / histSize);
	//создаем пустую белую картинку
	Mat hist_img = Mat(histHeight, histWidth, CV_8UC1, Scalar(255));
	
	//нормируем гистограмму по высоте картинки
	normalize(hist, hist, 0, hist_img.rows, NORM_MINMAX, -1, Mat());
	//сглаживаем гистограмму
	Mat histSmooth = histoSmooth(hist, histSize, 1);
	
	double maxVal = 0;
	int maxIdx;
	for (int i = 1; i < histSize; i++) {
		//рисуем линии, соедин€ющие два соседних столбца
		line(hist_img, Point(binWidth * (i - 1), histHeight - cvRound(histSmooth.at<float>(i - 1))),
			Point(binWidth * i, histHeight - cvRound(histSmooth.at<float>(i))),
			Scalar(0), 2, 8, 0);
		//параллельно ищем максимальный пик
		if (histSmooth.at<float>(i - 1) > maxVal) {
			maxVal = histSmooth.at<float>(i - 1);
			maxIdx = i;
		}
	}
	
	if (maxIdx < 128) {
		src_cpy = inversion(src_cpy);
		return inversion(binarization(src_cpy));
	}

	//считаем количество пикселей, €ркость которых больше, чем у максимального пика
	float sum = 0;
	for (int i = maxIdx; i < 256; i++) {
		sum += histSmooth.at<float>(i);
	}
	
	//находим порог по формуле
	int idx;
	int treashold;

	for (idx = 256; idx > maxIdx; --idx) {
		float currSum = 0;
		for (int i = idx; i < 256; i++) {
			currSum += histSmooth.at<float>(i);
		}
		if (currSum < 0.055*sum && currSum > 0.045*sum)
			break;
	}
	treashold = maxIdx - (idx - maxIdx);
	
	line(hist_img, Point(binWidth * treashold, 0), Point(binWidth * treashold, histHeight), Scalar(128), 2);

	//примен€ем порог
	for (int i = 0; i < src_gray.rows * src_gray.cols; i++)
		if (src_gray.data[i] < treashold)
			src_gray.data[i] = 0;
		else
			src_gray.data[i] = 255;
		
	//рисуем гистограмму
	const std::string kHistWindowName = "Hist image";
	const int kWaitKeyDelay = 1;
	namedWindow(kHistWindowName, WINDOW_AUTOSIZE);
	imshow(kHistWindowName, hist_img);
	waitKey(kWaitKeyDelay);

	return src_gray;
}

Mat ImageProcessorImpl::histoSmooth(const Mat &src, const int histSize, const int smoothDegree) {
	Mat src_cpy;
	src.copyTo(src_cpy);
	Mat dst;
	src.copyTo(dst);
	for (int d = 0; d < smoothDegree; d++) {
		for (int i = 1; i < histSize - 1; i++) {
			dst.at<float>(i) = (src_cpy.at<float>(i - 1) + src_cpy.at<float>(i) + src_cpy.at<float>(i + 1)) / 3;
		}
		dst.at<float>(0) = (src.at<float>(0) + src.at<float>(1)) / 2;
		dst.at<float>(histSize - 1) = (src_cpy.at<float>(histSize - 2) + src_cpy.at<float>(histSize - 1)) / 2;
		dst.copyTo(src_cpy);
	}
	return dst;
}

Mat ImageProcessorImpl::inversion(const cv::Mat &src) {
	Mat dst;
	src.copyTo(dst);
	for (int i = 0; i < dst.rows * dst.cols; i++)
		dst.data[i] = 255 - dst.data[i];
	return dst;
}