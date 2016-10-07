#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "symmetrical_peak_method.hpp"
#include "canny.hpp"

using namespace std;
using namespace cv;

const char* kAbout =
"Laboratory work"
" Image processing";

const char* kOptions =
"{ @image         | <none> | image to process                 }"
"{ v video        | <none> | video to process                 }"
"{ binsim         |        | binarization by simmetrical peak }"
"{ can canny      |        | canny operator                   }"
"{ h ? help usage |        | print help message               }";

//!!!!!!!!!!!!!!здесь добавьте параметр командной строки!!!!!!!!!!!!!!!!!

int main(int argc, const char** argv) {
	// Парсим аргументы командной строки
	CommandLineParser parser(argc, argv, kOptions);
	parser.about(kAbout);

	// Если получили --h, печатаем help-message и выходим из программы
	if (parser.get<bool>("help")) {
		parser.printMessage();
		return 0;
	}

	Mat src, dst;
	VideoCapture cap(argv[1]);
	cap >> src;
	if (src.empty()) {
		cout << "Failed to open image file '" + parser.get<string>(0) + "'."
			<< endl;
		return 0;
	}
	//вывод исходной картинки
	const string kSrcWindowName = "Source image";
	const int kWaitKeyDelay = 1;
	namedWindow(kSrcWindowName, WINDOW_AUTOSIZE);
	imshow(kSrcWindowName, src);
	waitKey(kWaitKeyDelay);

	//вывод бинарного изображения
	//цикл для видеопотока. В случае картинки отработает 1 раз и выйдет.
	CannyImpl procCan;
	ImageProcessorImpl procBin;
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!добавьте сюда объект вашего класса!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	bool firstFrame = true;
	for (;;) {
		if (src.empty() && !firstFrame)
			break;

		if (parser.get<bool>("binsim"))
		{
			const string kDstBinWindowName = "Destination binarizated image";
			namedWindow(kDstBinWindowName, WINDOW_AUTOSIZE);
			dst = procBin.binarization(src);
			imshow(kDstBinWindowName, dst);
		}
		if (parser.get<bool>("can"))
		{
			const string kDstCanWindowName = "Destination image with Canny";
			namedWindow(kDstCanWindowName, WINDOW_AUTOSIZE);
			dst = procCan.canny(src);
			imshow(kDstCanWindowName, dst);
		}
		//!!!!!!!!!добавьте сюда обрабоку изображения вашим методом по аналогии с тем, что сверху!!!!!!!!!!!

		firstFrame = false;
		imshow(kSrcWindowName, src);
		cap >> src;
		if (waitKey(30) >= 0) break;
	}
	waitKey();
	return 0;
}