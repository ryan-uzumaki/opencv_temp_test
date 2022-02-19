#include <algorithm>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/objdetect.hpp"
#include "stdlib.h"
#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  

using namespace std;
using namespace cv;


void object_recognition(Mat& image);

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
	return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
	return std::distance(first, std::max_element(first, last));
}


int main() {
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		Mat temp = Mat::zeros(frame.size(), frame.type());
		Mat m = Mat::zeros(frame.size(), frame.type());
		addWeighted(frame, 0.4, m, 0.0, 50, temp);
		Mat dst;
		bilateralFilter(temp, dst, 5, 20, 20);
		Mat m_ResImg;
		cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(m_ResImg, m_ResImg, element);//���и�ʴ����
		erode(m_ResImg, m_ResImg, element);//���и�ʴ����
		erode(m_ResImg, m_ResImg, element);//���и�ʴ����
		Mat mask;
		inRange(m_ResImg, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
		vector<double>area;
		for (int i = 0; i < contours.size(); i++) {
			area.push_back(contourArea(contours[i]));
		}
		size_t maxIndex = argmax(area.begin(), area.end());
		Rect ret_1 = boundingRect(contours[maxIndex]);
		int avgX, avgY;
		avgX = (ret_1.x + ret_1.width) / 2;
		avgY = (ret_1.y + ret_1.height) / 2;
		for (int i = 0; i < contours.size(); i++) {
			for (int j = 0; j < contours[i].size(); j++) {
				Point P = Point(contours[i][j].x, contours[i][j].y);
				Mat Contours = Mat::zeros(m_ResImg.size(), CV_8UC1);  //����
				Contours.at<uchar>(P) = 255;
			}
			Rect box(ret_1.x, ret_1.y, ret_1.width, ret_1.height);
			rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			drawContours(frame, contours, maxIndex, Scalar(255, 0, 0), 2, 8, hierarchy);
		}

		/*	for (size_t k=0; k < contours.size(); k++) {
				Rect ret_1 = boundingRect(contours[k]);
				int avgX, avgY;
				avgX = (ret_1.x + ret_1.width) / 2;
				avgY = (ret_1.y + ret_1.height) / 2;
				Rect box(ret_1.x, ret_1.y, ret_1.width, ret_1.height);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			}*/

		namedWindow("detected", WINDOW_FREERATIO);
		imshow("detected", frame);
		int c = waitKey(1);
		if (c == 27) { // �˳�
			break;
		}
	}
	return 0;
}



void object_recognition(Mat& image) {
	Mat temp = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 0.19, m, 0.5, 0, temp);
	Mat dst;
	bilateralFilter(temp, dst, 5, 20, 20);
	Mat m_ResImg;
	cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	//dilate(m_ResImg, m_ResImg, element);//�������
	//dilate(m_ResImg, m_ResImg, element);
	//dilate(m_ResImg, m_ResImg, element);

	erode(m_ResImg, m_ResImg, element);//���и�ʴ����
	erode(m_ResImg, m_ResImg, element);//���и�ʴ����
	erode(m_ResImg, m_ResImg, element);//���и�ʴ����
	//cvtColor(m_ResImg, m_ResImg, COLOR_HSV2BGR);
	//unsigned char pixelB, pixelG, pixelR;
	//unsigned char DifMax = 10;             //������ɫ���ֵ���ֵ����
	//unsigned char B = 138, G = 63, R = 23; //��ͨ������ֵ�趨���������ɫ����
	//Mat  HSVImg_after_erode = m_ResImg.clone();
	//for (int i = 0; i < m_ResImg.rows; i++)   //ͨ����ɫ������ͼƬ���ж�ֵ������
	//{
	//	for (int j = 0; j < m_ResImg.cols; j++)
	//	{
	//		pixelB = m_ResImg.at<Vec3b>(i, j)[0]; //��ȡͼƬ����ͨ����ֵ
	//		pixelG = m_ResImg.at<Vec3b>(i, j)[1];
	//		pixelR = m_ResImg.at<Vec3b>(i, j)[2];

	//		if (abs(pixelB - B) < DifMax && abs(pixelG - G) < DifMax && abs(pixelR - R) < DifMax)
	//		{                                           //������ͨ����ֵ�͸���ͨ����ֵ���бȽ�
	//			HSVImg_after_erode.at<Vec3b>(i, j)[0] = 255;     //������ɫ��ֵ��Χ�ڵ����óɰ�ɫ
	//			HSVImg_after_erode.at<Vec3b>(i, j)[1] = 255;
	//			HSVImg_after_erode.at<Vec3b>(i, j)[2] = 255;
	//		}
	//		else
	//		{
	//			HSVImg_after_erode.at<Vec3b>(i, j)[0] = 0;        //��������ɫ��ֵ��Χ�ڵ�����Ϊ��ɫ
	//			HSVImg_after_erode.at<Vec3b>(i, j)[1] = 0;
	//			HSVImg_after_erode.at<Vec3b>(i, j)[2] = 0;
	//		}
	//	}
	//}
	Mat mask;
	inRange(m_ResImg, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
	//cvtColor(HSVImg_after_erode, HSVImg_after_erode, COLOR_BGR2GRAY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	/*double cnts;
	cnts = contourArea(contours);
	RotatedRect rrt = minAreaRect(cnts);
	Mat pts;
	boxPoints(rrt, pts);
	drawContours(frame, contours, 0, Scalar(0, 0, 255), -1, 8);*/
	for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < contours[i].size(); j++) {
			Point P = Point(contours[i][j].x, contours[i][j].y);
			Mat Contours = Mat::zeros(m_ResImg.size(), CV_8UC1);  //����
			Contours.at<uchar>(P) = 255;
		}
		drawContours(image, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);
	}
	namedWindow("detected", WINDOW_FREERATIO);
	imshow("detected", image);
}

void detect_object(Mat& imageSource) {
	//imshow("Source Image", imageSource);
	Mat image = Mat::zeros(imageSource.size(), imageSource.type());
	image = imageSource.clone();
	//GaussianBlur(imageSource, image, Size(3, 3), 0);
	Canny(image, image, 50, 100);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //����
	for (int i = 0; i < contours.size(); i++) {
		//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���
		for (int j = 0; j < contours[i].size(); j++) {
			//���Ƴ�contours���������е����ص�
			Point P = Point(contours[i][j].x, contours[i][j].y);
			Contours.at<uchar>(P) = 255;
		}

		//���hierarchy��������
		/*char ch[256];
		sprintf_s(ch, "%d", i);
		string str = ch;
		cout << "����hierarchy�ĵ�" << str << " ��Ԫ������Ϊ��" << endl << hierarchy[i] << endl << endl;*/

		//��������
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
	}
	imshow("Contours Image", imageContours); //����
	//imshow("Point of Contours", Contours);   //����contours�ڱ�������������㼯
	waitKey(0);
}
