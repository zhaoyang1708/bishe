#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>   
#include <cassert>  
#include <vector>  

#define PI 3.1415926
using namespace cv;
using namespace std;

//const char * filePath = "FY2G_SEC_IR1_MSS_20160314_0300.AWX";


//求图像的局部最大值
void getLocalMaxima(Mat scr, float thresh, float *lm_sum, int *lm_num, float *lm_avg)
{
	*lm_sum = 0.0;
	*lm_num = 0;
	*lm_avg = 0.0;
	int count = 0;

	for (int a = 1; a < (scr.rows - 1); a++)
	for (int b = 1; b< (scr.cols - 1); b++)
	{
		float val = scr.at<float>(a, b);
		if ((val >= thresh) && (val >= scr.at<float>(a - 1, b)) && (val >= scr.at<float>(a + 1, b)) && (val >= scr.at<float>(a, b - 1)) && (val >= scr.at<float>(a, b + 1)))
		{
			if (val == 10) count++;
			*lm_sum += val;
			(*lm_num)++;
		}
	}

	if (*lm_num > count)
	{
		*lm_sum = *lm_sum - 10 * count;
		*lm_num = *lm_num - count;
		if (*lm_num > 0)
			*lm_avg = *lm_sum / *lm_num;
		else
			*lm_avg = 0;
	}
	else
		*lm_avg = 0;
}


void N_operation(Mat&scr)
{
	scr.convertTo(scr, CV_32F);		//显示转换成浮点数类型
	normalize(scr, scr, 0.0f, 1.0f, NORM_MINMAX);
	scr=scr*10;
	int lm_num;
	float lm_sum;
	float lm_avg;
	getLocalMaxima(scr, 1, &lm_sum, &lm_num, &lm_avg);
	if (lm_num > 0)
	scr=scr*(10 - lm_avg)*(10 - lm_avg);

}





void cvGabor(Mat&scr, Mat&dst, int width, double lamda, double theta)
{

	Mat gabor_kernel(width, width, CV_32FC1);
	double tmp1, tmp2, xtmp, ytmp, re;
	int i, j, x, y;
	for (i = 0; i < width; i++)
	for (j = 0; j < width; j++)
	{
		x = (i * 16 / (width - 1)) - 8;
		y = (j * 16 / (width - 1)) - 8;		
		xtmp = (float)x*cos(theta) + (float)y*sin(theta);
		ytmp = (float)(-x)*sin(theta) + (float)y*cos(theta);
		tmp1 = exp(-((pow(xtmp, 2) + pow(ytmp, 2)) / 2*pow(lamda, 2)));

		//tmp1 = (1 / (PI*pow(lamda, 2)))*exp(-((pow(xtmp, 2) + pow(ytmp, 2)) / pow(lamda, 2)));
		tmp2 = cos(2 * PI*xtmp / lamda);
		re = tmp1*tmp2;
		re=(gabor_kernel.at<float>(i, j));

	}

	filter2D(scr, dst, CV_32F, gabor_kernel, Point(-1, -1));
	dst=abs(dst);
	double max = 0;
	for (int i = 0; i<dst.rows; i++)
	for (int j = 0; j<dst.cols; j++)
	if (dst.at<float>(i,j) >= max)
		max = dst.at<float>(i,j);

	dst=dst*(1 / max);
}




void Itti(Mat&img1)
{
	Mat imageBlue, imageYellow, imageRed;
	vector<Mat> channels;
	split(img1, channels);
	imageBlue = channels.at(0);
	imageYellow = channels.at(1);
	imageRed = channels.at(2);



	Mat Intensity(img1.rows/4, img1.cols/4, CV_32FC1);
	Mat R(img1.rows, img1.cols, CV_32FC1);
	Mat G(img1.rows, img1.cols, CV_32FC1);
	Mat B(img1.rows, img1.cols, CV_32FC1);
	Mat Y(img1.rows, img1.cols, CV_32FC1);
	Mat C(img1.rows/4, img1.cols/4, CV_32FC1);
	Mat S(img1.rows/4, img1.cols/4, CV_32FC1);
	Intensity = (imageBlue + imageYellow + imageRed) / 3;
	R = imageRed - (imageYellow + imageBlue) / 2;
	G = imageYellow - (imageRed + imageBlue) / 2;
	B = imageBlue - (imageRed + imageYellow) / 2;
	Y = (imageRed + imageYellow) / 2 - abs(imageRed - imageYellow) / 2 - imageBlue;

	//Gscale(Intensity,&I[9]);
	
	//Mat tmp = Intensity;

	Mat I[9];
	I[0] = Intensity;
	pyrDown(I[0], I[1], Size(I[0].cols / 2, I[0].rows / 2));
	pyrDown(I[1], I[2], Size(I[1].cols / 2, I[1].rows / 2));
	pyrDown(I[2], I[3], Size(I[2].cols / 2, I[2].rows / 2));
	pyrDown(I[3], I[4], Size(I[3].cols / 2, I[3].rows / 2));
	pyrDown(I[4], I[5], Size(I[4].cols / 2, I[4].rows / 2));
	pyrDown(I[5], I[6], Size(I[5].cols / 2, I[5].rows / 2));
	pyrDown(I[6], I[7], Size(I[6].cols / 2, I[6].rows / 2));
	pyrDown(I[7], I[8], Size(I[7].cols / 2, I[7].rows / 2));
	
	
	Mat Iadd;
	Iadd = Mat::zeros(I[2].size(), CV_32FC1);
	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s=0; 
			s = c + delta;
			Mat tem;
			tem = Mat::zeros(I[c].size(), CV_32FC1);
			resize(I[s], tem, I[c].size(), 0, 0, INTER_NEAREST);
			absdiff(I[c], tem, tem);
			N_operation(tem);
			resize(tem, tem, Iadd.size(), 0, 0, INTER_NEAREST);
			add(Iadd, tem, Iadd);
		}
	}
	N_operation(Iadd);
	//resize(Iadd, Iadd, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Iadd, Iadd, 0.0f, 1.0f, NORM_MINMAX);
	imshow("I", Iadd);



	Mat Red[9],Blue[9],Green[9],Yellow[9];


	Red[0] = R;
	pyrDown(Red[0], Red[1], Size(Red[0].cols / 2, Red[0].rows / 2));
	pyrDown(Red[1], Red[2], Size(Red[1].cols / 2, Red[1].rows / 2));
	pyrDown(Red[2], Red[3], Size(Red[2].cols / 2, Red[2].rows / 2));
	pyrDown(Red[3], Red[4], Size(Red[3].cols / 2, Red[3].rows / 2));
	pyrDown(Red[4], Red[5], Size(Red[4].cols / 2, Red[4].rows / 2));
	pyrDown(Red[5], Red[6], Size(Red[5].cols / 2, Red[5].rows / 2));
	pyrDown(Red[6], Red[7], Size(Red[6].cols / 2, Red[6].rows / 2));
	pyrDown(Red[7], Red[8], Size(Red[7].cols / 2, Red[7].rows / 2));

	Green[0] = G;
	pyrDown(Green[0], Green[1], Size(Green[0].cols / 2, Green[0].rows / 2));
	pyrDown(Green[1], Green[2], Size(Green[1].cols / 2, Green[1].rows / 2));
	pyrDown(Green[2], Green[3], Size(Green[2].cols / 2, Green[2].rows / 2));
	pyrDown(Green[3], Green[4], Size(Green[3].cols / 2, Green[3].rows / 2));
	pyrDown(Green[4], Green[5], Size(Green[4].cols / 2, Green[4].rows / 2));
	pyrDown(Green[5], Green[6], Size(Green[5].cols / 2, Green[5].rows / 2));
	pyrDown(Green[6], Green[7], Size(Green[6].cols / 2, Green[6].rows / 2));
	pyrDown(Green[7], Green[8], Size(Green[7].cols / 2, Green[7].rows / 2));


	Blue[0] = B;
	pyrDown(Blue[0], Blue[1], Size(Blue[0].cols / 2, Blue[0].rows / 2));
	pyrDown(Blue[1], Blue[2], Size(Blue[1].cols / 2, Blue[1].rows / 2));
	pyrDown(Blue[2], Blue[3], Size(Blue[2].cols / 2, Blue[2].rows / 2));
	pyrDown(Blue[3], Blue[4], Size(Blue[3].cols / 2, Blue[3].rows / 2));
	pyrDown(Blue[4], Blue[5], Size(Blue[4].cols / 2, Blue[4].rows / 2));
	pyrDown(Blue[5], Blue[6], Size(Blue[5].cols / 2, Blue[5].rows / 2));
	pyrDown(Blue[6], Blue[7], Size(Blue[6].cols / 2, Blue[6].rows / 2));
	pyrDown(Blue[7], Blue[8], Size(Blue[7].cols / 2, Blue[7].rows / 2));



	Yellow[0] = Y;
	pyrDown(Yellow[0], Yellow[1], Size(Yellow[0].cols / 2, Yellow[0].rows / 2));
	pyrDown(Yellow[1], Yellow[2], Size(Yellow[1].cols / 2, Yellow[1].rows / 2));
	pyrDown(Yellow[2], Yellow[3], Size(Yellow[2].cols / 2, Yellow[2].rows / 2));
	pyrDown(Yellow[3], Yellow[4], Size(Yellow[3].cols / 2, Yellow[3].rows / 2));
	pyrDown(Yellow[4], Yellow[5], Size(Yellow[4].cols / 2, Yellow[4].rows / 2));
	pyrDown(Yellow[5], Yellow[6], Size(Yellow[5].cols / 2, Yellow[5].rows / 2));
	pyrDown(Yellow[6], Yellow[7], Size(Yellow[6].cols / 2, Yellow[6].rows / 2));
	pyrDown(Yellow[7], Yellow[8], Size(Yellow[7].cols / 2, Yellow[7].rows / 2));



	Mat RGadd;
	RGadd = Mat::zeros(Red[2].size(), CV_32FC1);
	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s = 0;
			s = c + delta;
			Mat tem,tem1,tem2;
			tem = Mat::zeros(Red[c].size(), CV_32FC1);
			tem1 = Mat::zeros(Red[c].size(), CV_32FC1);
			tem2 = Mat::zeros(Red[c].size(), CV_32FC1);
			tem1 = Red[c] - Yellow[c];
			tem2 = Yellow[s] - Red[s];
			resize(tem2, tem2, tem1.size(), 0, 0, INTER_NEAREST);
			absdiff(tem1, tem2, tem);
			N_operation(tem);
			resize(tem, tem, RGadd.size(), 0, 0, INTER_NEAREST);
			add(RGadd, tem, RGadd);
		}
	}


	Mat BYadd;
	BYadd = Mat::zeros(Blue[2].size(), CV_32FC1);
	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s = 0;
			s = c + delta;
			Mat tem, tem1, tem2;
			tem = Mat::zeros(Blue[c].size(), CV_32FC1);
			tem1 = Mat::zeros(Blue[c].size(), CV_32FC1);
			tem2 = Mat::zeros(Blue[c].size(), CV_32FC1);
			tem1 = Blue[c] - Yellow[c];
			tem2 = Yellow[s] - Blue[s];
			resize(tem2, tem2, tem1.size(), 0, 0, INTER_NEAREST);
			absdiff(tem1, tem2, tem);
			N_operation(tem);
			resize(tem, tem, BYadd.size(), 0, 0, INTER_NEAREST);
			add(BYadd, tem, BYadd);
		}
	}

	
	C = RGadd + BYadd;
	//resize(C,C, img1.size(), 0, 0, INTER_NEAREST);
	normalize(C, C, 0.0f, 1.0f, NORM_MINMAX);
	//normalize(RGadd, RGadd, 0.0f, 1.0f, NORM_MINMAX);
	//normalize(BYadd, BYadd, 0.0f, 1.0f, NORM_MINMAX);
	//imshow("RG", RGadd);
	//imshow("BY", BYadd);
	imshow("C", C);


	Mat O,orient0, orient45, orient90, orient135;
	O = Mat::zeros(Intensity.size(), CV_32FC1);
	orient0 = Mat::zeros(Intensity.size(), CV_32FC1);
	orient45 = Mat::zeros(Intensity.size(), CV_32FC1);
	orient90 = Mat::zeros(Intensity.size(), CV_32FC1);
	orient135 = Mat::zeros(Intensity.size(), CV_32FC1);
	


	
	cvGabor(Intensity, orient0, 11, 5, 0);
	cvGabor(Intensity, orient45, 11, 5, PI / 4);
	cvGabor(Intensity, orient90, 11, 5, PI / 2);
	cvGabor(Intensity, orient135, 11, 5, 3 * PI / 4);


	Mat O_0[9];
	O_0[0] = orient0;
	pyrDown(O_0[0], O_0[1], Size(O_0[0].cols / 2, O_0[0].rows / 2));
	pyrDown(O_0[1], O_0[2], Size(O_0[1].cols / 2, O_0[1].rows / 2));
	pyrDown(O_0[2], O_0[3], Size(O_0[2].cols / 2, O_0[2].rows / 2));
	pyrDown(O_0[3], O_0[4], Size(O_0[3].cols / 2, O_0[3].rows / 2));
	pyrDown(O_0[4], O_0[5], Size(O_0[4].cols / 2, O_0[4].rows / 2));
	pyrDown(O_0[5], O_0[6], Size(O_0[5].cols / 2, O_0[5].rows / 2));
	pyrDown(O_0[6], O_0[7], Size(O_0[6].cols / 2, O_0[6].rows / 2));
	pyrDown(O_0[7], O_0[8], Size(O_0[7].cols / 2, O_0[7].rows / 2));


	Mat O_45[9];
	O_45[0] = orient45;
	pyrDown(O_45[0], O_45[1], Size(O_45[0].cols / 2, O_45[0].rows / 2));
	pyrDown(O_45[1], O_45[2], Size(O_45[1].cols / 2, O_45[1].rows / 2));
	pyrDown(O_45[2], O_45[3], Size(O_45[2].cols / 2, O_45[2].rows / 2));
	pyrDown(O_45[3], O_45[4], Size(O_45[3].cols / 2, O_45[3].rows / 2));
	pyrDown(O_45[4], O_45[5], Size(O_45[4].cols / 2, O_45[4].rows / 2));
	pyrDown(O_45[5], O_45[6], Size(O_45[5].cols / 2, O_45[5].rows / 2));
	pyrDown(O_45[6], O_45[7], Size(O_45[6].cols / 2, O_45[6].rows / 2));
	pyrDown(O_45[7], O_45[8], Size(O_45[7].cols / 2, O_45[7].rows / 2));




	Mat O_90[9];
	O_90[0] = orient90;
	pyrDown(O_90[0], O_90[1], Size(O_90[0].cols / 2, O_90[0].rows / 2));
	pyrDown(O_90[1], O_90[2], Size(O_90[1].cols / 2, O_90[1].rows / 2));
	pyrDown(O_90[2], O_90[3], Size(O_90[2].cols / 2, O_90[2].rows / 2));
	pyrDown(O_90[3], O_90[4], Size(O_90[3].cols / 2, O_90[3].rows / 2));
	pyrDown(O_90[4], O_90[5], Size(O_90[4].cols / 2, O_90[4].rows / 2));
	pyrDown(O_90[5], O_90[6], Size(O_90[5].cols / 2, O_90[5].rows / 2));
	pyrDown(O_90[6], O_90[7], Size(O_90[6].cols / 2, O_90[6].rows / 2));
	pyrDown(O_90[7], O_90[8], Size(O_90[7].cols / 2, O_90[7].rows / 2));



	Mat O_135[9];
	O_135[0] = orient135;
	pyrDown(O_135[0], O_135[1], Size(O_135[0].cols / 2, O_135[0].rows / 2));
	pyrDown(O_135[1], O_135[2], Size(O_135[1].cols / 2, O_135[1].rows / 2));
	pyrDown(O_135[2], O_135[3], Size(O_135[2].cols / 2, O_135[2].rows / 2));
	pyrDown(O_135[3], O_135[4], Size(O_135[3].cols / 2, O_135[3].rows / 2));
	pyrDown(O_135[4], O_135[5], Size(O_135[4].cols / 2, O_135[4].rows / 2));
	pyrDown(O_135[5], O_135[6], Size(O_135[5].cols / 2, O_135[5].rows / 2));
	pyrDown(O_135[6], O_135[7], Size(O_135[6].cols / 2, O_135[6].rows / 2));
	pyrDown(O_135[7], O_135[8], Size(O_135[7].cols / 2, O_135[7].rows / 2));

	Mat Pyr_0, Pyr_45, Pyr_90, Pyr_135;
	Pyr_0 = Mat::zeros(I[2].size(), CV_32FC1);
	Pyr_45 = Mat::zeros(I[2].size(), CV_32FC1);
	Pyr_90 = Mat::zeros(I[2].size(), CV_32FC1);
	Pyr_135 = Mat::zeros(I[2].size(), CV_32FC1);
	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s = 0;
			s = c + delta;
			Mat tem;
			tem = Mat::zeros(O_0[c].size(), CV_32FC1);
			resize(O_0[s], tem, O_0[c].size(), 0, 0, INTER_NEAREST);
			absdiff(O_0[c], tem, tem);
			N_operation(tem);
			resize(tem, tem, Pyr_0.size(), 0, 0, INTER_NEAREST);
			add(Pyr_0, tem, Pyr_0);

		}
	}


	N_operation(Pyr_0);

	//resize(Pyr_0, Pyr_0, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Pyr_0, Pyr_0, 0.0f, 1.0f, NORM_MINMAX);
	//namedWindow("Pyr_0", CV_WINDOW_AUTOSIZE);
	//imshow("Pyr_0", Pyr_0);



	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s = 0;
			s = c + delta;
			Mat tem;
			tem = Mat::zeros(O_45[c].size(), CV_32FC1);
			resize(O_45[s], tem, O_45[c].size(), 0, 0, INTER_NEAREST);
			absdiff(O_45[c], tem, tem);
			N_operation(tem);
			resize(tem, tem, Pyr_45.size(), 0, 0, INTER_NEAREST);
			add(Pyr_45, tem, Pyr_45);

		}
	}


	N_operation(Pyr_45);
//	resize(Pyr_45, Pyr_45, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Pyr_45, Pyr_45, 0.0f, 1.0f, NORM_MINMAX);
	//imshow("Pyr_45", Pyr_45);



	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s = 0;
			s = c + delta;
			Mat tem;
			tem = Mat::zeros(O_90[c].size(), CV_32FC1);
			resize(O_90[s], tem, O_90[c].size(), 0, 0, INTER_NEAREST);
			absdiff(O_90[c], tem, tem);
			N_operation(tem);
			resize(tem, tem, Pyr_90.size(), 0, 0, INTER_NEAREST);
			add(Pyr_90, tem, Pyr_90);

		}
	}



	N_operation(Pyr_90);
//	resize(Pyr_90, Pyr_90, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Pyr_90, Pyr_90, 0.0f, 1.0f, NORM_MINMAX);
	//imshow("Pyr_90", Pyr_90);




	for (int c = 2; c < 5; c++)
	{
		for (int delta = 3; delta < 5; delta++)
		{
			int s = 0;
			s = c + delta;
			Mat tem;
			tem = Mat::zeros(O_135[c].size(), CV_32FC1);
			resize(O_135[s], tem, O_135[c].size(), 0, 0, INTER_NEAREST);
			absdiff(O_135[c], tem, tem);
			N_operation(tem);
			resize(tem, tem, Pyr_135.size(), 0, 0, INTER_NEAREST);
			add(Pyr_135, tem, Pyr_135);
		}
	}

	N_operation(Pyr_135);

	//resize(Pyr_135, Pyr_135, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Pyr_135, Pyr_135, 0.0f, 1.0f, NORM_MINMAX);
//	imshow("Pyr_135", Pyr_135);




	O = Pyr_0 + Pyr_45 + Pyr_90 + Pyr_135;
	normalize(O, O, 0.0f, 1.0f, NORM_MINMAX);
	imshow("O", O);
	S = Iadd + C+O;
	normalize(S, S, 0.0f, 1.0f, NORM_MINMAX);
	imshow("S", S);

}




int main()
{
	
	//Mat img_1 = imread("aircrafts.png");
	//Mat img_1 = imread("kejian1.bmp");
	//namedWindow("img1", CV_WINDOW_AUTOSIZE);
	//imshow("img1", img_1);



	Mat img_1;
	fstream fp;
	fp.open("FY2G_SEC_IR1_MSS_20160314_0300.AWX", ios::binary|ios::in);
	//文件指针跳到记录长度数据块处
	fp.seekg(20, ios::beg);
	short iRecordLen;
	fp.read((char*)&iRecordLen, sizeof(short));
	short iRecordNum;
	fp.read((char*)&iRecordNum, sizeof(short));
	short iHeight, iWidth;

	//文件指针跳到图像宽度数据块处
	fp.seekg(62, ios::beg);
	fp.read((char*)&iWidth, sizeof(short));
	fp.read((char*)&iHeight, sizeof(short));

	img_1.create(iHeight, iWidth, CV_8UC3);//CV_8UC1
	int size = img_1.rows * img_1.cols;
	fp.seekg(iRecordLen * iRecordNum, ios::beg);

	char *pSource = new char[size];

	fp.read(pSource, size);
	fp.close();

	for (int i = 0; i<img_1.rows; i++)
	{
		uchar * ps = img_1.ptr<uchar>(i);
		for (int j = 0; j < img_1.cols; j++)
		{
			ps[3*j] = *(pSource + i*img_1.cols + j + 0);
			ps[3 * j + 1] = ps[3 * j];//单通道时去掉
			ps[3 * j + 2] = ps[3 * j];//单通道时去掉

		}
	}
	delete pSource;

	/*
	int l;
	ifstream in(filename, ios::in | ios::binary);
	in.seekg(22, ios::beg);
	char *tmp;
	in.read(tmp, 2);

	cout << tmp << endl;
	in.seekg(63, ios::beg);
	img_1.cols = in.tellg();
	in.seekg(65, ios::beg);
	img_1.rows = in.tellg();
	for (int i = 0; i < img_1.rows;i++)
	for (int j = 0; j < img_1.cols; j++)
	{
		in.seekg(l++, ios::beg);
		img_1.at<int>(i, j) = in.tellg();
	}
	in.close();
	*/

	
	imshow("image1", img_1);
	Mat dst;
	Itti(img_1);
	waitKey(0);
	return 0;
}