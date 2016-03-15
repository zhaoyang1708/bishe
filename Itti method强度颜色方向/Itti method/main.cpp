#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>   
#include <cassert>  
#include <vector>  
#define PI 3.1415926
using namespace cv;
using namespace std;


/*
typedef struct GaussPyr
{
	Mat lev[9];
}GaussPyr;

Size PyrSize[9];


void initPyr(GaussPyr *p)
{
	for (int i = 0; i < 9; i++)
		p->lev[i] = Mat(PyrSize[i], CV_64F, 1);
}
*/
/*
//c-s过程中用到的跨尺度相减
void overScaleSub(Mat s1, Mat s2, Mat dst)
{
resize(s2, dst,s1.size(),0,0,0);
absdiff(s1, dst, dst);
}

*/



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


//N
void N_operation(Mat&scr)
{
	//double min, max;
	//minMaxLoc(scr, &min, &max);	//Testing
	//cout << min << "\t" << max << endl;
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



//C_S过程，金字塔最底层为第0层(实际上（c,s）=(3,6),(3,7),(4,7),(4,8),(5,8),(5,9),)
/*void CS_operation(GaussPyr *p1, GaussPyr *p2, Mat dst)
{
for (int c = 2; c < 5; c++)
for (int delta = 3, s = c + delta; delta < 5; delta++, s = c + delta)
{
Mat tem_c(PyrSize[c], CV_64F);
//IplImage *tem_c = cvCreateImage(PyrSize[c], IPL_DEPTH_64F, 1);
overScaleSub(p1->lev[c], p2->lev[s], tem_c);
Mat tem_5lev(PyrSize[4], CV_64F, 1);
normalize(tem_c, tem_5lev, 1.0, 0.0, NORM_MINMAX);
add(tem_5lev, dst, dst, NULL);

}
}
*/


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

		tmp1 = (1 / (PI*pow(lamda, 2)))*exp(-((pow(xtmp, 2) + pow(ytmp, 2)) / pow(lamda, 2)));
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
	Mat imageBlue, imageGreen, imageRed;
	vector<Mat> channels;
	split(img1, channels);
	imageBlue = channels.at(0);
	imageGreen = channels.at(1);
	imageRed = channels.at(2);

	//namedWindow("img", CV_WINDOW_AUTOSIZE);
	//imshow("img", imageRed);


	Mat Intensity(img1.rows, img1.cols, CV_32FC1);
	Mat I[9];
	Mat R(img1.rows, img1.cols, CV_32FC1);
	Mat G(img1.rows, img1.cols, CV_32FC1);
	Mat B(img1.rows, img1.cols, CV_32FC1);
	Mat Y(img1.rows, img1.cols, CV_32FC1);
	Mat C(img1.rows, img1.cols, CV_32FC1);
	Mat S(img1.rows, img1.cols, CV_32FC1);
	Intensity = (imageBlue + imageGreen + imageRed) / 3;
	R = imageRed - (imageGreen + imageBlue) / 2;
	G = imageGreen - (imageRed + imageBlue) / 2;
	B = imageBlue - (imageRed + imageGreen) / 2;
	Y = (imageRed + imageGreen) / 2 - abs(imageRed - imageGreen) / 2 - imageBlue;


	Mat tmp = Intensity;
	pyrDown(tmp, I[1], Size(tmp.cols / 2, tmp.rows / 2));
	Mat tmp1 = I[1];
	pyrDown(tmp1, I[2], Size(tmp1.cols / 2, tmp1.rows / 2));
	Mat tmp2 = I[2];
	pyrDown(tmp2, I[3], Size(tmp2.cols / 2, tmp2.rows / 2));
	Mat tmp3 = I[3];
	pyrDown(tmp3, I[4], Size(tmp3.cols / 2, tmp3.rows / 2));
	Mat tmp4 = I[4];
	pyrDown(tmp4, I[5], Size(tmp4.cols / 2, tmp4.rows / 2));
	Mat tmp5 = I[5];
	pyrDown(tmp5, I[6], Size(tmp5.cols / 2, tmp5.rows / 2));
	Mat tmp6 = I[6];
	pyrDown(tmp6, I[7], Size(tmp6.cols / 2, tmp6.rows / 2));
	Mat tmp7 = I[7];
	pyrDown(tmp7, I[8], Size(tmp7.cols / 2, tmp7.rows / 2));

	
	
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
			//cout << tem << Iadd;
			add(Iadd, tem, Iadd);
			//imshow("Iadd1", Iadd);

		}
	}
	N_operation(Iadd);
	resize(Iadd, Iadd, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Iadd, Iadd, 0.0f, 1.0f, NORM_MINMAX);
	imshow("I", Iadd);



	Mat Red[9],Green[9],Blue[9],Yellow[9];
	Mat tmpr = R;
	pyrDown(tmpr, Red[1], Size(tmpr.cols / 2, tmpr.rows / 2));
	Mat tmpr1 = Red[1];
	pyrDown(tmpr1, Red[2], Size(tmpr1.cols / 2, tmpr1.rows / 2));
	Mat tmpr2 = Red[2];
	pyrDown(tmpr2, Red[3], Size(tmpr2.cols / 2, tmpr2.rows / 2));
	Mat tmpr3 = Red[3];
	pyrDown(tmpr3, Red[4], Size(tmpr3.cols / 2, tmpr3.rows / 2));
	Mat tmpr4 = Red[4];
	pyrDown(tmpr4, Red[5], Size(tmpr4.cols / 2, tmpr4.rows / 2));
	Mat tmpr5 = Red[5];
	pyrDown(tmpr5, Red[6], Size(tmpr5.cols / 2, tmpr5.rows / 2));
	Mat tmpr6 = Red[6];
	pyrDown(tmpr6, Red[7], Size(tmpr6.cols / 2, tmpr6.rows / 2));
	Mat tmpr7 = Red[7];
	pyrDown(tmpr7, Red[8], Size(tmpr7.cols / 2, tmpr7.rows / 2));

	Mat tmpg = G;
	pyrDown(tmpg, Green[1], Size(tmpg.cols / 2, tmpg.rows / 2));
	Mat tmpg1 = Green[1];
	pyrDown(tmpg1, Green[2], Size(tmpg1.cols / 2, tmpg1.rows / 2));
	Mat tmpg2 = Green[2];
	pyrDown(tmpg2, Green[3], Size(tmpg2.cols / 2, tmpg2.rows / 2));
	Mat tmpg3 = Green[3];
	pyrDown(tmpg3, Green[4], Size(tmpg3.cols / 2, tmpg3.rows / 2));
	Mat tmpg4 = Green[4];
	pyrDown(tmpg4, Green[5], Size(tmpg4.cols / 2, tmpg4.rows / 2));
	Mat tmpg5 = Green[5];
	pyrDown(tmpg5, Green[6], Size(tmpg5.cols / 2, tmpg5.rows / 2));
	Mat tmpg6 = Green[6];
	pyrDown(tmpg6, Green[7], Size(tmpg6.cols / 2, tmpg6.rows / 2));
	Mat tmpg7 = Green[7];
	pyrDown(tmpg7, Green[8], Size(tmpg7.cols / 2, tmpg7.rows / 2));


	Mat tmpb = B;
	pyrDown(tmpb, Blue[1], Size(tmpb.cols / 2, tmpb.rows / 2));
	Mat tmpb1 = Blue[1];
	pyrDown(tmpb1, Blue[2], Size(tmpb1.cols / 2, tmpb1.rows / 2));
	Mat tmpb2 = Blue[2];
	pyrDown(tmpb2, Blue[3], Size(tmpb2.cols / 2, tmpb2.rows / 2));
	Mat tmpb3 = Blue[3];
	pyrDown(tmpb3, Blue[4], Size(tmpb3.cols / 2, tmpb3.rows / 2));
	Mat tmpb4 = Blue[4];
	pyrDown(tmpb4, Blue[5], Size(tmpb4.cols / 2, tmpb4.rows / 2));
	Mat tmpb5 = Blue[5];
	pyrDown(tmpb5, Blue[6], Size(tmpb5.cols / 2, tmpb5.rows / 2));
	Mat tmpb6 = Blue[6];
	pyrDown(tmpb6, Blue[7], Size(tmpb6.cols / 2, tmpb6.rows / 2));
	Mat tmpb7 = Blue[7];
	pyrDown(tmpb7, Blue[8], Size(tmpb7.cols / 2, tmpb7.rows / 2));



	Mat tmpy = Y;
	pyrDown(tmpy, Yellow[1], Size(tmpy.cols / 2, tmpy.rows / 2));
	Mat tmpy1 = Yellow[1];
	pyrDown(tmpy1, Yellow[2], Size(tmpy1.cols / 2, tmpy1.rows / 2));
	Mat tmpy2 = Yellow[2];
	pyrDown(tmpy2, Yellow[3], Size(tmpy2.cols / 2, tmpy2.rows / 2));
	Mat tmpy3 = Yellow[3];
	pyrDown(tmpy3, Yellow[4], Size(tmpy3.cols / 2, tmpy3.rows / 2));
	Mat tmpy4 = Yellow[4];
	pyrDown(tmpy4, Yellow[5], Size(tmpy4.cols / 2, tmpy4.rows / 2));
	Mat tmpy5 = Yellow[5];
	pyrDown(tmpy5, Yellow[6], Size(tmpy5.cols / 2, tmpy5.rows / 2));
	Mat tmpy6 = Yellow[6];
	pyrDown(tmpy6, Yellow[7], Size(tmpy6.cols / 2, tmpy6.rows / 2));
	Mat tmpy7 = Yellow[7];
	pyrDown(tmpy7, Yellow[8], Size(tmpy7.cols / 2, tmpy7.rows / 2));



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
			tem1 = Red[c] - Green[c];
			tem2 = Green[s] - Red[s];
			resize(tem2, tem2, tem1.size(), 0, 0, INTER_NEAREST);
			absdiff(tem1, tem2, tem);
			N_operation(tem);
			resize(tem, tem, RGadd.size(), 0, 0, INTER_NEAREST);
			//cout << tem << Iadd;
			add(RGadd, tem, RGadd);
			//imshow("RGadd", Iadd);

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
			//cout << tem << Iadd;
			add(BYadd, tem, BYadd);
			//imshow("BYadd", Iadd);

		}
	}


	
	C = RGadd + BYadd;
	resize(C,C, img1.size(), 0, 0, INTER_NEAREST);
	normalize(C, C, 0.0f, 1.0f, NORM_MINMAX);
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
	Mat tmp0_0 = orient0;
	pyrDown(tmp0_0, O_0[1], Size(tmp0_0.cols / 2, tmp0_0.rows / 2));
	Mat tmp0_1 = O_0[1];
	pyrDown(tmp0_1, O_0[2], Size(tmp0_1.cols / 2, tmp0_1.rows / 2));
	Mat tmp0_2 = O_0[2];
	pyrDown(tmp0_2, O_0[3], Size(tmp0_2.cols / 2, tmp0_2.rows / 2));
	Mat tmp0_3 = O_0[3];
	pyrDown(tmp0_3, O_0[4], Size(tmp0_3.cols / 2, tmp0_3.rows / 2));
	Mat tmp0_4 = O_0[4];
	pyrDown(tmp0_4, O_0[5], Size(tmp0_4.cols / 2, tmp0_4.rows / 2));
	Mat tmp0_5 = O_0[5];
	pyrDown(tmp0_5, O_0[6], Size(tmp0_5.cols / 2, tmp0_5.rows / 2));
	Mat tmp0_6 = O_0[6];
	pyrDown(tmp0_6, O_0[7], Size(tmp0_6.cols / 2, tmp0_6.rows / 2));
	Mat tmp0_7 = O_0[7];
	pyrDown(tmp0_7, O_0[8], Size(tmp0_7.cols / 2, tmp0_7.rows / 2));


	Mat O_45[9];
	Mat tmp45_0 = orient0;
	pyrDown(tmp45_0, O_45[1], Size(tmp45_0.cols / 2,tmp45_0.rows / 2));
	Mat tmp45_1 = O_45[1];
	pyrDown(tmp45_1, O_45[2], Size(tmp45_1.cols / 2,tmp45_1.rows / 2));
	Mat tmp45_2 = O_45[2];
	pyrDown(tmp45_2, O_45[3], Size(tmp45_2.cols / 2,tmp45_2.rows / 2));
	Mat tmp45_3 = O_45[3];
	pyrDown(tmp45_3, O_45[4], Size(tmp45_3.cols / 2,tmp45_3.rows / 2));
	Mat tmp45_4 = O_45[4];
	pyrDown(tmp45_4, O_45[5], Size(tmp45_4.cols / 2,tmp45_4.rows / 2));
	Mat tmp45_5 = O_45[5];
	pyrDown(tmp45_5, O_45[6], Size(tmp45_5.cols / 2,tmp45_5.rows / 2));
	Mat tmp45_6 = O_45[6];
	pyrDown(tmp45_6, O_45[7], Size(tmp45_6.cols / 2,tmp45_6.rows / 2));
	Mat tmp45_7 = O_45[7];
	pyrDown(tmp45_7, O_45[8], Size(tmp45_7.cols / 2,tmp45_7.rows / 2));




	Mat O_90[9];
	Mat tmp90_0 = orient0;
	pyrDown(tmp90_0, O_90[1], Size(tmp90_0.cols / 2, tmp90_0.rows / 2));
	Mat tmp90_1 = O_90[1];
	pyrDown(tmp90_1, O_90[2], Size(tmp90_1.cols / 2, tmp90_1.rows / 2));
	Mat tmp90_2 = O_90[2];
	pyrDown(tmp90_2, O_90[3], Size(tmp90_2.cols / 2, tmp90_2.rows / 2));
	Mat tmp90_3 = O_90[3];
	pyrDown(tmp90_3, O_90[4], Size(tmp90_3.cols / 2, tmp90_3.rows / 2));
	Mat tmp90_4 = O_90[4];
	pyrDown(tmp90_4, O_90[5], Size(tmp90_4.cols / 2, tmp90_4.rows / 2));
	Mat tmp90_5 = O_90[5];
	pyrDown(tmp90_5, O_90[6], Size(tmp90_5.cols / 2, tmp90_5.rows / 2));
	Mat tmp90_6 = O_90[6];
	pyrDown(tmp90_6, O_90[7], Size(tmp90_6.cols / 2, tmp90_6.rows / 2));
	Mat tmp90_7 = O_90[7];
	pyrDown(tmp90_7, O_90[8], Size(tmp90_7.cols / 2, tmp90_7.rows / 2));




	Mat O_135[9];
	Mat tmp135_0 = orient0;
	pyrDown(tmp135_0, O_135[1], Size(tmp135_0.cols / 2, tmp135_0.rows / 2));
	Mat tmp135_1 = O_135[1];
	pyrDown(tmp135_1, O_135[2], Size(tmp135_1.cols / 2, tmp135_1.rows / 2));
	Mat tmp135_2 = O_135[2];
	pyrDown(tmp135_2, O_135[3], Size(tmp135_2.cols / 2, tmp135_2.rows / 2));
	Mat tmp135_3 = O_135[3];
	pyrDown(tmp135_3, O_135[4], Size(tmp135_3.cols / 2, tmp135_3.rows / 2));
	Mat tmp135_4 = O_135[4];
	pyrDown(tmp135_4, O_135[5], Size(tmp135_4.cols / 2, tmp135_4.rows / 2));
	Mat tmp135_5 = O_135[5];
	pyrDown(tmp135_5, O_135[6], Size(tmp135_5.cols / 2, tmp135_5.rows / 2));
	Mat tmp135_6 = O_135[6];
	pyrDown(tmp135_6, O_135[7], Size(tmp135_6.cols / 2, tmp135_6.rows / 2));
	Mat tmp135_7 = O_135[7];
	pyrDown(tmp135_7, O_135[8], Size(tmp135_7.cols / 2, tmp135_7.rows / 2));

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
	resize(Pyr_0, Pyr_0, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Pyr_0, Pyr_0, 0.0f, 1.0f, NORM_MINMAX);
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
	resize(Pyr_45, Pyr_45, img1.size(), 0, 0, INTER_NEAREST);
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
	resize(Pyr_90, Pyr_90, img1.size(), 0, 0, INTER_NEAREST);
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
	resize(Pyr_135, Pyr_135, img1.size(), 0, 0, INTER_NEAREST);
	normalize(Pyr_135, Pyr_135, 0.0f, 1.0f, NORM_MINMAX);
	//imshow("Pyr_135", Pyr_135);

	O = Pyr_0 + Pyr_45 + Pyr_90 + Pyr_135;
	normalize(O, O, 0.0f, 1.0f, NORM_MINMAX);
	imshow("O", O);
	S = Iadd + C+O;
	normalize(S, S, 0.0f, 1.0f, NORM_MINMAX);
	imshow("S", S);
}




int main()
{
	Mat img_1 = imread("aircrafts.png");
	namedWindow("img1", CV_WINDOW_AUTOSIZE);
	imshow("img1", img_1);
	Mat dst;
	Itti(img_1);
	waitKey(0);
	return 0;
}