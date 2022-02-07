#include<opencv4/opencv2/opencv.hpp>
#include<iostream>
#include<bits/stdc++.h>
#include<math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char const *argv[])
{
    Mat mask;
    Mat result;
    Mat src, srcGray, dst;
    src = imread("1.png");
    //imshow("src",src);
    cvtColor(src, srcGray, COLOR_BGR2GRAY);
    //imshow("srcGray",srcGray);

    threshold(srcGray, dst, 150, 255, THRESH_BINARY);
    // imshow("dst", dst);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(dst, dst, element);
    //dilate(dst, dst, element);
  

 
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
    if( !contours.empty() && !hierarchy.empty() ){    
        vector<vector<Point> >::iterator it;
        for( it = contours.begin(); it != contours.end(); ){  
            //按轮廓长度筛选
            if( arcLength(*it, true) < 450)
                contours.erase(it);
            else it ++;
        }
    }

    //加载svm模型
	Ptr<SVM> svm = StatModel::load<SVM>("/home/huangzengrong/opencv_svm/mnist_svm.xml");


    vector<Rect> boundRect(contours.size()); 
    for(int i = 0; i< contours.size(); i++){ 
        boundRect[i] = boundingRect(contours[i]); 
        //框出轮廓 
        rectangle(src, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 102, 186), 1);

        Mat roi,roi2,roi3;
        Mat drawing2,mask2;
        roi = src(boundRect[i]);

        int len = max(roi.cols,roi.rows);
        Mat drawing = Mat::zeros(Size(len,len), CV_8UC3);
        //drawing = roi2 + drawing;

        int a = (drawing.cols/2)-(roi.cols/2);
        int b = (drawing.rows/2)-(roi.rows/2);
        Mat imageROI = drawing(Rect(a,b, roi.cols, roi.rows));
        threshold(roi,mask2, 150, 255, THRESH_BINARY_INV);
        addWeighted(imageROI,0.5,roi,1.0,0, imageROI);
        //imshow("imageROI",imageROI);

        threshold(drawing,drawing2, 150, 255, THRESH_BINARY_INV);
        resize(drawing2,roi2,Size(28,28));
        imshow("rrr2",roi2);

        vector<Mat> channels;
        Mat aChannels[3];
        split(roi2, aChannels);
        split(roi2, channels);
        threshold(channels[2],roi3, 150, 255, THRESH_BINARY_INV);

        //resize(imageROI,roi3,Size(28,28));
        //imshow("image",imageROI);
        //imshow("roi2",roi2);
        imshow("ROI3",roi3);


        //(1,784)
	    roi3 = roi3.reshape(1, 1);
        //更换数据类型有uchar->float32
	    roi3.convertTo(roi3, CV_32F);


        //预测图片
	    int ret = svm->predict(roi3);
	    //cout << ret << endl;
        string text = to_string(ret);

        Point pt = Point((boundRect[i].tl()+ boundRect[i].br())/2);  //最小外接矩形的中心点坐标
        putText(src, text, boundRect[i].br(), FONT_HERSHEY_PLAIN, 1.5, Scalar(255,255,255), 2, 8, 0);
        
    }
    imshow("img", src);
	
	waitKey(0);

    return 0;

}
