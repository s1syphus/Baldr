//baldr.cpp

/*
 *  Look at readme
 *
 *
 *    Current trying to get something working, this will need to be heavily refactored soon
 *
 *
 *
 *
 */

//includes


#include "opencv2/opencv.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include<stdio.h>
#include "opencv2/xfeatures2d.hpp"


int main(int argc, char** argv){

  //testing some things
  //this should be moved soon
  
  /*
  cv::VideoCapture cap(argv[1]);
  if(!cap.isOpened()){
    cout<<"ERROR\n";
    return 1;
    }

  int frameCount = 0;
  bool shouldStop = false;
  

  while(!shouldStop){
    cv::Mat frame;
    cap >> frame;
    if(frame.empty()){
      shouldStop = true;
      continue;
    }
    char filename[128];
    sprintf(filename, "../frames/frame_%06d.jpg", frameCount);
    cv::imwrite(filename, frame);
   frameCount++;
  }
  //add timing in here at some point
  const cv::Mat input = cv::imread("../frames/frame_000001.jpg", 0);

  cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
  std::vector<cv::KeyPoint> key_points;
  f2d->detect(input, key_points);
  cv::Mat output;
  cv::drawKeypoints(input, key_points, output);
  cv::imwrite("sifted.jpg", output);
  */
  return 0;
}




