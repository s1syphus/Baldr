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
#include<stdio.h>
#include "opencv2/xfeatures2d.hpp"
#include <time.h>
#include <iostream>

using namespace cv;

int main(int argc, char** argv){

  clock_t start, fromSift;

  VideoCapture cap(argv[1]);
  if(!cap.isOpened()){
    std::cout<<"ERROR\n";
    return 1;
  }
  
  int frameCount = 0;
  Mat frame, output;
  char rawfilename[128], siftedfilename[128];
  Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
  std::vector<KeyPoint> keypoints;


  while(1){
    start = clock();
    cap >> frame;
    if(frame.empty()){
      break;
    }
    //raw frames
    sprintf(rawfilename, "../rawFrames/frame_%06d.jpg", frameCount);
    imwrite(rawfilename, frame);
    std::cout<<(float) (clock() - start)/CLOCKS_PER_SEC <<" sec to read frame\t";
    fromSift = clock();
    f2d->detect(frame, keypoints);
    drawKeypoints(frame, keypoints, output);
    sprintf(siftedfilename, "../siftedFrames/frame_%06d.jpg", frameCount);
    imwrite(siftedfilename, output);
    std::cout<<(float) (clock() - fromSift)/CLOCKS_PER_SEC <<" sec to SIFT\t";
    std::cout<<(float) (clock() - start)/CLOCKS_PER_SEC <<" sec total\n";

    frameCount++;
    }



  return 0;

  }


