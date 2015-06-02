
//baldr.cpp
//refactored or something

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

//declarations, should probably move to header files at some point
void readInVideo(char*, bool);
void matchKeyPoints(string, string);

int main(int argc, char** argv){

  //readInVideo(argv[1],true); 

  string image1 = "rawFrames/frame_000000.jpg";
  string image2 = "rawFrames/frame_000005.jpg";

  matchKeyPoints(image1, image2); 


  return 0;
  }

void matchKeyPoints(string image_1, string image_2){
  //this is more of a proof of concept than a needed function.
  //we need to output a vector of corresepondences between all neighboring images
  //in order to compute the rotations

  string outputFile = "matchedOutput.jpg";
  Mat image1 = imread(image_1, CV_LOAD_IMAGE_GRAYSCALE);
  Mat image2 = imread(image_2, CV_LOAD_IMAGE_GRAYSCALE);
  double hessian = 800.0;
  Ptr<Feature2D> surf;
  vector<KeyPoint> keypoints1, keypoints2;
  surf = SURF::create(hessian);
  surf->detect(image1, keypoints1);
  surf->detect(image2, keypoints2);
  Mat desc1, desc2;
  surf->compute(image1, keypoints1, desc1);
  surf->compute(image2, keypoints2, desc2);
  FlannBasedMatcher matcher; 
  vector<DMatch> matches;
  matcher.match(desc1, desc2, matches );
  double max_dist = 0.0, min_dist = 100, dist;
  for(size_t i = 0; i < desc1.rows; i++){
    dist = matches[i].distance;
    if(dist < min_dist) min_dist = dist;
    if(dist > max_dist) max_dist = dist;
  }
  double thresh = 0.02;
  vector<DMatch> goodMatches;
  for(size_t i = 0; i < desc1.rows; i++){
    if(matches[i].distance <= max(2*min_dist, thresh)){
      goodMatches.push_back(matches[i]);
    }
  } 
  Mat img_matches;
  drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  
 // imwrite(outputFile, img_matches);

  vector<Point2f> obj, scene;
  for(size_t i = 0; i < goodMatches.size(); i++){
    obj.push_back(keypoints1[goodMatches[i].queryIdx].pt);
    scene.push_back(keypoints2[goodMatches[i].trainIdx].pt);
  }

  Mat H = findHomography(obj, scene, CV_RANSAC);
  cout<<H<<endl;


/*
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
  obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  line( img_matches, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );

  imshow("Meh", img_matches);
  waitKey(0);
*/
}



void readInVideo(char* input, bool writeImages){
  //using SURF
  clock_t start, fromSurf;

    VideoCapture cap(input);
    if(!cap.isOpened()){
      std::cout<<"ERROR\n";
      return;
    }
    
    int frameCount = 0;
    Mat frame, output;
    char rawfilename[128], surfedfilename[128];
    Ptr<Feature2D> f2d = SURF::create();
    std::vector<KeyPoint> keypoints;

    while(1){
      start = clock();
      cap >> frame;
      if(frame.empty()){
        break;
      }
      //raw frames
      sprintf(rawfilename, "rawFrames/frame_%06d.jpg", frameCount);
      imwrite(rawfilename, frame);
      std::cout<<(float) (clock() - start)/CLOCKS_PER_SEC <<" sec to read frame\t";
      fromSurf = clock();
      f2d->detect(frame, keypoints);
      drawKeypoints(frame, keypoints, output);
      sprintf(surfedfilename, "surfFrames/frame_%06d.jpg", frameCount);
      imwrite(surfedfilename, output);
      std::cout<<(float) (clock() - fromSurf)/CLOCKS_PER_SEC <<" sec to SURF\t";
      std::cout<<(float) (clock() - start)/CLOCKS_PER_SEC <<" sec total\n";
      frameCount++;
      }
  }



