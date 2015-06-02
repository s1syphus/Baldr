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

#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <time.h>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;



const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 250;
//const float GOOD_PORTION = 0.15f;
const float GOOD_PORTION = 0.75f;


void printKeyPoints(vector<KeyPoint> &, int);
void readInVideo(char*);


struct SURFDetector
{
    Ptr<Feature2D> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};


template<class KPMatcher>

struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};

static Mat drawGoodMatches(
    const Mat& img1,
    const Mat& img2,
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    std::vector<DMatch>& matches,
    std::vector<Point2f>& scene_corners_
    )
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;

    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i] );
    }
    std::cout << "\nMax distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;

    std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

    // drawing the results
    Mat img_matches;

    drawMatches( img1, keypoints1, img2, keypoints2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0);
    obj_corners[1] = Point( img1.cols, 0 );
    obj_corners[2] = Point( img1.cols, img1.rows );
    obj_corners[3] = Point( 0, img1.rows );
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography( obj, scene, RANSAC );
    perspectiveTransform( obj_corners, scene_corners, H);

    scene_corners_ = scene_corners;
/*
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches,
          scene_corners[0] + Point2f( (float)img1.cols, 0), scene_corners[1] + Point2f( (float)img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
          scene_corners[1] + Point2f( (float)img1.cols, 0), scene_corners[2] + Point2f( (float)img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
          scene_corners[2] + Point2f( (float)img1.cols, 0), scene_corners[3] + Point2f( (float)img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
          scene_corners[3] + Point2f( (float)img1.cols, 0), scene_corners[0] + Point2f( (float)img1.cols, 0),
          Scalar( 0, 255, 0), 2, LINE_AA );
          */
    return img_matches;
}
int main(int argc, char** argv){

//  readInVideo(argv[1]);
/*
  Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
  std::vector<KeyPoint> keypoints, keypoints2;
  Mat test = imread("../rawFrames/frame_000001.jpg");
  f2d->detect(test, keypoints);

  printKeyPoints(keypoints,10);
  cout<<"\n\n";
  Mat test2 = imread("../rawFrames/frame_000020.jpg");
  f2d->detect(test2, keypoints2);
  printKeyPoints(keypoints2,10);
*/

  //Going to use SURF instead of SIFT
  Mat image1 = imread("../newRawFrames/frame_000050.jpg");
  Mat image2 = imread("../newRawFrames/frame_000051.jpg");

  int minHess = 400;
  Ptr<Feature2D> f2d = xfeatures2d::SURF::create(minHess);
  vector<KeyPoint> keypoints1, keypoints2;
  vector<DMatch> matches;

  Mat descriptors1, descriptors2;
  
  SURFDetector surf;
  SURFMatcher<BFMatcher> matcher;

    for (int i = 0; i <= LOOP_NUM; i++)
    {
        surf(image1, Mat(), keypoints1, descriptors1);
        surf(image2, Mat(), keypoints2, descriptors2);
        matcher.match(descriptors1, descriptors2, matches);
    }

/*
  cout<<"About to match\n";

  FlannBasedMatcher matcher;    //Nearest neighbor matcher
  vector<DMatch> matches;
*/
  vector<Point2f> corner;

    Mat img_matches = drawGoodMatches(image1, image2, keypoints1, keypoints2, matches, corner);

    //-- Show detected matches

    namedWindow("surf matches", 0);
    imshow("surf matches", img_matches);
    imwrite("output_image.jpg", img_matches);



  return 0;

  }


void printKeyPoints(vector<KeyPoint> &keypoints, int numPoints){
  for(int i = 0; i < numPoints; i++){
    cout  <<keypoints[i].angle<<"\t"<<keypoints[i].class_id<<"\t"<<keypoints[i].octave<<"\t"
          <<keypoints[i].pt.x<<keypoints[i].pt.y<<"\t"<<keypoints[i].response<<"\t"
          <<keypoints[i].size<<endl;
  }
}

void readInVideo(char* input){
  //using SURF
  clock_t start, fromSift;

    VideoCapture cap(input);
    if(!cap.isOpened()){
      std::cout<<"ERROR\n";
      return;
    }
    
    int frameCount = 0;
    Mat frame, output;
    char rawfilename[128], siftedfilename[128];
    Ptr<Feature2D> f2d = SURF::create();
    std::vector<KeyPoint> keypoints;

    while(1){
      start = clock();
      cap >> frame;
      if(frame.empty()){
        break;
      }
      //raw frames
      sprintf(rawfilename, "../newRawFrames/frame_%06d.jpg", frameCount);
      imwrite(rawfilename, frame);
      std::cout<<(float) (clock() - start)/CLOCKS_PER_SEC <<" sec to read frame\t";
      fromSift = clock();
      f2d->detect(frame, keypoints);
      drawKeypoints(frame, keypoints, output);
      sprintf(siftedfilename, "../newSurfFrames/frame_%06d.jpg", frameCount);
      imwrite(siftedfilename, output);
      std::cout<<(float) (clock() - fromSift)/CLOCKS_PER_SEC <<" sec to SURF\t";
      std::cout<<(float) (clock() - start)/CLOCKS_PER_SEC <<" sec total\n";
      frameCount++;
      }
  }
/* 
 * This works but using SIFT
 */

/*
void readInVideo(char* input){
  clock_t start, fromSift;

    VideoCapture cap(input);
    if(!cap.isOpened()){
      std::cout<<"ERROR\n";
      return;
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
  }
*/


