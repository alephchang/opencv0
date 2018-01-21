#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if(argc<3){
        cout << "Usage: data_dir vocab_path" <<endl;
        return 1;
    }
    string dataset_dir = argv[1];
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }
    fin.close();

    // read the images and database  
    cout<<"reading database"<<endl;
    DBoW3::Vocabulary vocab(argv[2]);
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want: 
    if ( vocab.empty() )
    {
        cerr<<"Vocabulary does not exist."<<endl;
        return 1;
    }
    cout<<"reading images... "<<endl;
    cout<<"detecting ORB features ... "<<endl;
    cout<<"comparing images with images "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> images; 
    vector<Mat> descriptors;
    for(size_t i = 0; i < rgb_files.size(); ++i){
        images.push_back(imread(rgb_files[i]));
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute( images.back(), Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
        DBoW3::BowVector v1;
        vocab.transform(descriptors.back(), v1);
        for(size_t j = 0; j < i; ++j){
            DBoW3::BowVector v2;
            vocab.transform( descriptors[j], v2 );
            double score = vocab.score(v1, v2);
            cout<<"image "<<i<<" vs image "<<j<<" : "<<score<<endl;
        }
    }
    cout <<endl;
}