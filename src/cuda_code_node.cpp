
#include <iostream>


#include "common/myutility.h"

#include "func/addition.h"
#include "func/matrix_cuda.h"
#include "func/cloud_normal.h"
#include "func/reduction.h"
#include "func/cloud_change.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cuda_code_node");

  //***** 设置日志 *****//
  FLAGS_log_dir          = std::string(getenv("HOME")) + "/bag/perception_test/log";
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr  = true; // shouldbe false
  FLAGS_logbuflevel      = -1;
  // todo: use CLog from common
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "cuda_code node starts";


  int dev = 0;
  cudaDeviceProp devProp;
  printf("cuda device properties: %d\n", cudaGetDeviceProperties(&devProp, dev));
  std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
  std::cout << "SM的数量: " << devProp.multiProcessorCount << std::endl;
  std::cout << "每个线程块的共享内存大小: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
  std::cout << "每个线程块的最大线程数: " << devProp.maxThreadsPerBlock << std::endl;
  std::cout << "每个EM的最大线程数: " << devProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "每个SM的最大线程束数: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
  std::cout << std::endl
            << std::endl;

  NumAdd num_add;

  Matrix mat;
  mat.matrixMul(mat.A, mat.B, mat.C);
  mat.matrixAdd(mat.A, mat.B, mat.C);

  Reduction reduction;

  CloudChange cloud_change;
  ;

  // CloudNormal cloud_normal;

  std::cout << "cuda node finish" << std::endl;

  return 0;
}