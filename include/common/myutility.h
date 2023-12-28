#ifndef MYUTILITY_H
#define MYUTILITY_H

#include <ros/ros.h>

#include <deque>
#include <iostream>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Geometry>

#include "common/point_type.h"
#include "common/param_server.h"
#include "common/read_params.h"


/******************/
enum class SensorType
{
  OUSTER,
  HESAI
};

struct LidarParam
{
  SensorType sensor;
  int id;
  std::string frame;
  std::string topic;
  int vertical;
  int horizon;
  Eigen::Affine3d extrans;
  Eigen::Affine3d pose_last = Eigen::Affine3d::Identity( );
};


class TimeCount
{
 public:
  void tic( )
  {
    begin = ros::Time::now( ).toSec( );
  }
  void toc( )
  {
    end = ros::Time::now( ).toSec( );
    sum += end - begin;
    count++;
  }
  double averageTime( )
  {
    return sum / double(count);
  }
  double currentTime( )
  {
    return (end - begin) * 1000;
  }
  void reset( )
  {
    begin = 0, end = 0, mean = 0, sum = 0;
    count = 0;
  }

 private:
  double begin = 0, end = 0, mean = 0, sum = 0;
  int count = 0;
};

inline void stamp2ymdhms(std::string &ymdhms, ros::Time stamp = ros::Time::now( ))
{
  char now[64];
  time_t tt;
  struct tm *ttime;
  tt    = stamp.toSec( );
  ttime = localtime(&tt);
  strftime(now, 64, "%Y%m%d%H%M%S", ttime);
  ymdhms = std::string(now);
}

/******************/

#endif