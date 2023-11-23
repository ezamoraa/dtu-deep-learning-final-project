#pragma once

#include <ros/ros.h>
#include <ros/time.h>
#include <fstream>

class TimingUtils
{
public:
  TimingUtils(const std::string &csvFilePath);

  void startTimer();
  void stopTimerAndWrite();

private:
  ros::Time start_time_;
  std::ofstream csv_file_;
};