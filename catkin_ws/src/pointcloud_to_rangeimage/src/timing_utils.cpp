#include "pointcloud_to_rangeimage/timing_utils.h"

TimingUtils::TimingUtils(const std::string &csvFilePath) : start_time_(ros::Time::now())
{
  // Open the CSV file in append mode
  csv_file_.open(csvFilePath, std::ios::app);

  // Check if the file is opened successfully
  if (!csv_file_.is_open())
  {
    // Handle file open error
    ROS_ERROR("Error opening CSV file for timing information.");
  }
  else if (csv_file_.tellp() == std::ofstream::pos_type(0))
  {
    csv_file_ << "Timestamp(s),Execution time(s))\n";
  }
}

void TimingUtils::startTimer()
{
  start_time_ = ros::Time::now();
}

void TimingUtils::stopTimerAndWrite()
{
  // Get the current timestamp
  std::time_t now = ros::Time::now().toSec();

  // Calculate elapsed time
  ros::Duration elapsed_time = ros::Time::now() - start_time_;

  // Write timestamp and elapsed time to the CSV file
  csv_file_ << now << "," << elapsed_time.toSec() << "\n";
}