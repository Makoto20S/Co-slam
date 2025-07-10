#include "utility.h"
#include "inc_octree.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    //fusion subscriber
    std::string _fusion_topic;
    ros::Subscriber subFusionTrans;
    ros::Subscriber subBasetoENU;
    double fusionTrans[6]; //x,y,z,roll,pitch,yaw
    double Base_to_ENU[6]; //x,y,z,roll,pitch,yaw

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;
    ros::Publisher pubTransENU;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(robot_id + "/" + lidarFrame, robot_id + "/" + baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(robot_id + "/" + lidarFrame, robot_id + "/" + baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }

        fusionTrans[0] = 0; fusionTrans[1] = 0; fusionTrans[2] = 0; fusionTrans[3] = 0; fusionTrans[4] = 0; fusionTrans[5] = 0;
        Base_to_ENU[0] = 0; Base_to_ENU[1] = 0; Base_to_ENU[2] = 0; Base_to_ENU[3] = 0; Base_to_ENU[4] = 0; Base_to_ENU[5] = 0;
        nh.getParam("/mapfusion/interRobot/sc_topic", _fusion_topic);

        //subBasetoENU     = nh.subscribe<grid_map::inc_octree>("ouster0/lio_sam/mapping/inc_octree", 2000,&TransformFusion::BasetoENUHandler, this, ros::TransportHints().tcpNoDelay());
        subFusionTrans   = nh.subscribe<nav_msgs::Odometry>(robot_id + "/" + _fusion_topic + "/trans_map", 2000,&TransformFusion::FusionTransHandler, this, ros::TransportHints().tcpNoDelay());//发布机器人自身坐标系与基准机器人的基准变换

        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(robot_id + "/lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(robot_id + "/" + odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        //pubTransENU   = nh.advertise<nav_msgs::Odometry>(robot_id + "/" + "trans_ENU", 2000);
        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(robot_id + "/" + odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    (robot_id + "/lio_sam/imu/path", 1);
    }

    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

//    void BasetoENUHandler(const grid_map::inc_octree::ConstPtr& inc_octreeMsg) {
//      if (inc_octreeMsg->robotID == "Base_ENU") {
//        Base_to_ENU[0] = inc_octreeMsg->poseX;
//        Base_to_ENU[1] = inc_octreeMsg->poseY;
//        Base_to_ENU[2] = inc_octreeMsg->poseZ;
//        Base_to_ENU[3] = inc_octreeMsg->poseRoll;
//        Base_to_ENU[4] = inc_octreeMsg->posePitch;
//        Base_to_ENU[5] = inc_octreeMsg->poseYaw;
//        std::cout<<robot_id<<" receive Base_to_ENU successful !"<<std::endl;
//      }
//    }
    void FusionTransHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
        //receive the odometry
        std::cout << robot_id << ": receiving messages..." <<std::endl;
        fusionTrans[0] = odomMsg->pose.pose.position.x;
        fusionTrans[1] = odomMsg->pose.pose.position.y;
        fusionTrans[2] = odomMsg->pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odomMsg->pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(fusionTrans[3], fusionTrans[4], fusionTrans[5]);

//        Eigen::Affine3f rb2base = pcl::getTransformation(fusionTrans[0], fusionTrans[1], fusionTrans[2], fusionTrans[3], fusionTrans[4], fusionTrans[5]);
//        Eigen::Affine3f base2ENU = pcl::getTransformation(Base_to_ENU[0], Base_to_ENU[1], Base_to_ENU[2], Base_to_ENU[3], Base_to_ENU[4], Base_to_ENU[5]);
//        Eigen::Affine3f rb2ENU = base2ENU * rb2base;
//        float rb_to_ENU[6];
//        pcl::getTranslationAndEulerAngles(rb2ENU, rb_to_ENU[0], rb_to_ENU[1], rb_to_ENU[2], rb_to_ENU[3], rb_to_ENU[4], rb_to_ENU[5]);
//        nav_msgs::Odometry Robot2ENU;
//        Robot2ENU.header = odomMsg->header;
//        Robot2ENU.pose.pose.position.x = rb_to_ENU[0];
//        Robot2ENU.pose.pose.position.y = rb_to_ENU[1];
//        Robot2ENU.pose.pose.position.z = rb_to_ENU[2];
//        Robot2ENU.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(rb_to_ENU[3], rb_to_ENU[4],rb_to_ENU[5]);
//        pubTransENU.publish(Robot2ENU);
//        std::cout<<robot_id<<" publish pubTransENU successful !"<<std::endl;
//        std::cout<<robot_id<<" received tf: ";
//        std::cout << fusionTrans[0] << " " << fusionTrans[1] << " " << fusionTrans[2] << " " << fusionTrans[3] << " "
//                  << fusionTrans[4] << " " << fusionTrans[5] << std::endl;


//        //tf----两边同时发布相同的tf会提示警告
//        tf::TransformBroadcaster tfMap2Odom;
//        tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(fusionTrans[3], fusionTrans[4], fusionTrans[5]), tf::Vector3(fusionTrans[0], fusionTrans[1], fusionTrans[2]));
//        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, robot_id + "/" + odometryFrame));//robot_id的基准到基准机器人map基准的变换

    }
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // publish tf 必须要有一个static发布tf，后面tf的sendTransform才可以生效
        static tf::TransformBroadcaster  tfMap2Odom;
        tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(fusionTrans[3], fusionTrans[4], fusionTrans[5]), tf::Vector3(fusionTrans[0], fusionTrans[1], fusionTrans[2]));
        tf::StampedTransform trans_map_to_odom = tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, robot_id + "/" + odometryFrame);
        //tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, robot_id + "/" + odometryFrame));//以imu的频率持续发布当前机器人到基准机器人基准的变换
        tfMap2Odom.sendTransform(trans_map_to_odom);

        tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(odomMsg->pose.pose, tCur);
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, robot_id + "/" + odometryFrame, robot_id + "/" + baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        //should not be static tf, if static, they will not change!
        tf::TransformBroadcaster tfMap2Odom;
        tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(fusionTrans[3], fusionTrans[4], fusionTrans[5]), tf::Vector3(fusionTrans[0], fusionTrans[1], fusionTrans[2]));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, robot_id + "/" + odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, robot_id + "/" + odometryFrame, robot_id + "/" + baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = robot_id + "/" + odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 0.1)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = robot_id + "/" + odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
