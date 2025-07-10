#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <msg/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

//new added
#include "msg/cloud_info.h"
pcl::PointCloud<DiscoType>::Ptr extractedCloud;
//pcl::PointCloud<LwType>::Ptr extractedCloud;
#include "msg/context_info.h"

#include "IMU_Processing.hpp"

// gstam
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

#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "GNSS_Processing.hpp"

#include "msg/save_map.h"
#include "msg/rtk_pos_raw.h"
#include "msg/rtk_heading_raw.h"

#include "MapOptimization.hpp"
#include <GeographicLib/UTMUPS.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
int    add_point_size = 0, kdtree_delete_counter = 0;
bool   pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
std::mutex odoLock; 
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

//New added
lio_sam::cloud_info cloudInfo;
std::string robot_id;
string lidarFrame;
string odometryFrame;
double base_time;
float last_pose[6]; 

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0,last_timestamp_leg= -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    scan_count = 0, publish_count = 0, imu_init_time = 8;
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;

bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<nav_msgs::Odometry::ConstPtr> leg_buffer;
std::deque<nav_msgs::Odometry> odomQueue; 

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;

state_ikfom state_point;
state_ikfom new_state_point;

Eigen::Vector3d pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
ros::Publisher  pubLaserCloudInfo; 

ros::ServiceServer srvSaveMap;

float mappingSurfLeafSize;

string leg_topic;
bool useleg;



void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

bool first_lidar = false;
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();

    if(!first_lidar)
        first_lidar = true;

    scan_count ++;
//    double time_now = msg->header.stamp.toSec() - base_time;
//    if (time_now < last_timestamp_lidar)
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    // time_buffer.push_back(msg->header.stamp.toSec());
    // cout<<"lidar time begin:"<<ptr->points.front().curvature<<endl;
    // cout<<"lidar time end:"<<ptr->points.back().curvature<<endl;
    time_buffer.push_back(msg->header.stamp.toSec()-ptr->points.back().curvature/1000.0);
    last_timestamp_lidar = msg->header.stamp.toSec();
//    time_buffer.push_back(time_now-ptr->points.back().curvature/1000.0);//因为msg的时间戳是雷达帧结束时间，所以要转成开始时间.
//    last_timestamp_lidar = time_now;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    if(!first_lidar)
        first_lidar = true;
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}





double timediff_lidar_wrt_imu = 0.0;

void odo_cbk(const nav_msgs::Odometry::ConstPtr& msg_in) { 
  odoLock.lock(); 
  odomQueue.push_back(*msg_in); 
  odoLock.unlock(); 
} 
 
void pts_cbk(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) { 
  static int num = 0; bool is_key_frame = false; 
  double timeScanCur = laserCloudMsg->header.stamp.toSec(); 
  odoLock.lock(); 
  while (!odomQueue.empty()) { 
      if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01) 
          odomQueue.pop_front(); 
      else 
          break; 
  } 
  std::cout<<setprecision(15)<<" time 1 is : "<<odomQueue.front().header.stamp.toSec()<<"; time 2 is : "<<timeScanCur<<std::endl; 
  nav_msgs::Odometry cur_pose; 
  if (odomQueue.size() > 0) { 
    cur_pose = odomQueue.front(); 
  } else { 
    std::cout<<" odomQueue is unmatch  pts_cbk, cannot trans ! "<<std::endl; 
    return; 
  } 
  odomQueue.pop_front(); 
  odoLock.unlock(); 
 
  lio_sam::cloud_info cloudInfo; sensor_msgs::PointCloud2 tempCloud; 
  tempCloud = *laserCloudMsg; 
  tempCloud.header.frame_id = robot_id + "/" + lidarFrame; 
  cloudInfo.header = tempCloud.header; 
  cloudInfo.cloud_deskewed = tempCloud;//----原始点云用于sc 
  cloudInfo.cloud_corner.data.clear(); 
  cloudInfo.cloud_surface.data.clear();//----原始点云用于回环icp 
  tf::Quaternion q1(cur_pose.pose.pose.orientation.x, cur_pose.pose.pose.orientation.y, 
                    cur_pose.pose.pose.orientation.z, cur_pose.pose.pose.orientation.w);//四元数初始化参数顺序为x,y,z,w 
  double roll, pitch, yaw; 
  tf::Matrix3x3(q1).getRPY(roll, pitch, yaw); 
  cloudInfo.initialGuessX = cur_pose.pose.pose.position.x; 
  cloudInfo.initialGuessY = cur_pose.pose.pose.position.y; 
  cloudInfo.initialGuessZ = cur_pose.pose.pose.position.z; 
  cloudInfo.initialGuessRoll  = roll; 
  cloudInfo.initialGuessPitch = pitch; 
  cloudInfo.initialGuessYaw   = yaw; 
  cloudInfo.imuRollInit  = roll; 
  cloudInfo.imuPitchInit = pitch; 
  cloudInfo.imuYawInit   = yaw; 
  cloudInfo.imuAvailable = num; 
 
  if (num == 0) { 
    is_key_frame = true; 
    last_pose[0] = cloudInfo.initialGuessX; last_pose[1] = cloudInfo.initialGuessY; last_pose[2] = cloudInfo.initialGuessZ; 
    last_pose[3] = cloudInfo.initialGuessRoll; last_pose[4] = cloudInfo.initialGuessPitch; last_pose[5] = cloudInfo.initialGuessYaw; 
  } else { 
    if (abs(roll - last_pose[3]) < 0.2 && abs(pitch - last_pose[4]) < 0.2 && abs(yaw - last_pose[5]) < 0.2 && 
        sqrt((cloudInfo.initialGuessX - last_pose[0]) * (cloudInfo.initialGuessY - last_pose[1]) +(cloudInfo.initialGuessY - last_pose[1]) + 
             (cloudInfo.initialGuessZ - last_pose[2]) * (cloudInfo.initialGuessZ - last_pose[2])) < 1.0) { 
      is_key_frame = false; 
    } else { 
      is_key_frame = true; 
      last_pose[0] = cloudInfo.initialGuessX; last_pose[1] = cloudInfo.initialGuessY; last_pose[2] = cloudInfo.initialGuessZ; 
      last_pose[3] = cloudInfo.initialGuessRoll; last_pose[4] = cloudInfo.initialGuessPitch; last_pose[5] = cloudInfo.initialGuessYaw; 
    } 
  } 
 
  if (!is_key_frame) 
    return; 
  pubLaserCloudInfo.publish(cloudInfo); 
  num++; 
  std::cout<< robot_id<<" pub " <<num<< " cloudInfo !!!"<<std::endl; 
} 

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

//    static int i =0;
//    if (i==0)
//       base_time = msg_in->header.stamp.toSec() - 1000;
//    i++;
//    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec()-base_time);
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
        imu_rtk_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);//imu数据
    imu_rtk_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void leg_cbk(const nav_msgs::Odometry::ConstPtr &msg_in)
{
    nav_msgs::Odometry::Ptr msg(new nav_msgs::Odometry(*msg_in));
    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_leg)
    {
        ROS_WARN("leg loop back, clear buffer");
        leg_buffer.clear();
    }

    last_timestamp_leg = timestamp;

    leg_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
/* last_time_packed是否初始化 */
bool initialized_last_time_packed = false;
double last_time_packed=-1;
/* 数据打包的时间间隔,如果以某个传感器为打包断点,那么时间间隔就是传感器频率的倒数,unit:s */
double time_interval_packed = 0.1;//0.1
/* lidar数据异常(没有收到数据)情况下的最大等待时间,对于低频传感器通常设置为帧间隔时间的一半,
* 超过该时间数据还没到来,就认为本次打包该传感器数据异常,unit:s */
double time_wait_max_lidar = 0.05;

bool sync_packages(MeasureGroup &meas) //依次同步 LiDAR、IMU 和腿部传感器数据，确保时间对齐，并打包到 MeasureGroup 中
{

    if (lidar_buffer.empty() || imu_buffer.empty()) { //LiDAR 或 IMU 数据缓冲区为空
        return false;
    }

    /*** push a lidar scan ***/
    if(!first_lidar) //检查是否接收到第一帧 LiDAR 数据
    {
        return false;
    }

    if(!lidar_pushed) //当前 LiDAR 数据尚未被处理（lidar_pushed == false），则取出缓冲区中的第一帧数据
    {
        if(!lidar_buffer.empty())
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();

            last_lidar_end_time = lidar_end_time; //开始记录时间

            if (meas.lidar->points.size() <= 5) // time too little
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }

            meas.lidar_end_time = lidar_end_time;
            meas.lidar_vaild = true;
            lidar_pushed = true;
            meas.package_end_time = lidar_end_time;
        }
        
    }
    /* 如果lidar还没取出 */
    if (!lidar_pushed)
    return false;

    // if (last_timestamp_imu < lidar_end_time)
    // {
    //     return false;
    // }
    
    if (last_timestamp_imu <= meas.package_end_time)
        return false;

    if(useleg)
    {
        if (last_timestamp_leg <= meas.package_end_time)
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    // while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    while ((!imu_buffer.empty()) && (imu_time < meas.package_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        // if(imu_time > lidar_end_time) break;
        if(imu_time > meas.package_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    /*** push leg data, and pop from leg buffer ***/
    if(!leg_buffer.empty() && useleg)
    {
        double leg_time = leg_buffer.front()->header.stamp.toSec();
        meas.leg.clear();
        while ((!leg_buffer.empty()) && (leg_time < lidar_end_time))
        {
            leg_time = leg_buffer.front()->header.stamp.toSec();
            if(leg_time > lidar_end_time) break;
            meas.leg.push_back(leg_buffer.front());
            leg_buffer.pop_front();
        }
    }
    
    if(!meas.leg.empty())
    {
        meas.leg_vaild = true;
    }
    else
    {
        meas.leg_vaild = false;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;

    last_time_packed = meas.package_end_time;
    return true;
}


void newpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(new_state_point.rot.matrix() * p_body + new_state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix()*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) //局部坐标系到中间坐标系,中间坐标系到全局坐标系
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix()*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}


BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

//lasermap_fov_segment
//如果Localmap没有初始化，根据lidar中心坐标计算locamap的范围，初始化成功，直接return。
//检测当前lidar中心点距离边界的距离，当离边界小于一定阈值后，认为需要移动地图中心。不需移动，直接return。
//按一定策略移动整个地图的对角点，并将移出的区域传入ikdtree，进行地图点的删除。
void lasermap_fov_segment()
{
    cub_needrm.clear();     // 清空需要移除的区域
    kdtree_delete_counter = 0;

    V3D pos_LiD = pos_lid;  // W系下位置
    //初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    //各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);//当前位置距离最小点距离在i轴上距离
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);//当前位置距离最大点距离在i轴上距离
        // if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
        if (dist_to_map_edge[i][0] <= 60 || dist_to_map_edge[i][1] <= 60) need_move = true; //todo
    }
    if (!need_move) return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;

    // float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    float mov_dist = 20.0;
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        // if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
        if (dist_to_map_edge[i][0] <= 20){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);//存储要截去的体积
        // } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
        } else if (dist_to_map_edge[i][1] <= 20){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm); //删除指定范围内的点
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix()*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));//bdodm-body
    
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point;   //点所在体素的中心
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[j], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull_)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
//            pointBodyToWorld(&laserCloudFullRes->points[i], \
//                                &laserCloudWorld->points[i]);
            newpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
//        laserCloudmsg.header.frame_id = "camera_init";
        laserCloudmsg.header.frame_id = robot_id + "/" + odometryFrame;
        pubLaserCloudFull_.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);

    auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
    out.pose.orientation.x = q_.coeffs()[0];
    out.pose.orientation.y = q_.coeffs()[1];
    out.pose.orientation.z = q_.coeffs()[2];
    out.pose.orientation.w = q_.coeffs()[3];   
}

template<typename T>
void set_new_posestamp(T & out)
{
    out.pose.position.x = new_state_point.pos(0);
    out.pose.position.y = new_state_point.pos(1);
    out.pose.position.z = new_state_point.pos(2);

    auto q_ = Eigen::Quaterniond(new_state_point.rot.matrix());
    out.pose.orientation.x = q_.coeffs()[0];
    out.pose.orientation.y = q_.coeffs()[1];
    out.pose.orientation.z = q_.coeffs()[2];
    out.pose.orientation.w = q_.coeffs()[3];
}

void change_to_ldiar(state_ikfom &st_point){
    Eigen::Quaterniond q_ = Eigen::Quaterniond(st_point.offset_R_L_I.matrix());
    gtsam::Pose3 bodytolidar = gtsam::Pose3(gtsam::Rot3::Quaternion(q_.w(), q_.x(), q_.y(), q_.z()), gtsam::Point3(st_point.offset_T_L_I.x(),
                                            st_point.offset_T_L_I.y(), st_point.offset_T_L_I.z()));
    Eigen::Quaterniond q_1 = Eigen::Quaterniond(st_point.rot.matrix());
    gtsam::Pose3 bodmtobody = gtsam::Pose3(gtsam::Rot3::Quaternion(q_1.w(), q_1.x(), q_1.y(), q_1.z()),
                                           gtsam::Point3(st_point.pos(0), st_point.pos(1), st_point.pos(2)));
    gtsam::Pose3 bodmtoldiar = bodmtobody.compose(bodytolidar);
    gtsam::Pose3 lidmtoldiar = bodytolidar.between(bodmtoldiar);//将原点建图由imu系转换到lidar系
    Eigen::Vector3d pose_t(lidmtoldiar.translation().x(), lidmtoldiar.translation().y(), lidmtoldiar.translation().z());
    st_point.pos = pose_t;
    Eigen::Matrix3d pose_r = lidmtoldiar.rotation().matrix();
    st_point.rot = Sophus::SO3d(pose_r);
}

void publish_fuse_odometry(const ros::Publisher & pubLaserOdometryGlobal, const ros::Publisher &pubLaserCloudInfo, lio_sam::cloud_info &cloudIn) {
    // Publish odometry for ROS (global)
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);//ldodm-lidar
    odomAftMapped.header.frame_id = robot_id + "/" + odometryFrame;
    odomAftMapped.child_frame_id = robot_id + "/" + lidarFrame + "/odom_mapping";
    set_new_posestamp(odomAftMapped.pose);
    pubLaserOdometryGlobal.publish(odomAftMapped);

    if (is_key_frame == false)
        return;

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    tf::Quaternion q1(odomAftMapped.pose.pose.orientation.x, odomAftMapped.pose.pose.orientation.y,
                      odomAftMapped.pose.pose.orientation.z, odomAftMapped.pose.pose.orientation.w);//四元数初始化参数顺序为x,y,z,w
    double roll, pitch, yaw;
    tf::Matrix3x3(q1).getRPY(roll, pitch, yaw);
    cloudIn.initialGuessX = odomAftMapped.pose.pose.position.x;//fast-lio-sam优化出的位姿
    cloudIn.initialGuessY = odomAftMapped.pose.pose.position.y;
    cloudIn.initialGuessZ = odomAftMapped.pose.pose.position.z;
    cloudIn.initialGuessRoll  = roll;
    cloudIn.initialGuessPitch = pitch;
    cloudIn.initialGuessYaw   = yaw;
    cloudIn.imuRollInit  = roll;
    cloudIn.imuPitchInit = pitch;
    cloudIn.imuYawInit   = yaw;
    cloudInfo.imuAvailable = cloudKeyPoses6D->size() - 1;
    pubLaserCloudInfo.publish(cloudIn);

    // Publish TF
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, robot_id + "/" + odometryFrame, robot_id + "/" + lidarFrame + "/lidar_link") );
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_new_posestamp(msg_body_pose);//ldodm-lidar
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
//    msg_body_pose.header.frame_id = "camera_init";
    msg_body_pose.header.frame_id = robot_id + "/" + odometryFrame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.header.stamp = ros::Time().fromSec(lidar_end_time);
        path.header.frame_id = robot_id + "/" + odometryFrame;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

//发布更新后的轨迹
void publish_path_update(const ros::Publisher pubPath)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
//    string odometryFrame = "camera_init";
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int kkk = 0;
        kkk++;
        if (kkk % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            globalPath.header.frame_id = robot_id + "/" + odometryFrame;
            for(int i = 0; i < globalPath.poses.size(); i++) {
                globalPath.poses[i].header.frame_id = robot_id + "/" + odometryFrame;
            }
            pubPath.publish(globalPath);
        }
    }
}

//  发布gnss 轨迹
void publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(lidar_end_time);
    gps_path.header.frame_id = "camera_init";

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        pubPath.publish(gps_path);
    }
}

bool saveMapService(na_mapping::save_mapRequest& req, na_mapping::save_mapResponse& res)
{
    cout<<"start save map"<<endl;

    pcl::PointCloud<PointType>::Ptr MapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr MapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr MapKeyFramesDS(new pcl::PointCloud<PointType>());

    pcl::VoxelGrid<PointType> downSizeFilterMapKeyPoses;
    downSizeFilterMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity);
    downSizeFilterMapKeyPoses.setInputCloud(cloudKeyPoses3D);
    downSizeFilterMapKeyPoses.filter(*MapKeyPosesDS);

    for (int i = 0; i < (int)MapKeyPosesDS->size(); ++i)
    {
        int thisKeyInd = (int)MapKeyPosesDS->points[i].intensity;
        *MapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd],state_point); //  fast_lio only use  surfCloud
    }

    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    // downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    // downSizeFilterGlobalMapKeyFrames.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterGlobalMapKeyFrames.setInputCloud(MapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*MapKeyFramesDS);

    pcl::io::savePCDFileBinary(savemappath,*MapKeyFramesDS);

    if (usertk==true)
    {
        FILE *fp;
        fp=fopen(saveposepath.c_str(),"w+");
        fprintf(fp,"%2.10lf %3.10lf %2.5lf",lat0,lon0,alt0);
        fclose(fp); 
    }

    cout<<"save map done"<<endl;
    res.success = true;
    return true;
}

void gnss_cbk(const na_mapping::rtk_pos_raw::ConstPtr& msg_in)
{
    //判断是否使用rtk融合
    if (usertk==false)
        return;
    //判断gps是否有效
    if (msg_in->pos_type!=50)
        return;

    if (msg_in->hgt_std_dev > 0.05)
        return;

    int zone;
    bool northp;

    //初始化utm坐标系原点
    if(!rtk_p0_init)
    {
        lat0 = msg_in->lat;
        lon0 = msg_in->lon;
        alt0 = msg_in->hgt;   

        GeographicLib::UTMUPS::Forward(lat0, lon0, zone, northp, utm_x0, utm_y0);     
        utm_z0 = alt0;
        rtk_p0_init = true;
    }

    //  ROS_INFO("GNSS DATA IN ");
    double timestamp = msg_in->header.stamp.toSec()+rtk_time_grift;

    mtx_buffer.lock();

    // 没有进行时间纠正
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec()+rtk_time_grift;
    gnss_data.status = msg_in->pos_type;
    gnss_data.service = 0;

    //设置rtk协方差
    double posecov;
    posecov=0.05*0.05;

    gnss_data.pose_cov[0] = posecov;
    gnss_data.pose_cov[1] = posecov;
    gnss_data.pose_cov[2] = 4.0*posecov;

    mtx_buffer.unlock();
    
    double utm_x,utm_y,utm_z;
    GeographicLib::UTMUPS::Forward(msg_in->lat, msg_in->lon, zone, northp, utm_x, utm_y);
    utm_z = msg_in->hgt;
    utm_x -= utm_x0;
    utm_y -= utm_y0;
    utm_z -= utm_z0;

    nav_msgs::Odometry gnss_data_enu ;
    gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
    gnss_data_enu.pose.pose.position.x =  utm_x ;  //东
    gnss_data_enu.pose.pose.position.y =  utm_y ;  //北
    gnss_data_enu.pose.pose.position.z =  utm_z ;  //天

    gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
    gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
    gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

    gnss_buffer.push_back(gnss_data_enu);
    // visial gnss path in rviz:
    msg_gnss_pose.header.frame_id = "camera_init";
    msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);

    Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

    gnss_pose(0,3) = utm_x ;
    gnss_pose(1,3) = utm_y ;
    gnss_pose(2,3) = utm_z ;

    msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
    msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
    msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

    gps_path.poses.push_back(msg_gnss_pose); 
    rtk_time.push_back(msg_in->header.stamp.toSec());
}
// void gnss_cbk(const na_mapping::rtk_pos_raw::ConstPtr& msg_in)
// {
//     //判断是否使用rtk融合
//     if (usertk==false)
//         return;
//     //判断gps是否有效
//     if (msg_in->pos_type!=50)
//         return;

//     if (msg_in->hgt_std_dev > 0.05)
//         return;

//     //初始化局部东北天坐标系原点
//     if(use_rtk_heading && !rtk_p0_init)
//     {
//         if(rtk_heading_vaild)
//         {
//             lat0 = msg_in->lat;
//             lon0 = msg_in->lon;
//             alt0 = msg_in->hgt;
//             gnss_data.InitOriginPosition(lat0, lon0, alt0) ; 
            
//             //计算初始雷达的位置（经纬高）
//             // Eigen::Vector3d pose_Lidar_t0 = Vector3d::Zero();
//             // Eigen::AngleAxisd rot_z_btol(rtk_heading, Eigen::Vector3d::UnitZ());
//             // Eigen::Matrix3d dog_attitude = rot_z_btol.matrix();
//             // pose_Lidar_t0 = - dog_attitude * rtk_T_wrt_Lidar;
//             // gnss_data.InitOriginPosition(msg_in->lat, msg_in->lon, msg_in->hgt) ;
//             // gnss_data.Reverse(pose_Lidar_t0[0],pose_Lidar_t0[1],pose_Lidar_t0[2],lat0,lon0,alt0);
//             // gnss_data.InitOriginPosition(lat0, lon0, alt0) ; 

//             rtk_p0_init = true;
//         }
//         else
//         {
//             return;
//         }
//     }
//     //  ROS_INFO("GNSS DATA IN ");
//     double timestamp = msg_in->header.stamp.toSec()+rtk_time_grift;

//     mtx_buffer.lock();

//     // 没有进行时间纠正
//     if (timestamp < last_timestamp_gnss)
//     {
//         ROS_WARN("gnss loop back, clear buffer");
//         gnss_buffer.clear();
//     }

//     last_timestamp_gnss = timestamp;

//     // convert ROS NavSatFix to GeographicLib compatible GNSS message:
//     gnss_data.time = msg_in->header.stamp.toSec()+rtk_time_grift;
//     gnss_data.status = msg_in->pos_type;
//     gnss_data.service = 0;

//     //通过gps_qual给出协方差
//     double posecov;
//     posecov=0.05*0.05;
//     // posecov=0.01*0.01;
//     // if (gnss_data.status == 4)
//     //     posecov=0.05*0.05;
//     // else if (gnss_data.status == 5)
//     //     posecov=1.0*1.0;
//     // else if (gnss_data.status == 1)
//     //     posecov=10.0*10.0;
//     // else
//     //     return;

//     gnss_data.pose_cov[0] = posecov;
//     gnss_data.pose_cov[1] = posecov;
//     gnss_data.pose_cov[2] = 2.0*posecov;

//     mtx_buffer.unlock();
   
//     if(!gnss_inited){           //  初始化位置
//         // gnss_data.InitOriginPosition(msg_in->lat, msg_in->lon, msg_in->hgt) ; 
//         // lat0=msg_in->lat;
//         // lon0=msg_in->lon;
//         // alt0=msg_in->hgt;
//         gnss_inited = true ;
//     }else{                               
//         //经纬高转东北天
//         gnss_data.UpdateXYZ(msg_in->lat, msg_in->lon, msg_in->hgt) ;             
//         nav_msgs::Odometry gnss_data_enu ;
//         // add new message to buffer:
//         gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);

//         // V3D dp;
//         // dp = state_point.rot.matrix()*rtk_T_wrt_Lidar;

//         gnss_data_enu.pose.pose.position.x =  gnss_data.local_E ;  //东
//         gnss_data_enu.pose.pose.position.y =  gnss_data.local_N ;  //北
//         gnss_data_enu.pose.pose.position.z =  gnss_data.local_U ;  //天

//         // gnss_data_enu.pose.pose.orientation.x =  geoQuat.x ;                //  gnss 的姿态不可观，所以姿态只用于可视化，取自imu
//         // gnss_data_enu.pose.pose.orientation.y =  geoQuat.y;
//         // gnss_data_enu.pose.pose.orientation.z =  geoQuat.z;
//         // gnss_data_enu.pose.pose.orientation.w =  geoQuat.w;

//         gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
//         gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
//         gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

//         gnss_buffer.push_back(gnss_data_enu);

//         // visial gnss path in rviz:
//         msg_gnss_pose.header.frame_id = "camera_init";
//         msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);
//         // Eigen::Vector3d gnss_pose_ (gnss_data.local_E, gnss_data.local_N, - gnss_data.local_U); 
//         // Eigen::Vector3d gnss_pose_ (gnss_data.local_N, gnss_data.local_E, - gnss_data.local_U); 
//         Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

//         // V3D dp;
//         // dp=state_point.rot.matrix()*rtk_T_wrt_Lidar;

//         gnss_pose(0,3) = gnss_data.local_E ;
//         gnss_pose(1,3) = gnss_data.local_N ;
//         gnss_pose(2,3) = gnss_data.local_U ;

//         // gnss_pose(0,3) = gnss_data.local_E - dp[0];
//         // gnss_pose(1,3) = gnss_data.local_N - dp[1];
//         // gnss_pose(2,3) = gnss_data.local_U - dp[2];

//         // Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
//         // gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
//         // gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下

//         msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
//         msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
//         msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

//         gps_path.poses.push_back(msg_gnss_pose); 
//     }
// }

void gnss_heading_cbk(const na_mapping::rtk_heading_raw::ConstPtr& msg_in)
{
    if(msg_in->pos_type != 50 || use_rtk_heading == false)
    {
        return ;
    }
    
    // rtk_heading = (180.0 - msg_in->heading) * M_PI / 180.0;
    rtk_heading =  - msg_in->heading * M_PI / 180.0;
    rtk_heading_vaild = true;
}

int robotID2Number(std::string robo){
    return robo.back() - '0';
}

void contextLoopInfoHandler(const lio_sam::context_infoConstPtr& msgIn){//虚拟回环约束局部优化
    //close global loop by do nothing
//        return;
    if(msgIn->robotID != robot_id)
        return;

//    int indexFrom = msgIn->numRing;
//    int indexTo = msgIn->numSector;

//    gtsam::Pose3 poseBetween = gtsam::Pose3( gtsam::Rot3::RzRyRx(msgIn->poseRoll, msgIn->posePitch, msgIn->poseYaw),
//                                     gtsam::Point3(msgIn->poseX, msgIn->poseY, msgIn->poseZ) );
//    float noiseScore = msgIn->poseIntensity;
//    gtsam::Vector Vector6(6);
//    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
//        noiseScore;
//    auto noiseBetween = gtsam::noiseModel::Diagonal::Variances(Vector6);

//    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));//添加利用不同机器人之间的基准变换得到的当前机器人两帧之间虚拟约束
//    isam->update(gtSAMgraph);
//    isam->update();
//    isam->update();
//    isam->update();
//    isam->update();
//    isam->update();
//    isamCurrentEstimate = isam->calculateEstimate();

//    aLoopIsClosed = true;
//    correctPoses(state_point, ikdtree);

      saveVirtualLoop(msgIn);
}

//todo read 主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    ros::NodeHandle n("~");

    //new added for multi-robot
    n.param<std::string>("robot_id", robot_id, "jackal0");
    nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
    nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);                // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);              // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic 
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);     // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    n.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);                        // 卡尔曼滤波的最大迭代次数
    nh.param<string>("map_file_path",map_file_path,"");                         // 地图保存路径
    nh.param<string>("lio_sam/pointCloudTopic",lid_topic,"/livox/lidar");              // 雷达点云topic名称
    nh.param<string>("lio_sam/imuTopic", imu_topic,"/livox/imu");               // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);                 // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);          // VoxelGrid降采样时的体素大小
    n.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    n.param<double>("filter_size_map",filter_size_map_min,0.5);
    n.param<double>("cube_side_length",cube_len,200);                          // 地图的局部区域的长度（FastLio2论文中有解释）
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);                       // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    n.param<double>("gyr_cov",gyr_cov,0.1);                            // IMU陀螺仪的协方差
    n.param<double>("acc_cov",acc_cov,0.1);                            // IMU加速度计的协方差
    n.param<int>("imu_init_time",imu_init_time,8);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);                     // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);                     // IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);                   // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("lio_sam/sensor", p_pre->lidar_type, AVIA);            // 激光雷达的类型
    nh.param<int>("lio_sam/N_SCAN", p_pre->N_SCANS, 16);                  // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    n.param<int>("point_filter_num", p_pre->point_filter_num, 2);              // 采样间隔，即每隔point_filter_num个点取1个点
    n.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);    // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("lio_sam/extrinsicTrans", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("lio_sam/extrinsicRot", extrinR, vector<double>()); // 雷达相对于IMU的外参R
    

    // save keyframes
    nh.param<float>("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0); //判断是否为关键帧的距离阈值(m)
    nh.param<float>("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2); //判断是否为关键帧的角度阈值(rad)
    // Visualization
    nh.param<float>("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 50); //重构ikd树的搜索范围(m)
    nh.param<float>("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 2.0);  //重构ikd树对关键帧位置的降采样体素大小
    nh.param<float>("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 0.4);        //重构ikd树的降采样体素大小
    // loop clousre
    n.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false); //是否加入回环因子
    nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);   //回环检测的频率
    nh.param<float>("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 20.0); //回环检测的搜索范围(m)
    nh.param<float>("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0); //回环检测的时间阈值(s)
    nh.param<float>("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3); //回环检测icp匹配的分数阈值
    nh.param<int>("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);    //回环检测局部地图的使用的相邻关键帧数量
    nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);           //回环检测局部地图的降采样体素大小
    //rtk
    nh.param<bool>("usertk", usertk, false); //是否加入gps因子
    nh.param<bool>("use_rtk_heading", use_rtk_heading, false); //是否使用rtk航向初始化
    nh.param<string>("common/gnss_topic", gnss_topic,"/rtk_pos_raw");   //gps的topic名称
    nh.param<string>("common/gnss_heading_topic", gnss_heading_topic,"/rtk_heading_raw");   //gps的topic名称
    nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 0.2);           //gps的协方差阈值
    nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 0.01);        //位姿的协方差阈值，过小就不用加入gps因子
    nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, true);           //是否使用gps的高度信息
    nh.param<vector<double>>("mapping/rtk2Lidar_T", rtk2Lidar_T, vector<double>()); // rtk相对于雷达的外参T（即rtk在Lidar坐标系中的坐标）

    nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);                   //使用的cpu核数
    nh.param<bool>("recontructKdTree", recontructKdTree, true);         //是否重构ikd树
    nh.param<std::string>("savemappath", savemappath, "/home/ywb/s-fast-lio/src/S-FAST_LIO/PCD/cloud_map.pcd"); //保存地图点云的路径
    nh.param<std::string>("saveposepath", saveposepath, "/home/ywb/s-fast-lio/src/S-FAST_LIO/PCD/pose.txt");    //保存地图原点的位置（只有用gps的时候会保存）
  
    nh.param<string>("common/leg_topic", leg_topic,"/leg_odom");   //leg的topic名称
    nh.param<bool>("useleg",useleg,false); //是否使用leg
//    cout<<"Lidar_type: "<<p_pre->lidar_type<<endl;
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp    = ros::Time::now();
/*    path.header.frame_id ="camera_init";*/
    path.header.frame_id = robot_id + "/" + odometryFrame;
    // ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    /*** ROS subscribe initialization ***/
//    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
//        nh.subscribe(robot_id + "/" + lid_topic, 200000, livox_pcl_cbk) : \
//        nh.subscribe(robot_id + "/" + lid_topic, 200000, standard_pcl_cbk);
//    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
//        nh.subscribe("/ouster/points", 200000, livox_pcl_cbk) : \
//        nh.subscribe("/ouster/points", 200000, standard_pcl_cbk);
//    ros::Subscriber sub_imu = nh.subscribe(robot_id + "/" + imu_topic, 200000, imu_cbk,ros::TransportHints().unreliable());
//    ros::Subscriber sub_imu = nh.subscribe("/ouster/imu", 200000, imu_cbk,ros::TransportHints().unreliable());
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk); //gnss
    ros::Subscriber sub_gnss_heading = nh.subscribe(gnss_heading_topic, 200000, gnss_heading_cbk); 
    ros::Subscriber sub_leg = nh.subscribe(leg_topic, 200000, leg_cbk,ros::TransportHints().unreliable()); 

    ros::Subscriber sub_fastlio_pose = nh.subscribe("/Odometry", 200000, odo_cbk,ros::TransportHints().unreliable()); 
    ros::Subscriber sub_pts = nh.subscribe("/cloud_registered_body", 200000, pts_cbk,ros::TransportHints().unreliable()); 

    //new added
//    ros::Subscriber subLoop = nh.subscribe(robot_id + "/context/loop_info", 100, contextLoopInfoHandler, ros::TransportHints().tcpNoDelay());
    ros::Publisher  pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(robot_id + "/lio_sam/mapping/map_global", 1000);
    ros::Publisher  pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry> (robot_id + "/lio_sam/mapping/odometry", 1000);
    pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> (robot_id + "/lio_sam/mapping/cloud_info", 1000);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>(robot_id + "/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> (robot_id + "/path", 100000);
    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>(robot_id + "/s_fast_lio/path_update", 100000);                   //  isam更新后的path
    ros::Publisher pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);

    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);

    srvSaveMap  = nh.advertiseService("/save_map" ,  &saveMapService); // 保存地图服务

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);


    shared_ptr<ImuProcess> p_imu1(new ImuProcess());
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    rtk_T_wrt_Lidar<<VEC_FROM_ARRAY(rtk2Lidar_T);
    p_imu1->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov), 
                        V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov), imu_init_time);

    // 回环检测线程
    std::thread loopthread(loopClosureThread, &lidar_end_time, &state_point);

    Eigen::Matrix3d Sigma_leg = Eigen::Matrix3d::Identity(); //leg里程计的协方差
    Sigma_leg(0, 0) = 0.01;
    Sigma_leg(1, 1) = 0.01;
    Sigma_leg(2, 2) = 0.01;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);

    while (ros::ok())
    {
        if (flg_exit) break;
        ros::spinOnce();

        if(sync_packages(Measures)) //通过sync_packages(Measures)这个函数，：依次同步 LiDAR、IMU 和腿部传感器数据，确保时间对齐，并打包到MeasureGroup中
        {   
            if(use_rtk_heading && !rtk_heading_init)
            {
                if(rtk_heading_vaild)
                {
                    //姿态初始化
                    Eigen::AngleAxisd rot_z_btol(rtk_heading, Eigen::Vector3d::UnitZ());
                    Eigen::Matrix3d dog_attitude = rot_z_btol.matrix();
                    state_point.rot=Sophus::SO3d(dog_attitude);
                    
                    //位置初始化
                    Eigen::Vector3d pose_Lidar_t0 = Vector3d::Zero();
                    pose_Lidar_t0 = - dog_attitude * rtk_T_wrt_Lidar;
                    state_point.pos = pose_Lidar_t0;

                    rtk_heading_init = true;
                    kf.change_x(state_point);
                }
                else
                {
                    rate.sleep();
                    continue;
                }
            }

            if (flg_first_scan)  //首次激光雷达扫描的处理,检查是否是第一次扫描
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu1->first_lidar_time = first_lidar_time;
                flg_first_scan = false; //表示已经处理过第一次扫描
                continue;
            }

            state_point_last = state_point;

            p_imu1->Process(Measures, kf, feats_undistort);  //去畸变

            //new added
            extractedCloud.reset(new pcl::PointCloud<DiscoType>()); //**创建并清空点云容器
            extractedCloud->clear();
            for (int i = 0; i <feats_undistort->points.size(); ++i) {
              DiscoType newPoint;
              newPoint.x = feats_undistort->points.at(i).x; //at(i) 是一种访问点云中元素的方法，
              newPoint.y = feats_undistort->points.at(i).y; //它比直接使用索引（points[i]）更安全，
              newPoint.z = feats_undistort->points.at(i).z; //因为它会在索引超出范围时抛出异常。
              newPoint.intensity = feats_undistort->points.at(i).intensity;
              extractedCloud->points.push_back(newPoint);
            }
//            extractedCloud.reset(new pcl::PointCloud<LwType>());
//            extractedCloud->clear();
            sensor_msgs::PointCloud2 tempCloud; //ROS中的标准消息类型，用于传输点云数据
            pcl::toROSMsg(*extractedCloud, tempCloud); //类型转换，将PCL转换成ROS消息类型（该类型ROS可处理点云数据）
            tempCloud.header.stamp.fromSec(lidar_end_time); //转换为ROS time
            tempCloud.header.frame_id = robot_id + "/" + lidarFrame; //设置id
            cloudInfo.header = tempCloud.header;
            cloudInfo.cloud_deskewed = tempCloud;//----原始点云用于sc
            cloudInfo.cloud_corner.data.clear();
            cloudInfo.cloud_surface.data.clear();//----原始点云用于回环icp

            //如果feats_undistort为空 ROS_WARN
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            state_point = kf.get_x(); //根据KF最新的IMU状态和杆臂，更新lidar中心点坐标，pos_lid
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;//雷达在地图坐标系下的位置

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            lasermap_fov_segment();     //更新localmap边界，然后降采样当前帧点云

            //点云下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();

            //new added----原始点云降采样点用于回环icp
//            extractedCloud.reset(new pcl::PointCloud<DiscoType>());
//            extractedCloud->clear();
//            for (int i = 0; i <feats_down_body->points.size(); ++i) {
//                DiscoType newPoint;
//                newPoint.x = feats_down_body->points.at(i).x;
//                newPoint.y = feats_down_body->points.at(i).y;
//                newPoint.z = feats_down_body->points.at(i).z;
//                newPoint.intensity = feats_down_body->points.at(i).intensity;
//                extractedCloud->points.push_back(newPoint);
//            }
//            extractedCloud.reset(new pcl::PointCloud<LwType>());
//            extractedCloud->clear();
//            for (int i = 0; i <feats_down_body->points.size(); ++i) {
//                LwType newPoint;
//                if (abs(feats_down_body->points.at(i).x)<30.0 && abs(feats_down_body->points.at(i).y)<30.0 && abs(feats_down_body->points.at(i).z)<30.0) {
//                newPoint.x = feats_down_body->points.at(i).x * 1000;
//                newPoint.y = feats_down_body->points.at(i).y * 1000;
//                newPoint.z = feats_down_body->points.at(i).z * 1000;
//                extractedCloud->points.push_back(newPoint);
//                }
//            }
//            sensor_msgs::PointCloud2 featureCloud;
//            pcl::toROSMsg(*extractedCloud, featureCloud);
//            featureCloud.header.stamp.fromSec(lidar_end_time);
//            featureCloud.header.frame_id = robot_id + "/" + lidarFrame;
//            cloudInfo.cloud_deskewed = featureCloud;
//            std::cout<<" ouster size is: "<<extractedCloud->points.size()<<std::endl;
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            if(ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for(int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
                continue;
            }


            if(0)
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            Eigen::Vector3d z_leg = Eigen::Vector3d::Zero();
            if(Measures.leg_vaild == true)
            {
                nav_msgs::Odometry::ConstPtr leg_back = Measures.leg.back();
                //z_leg只有前向和左向的速度
                z_leg(0)=leg_back->twist.twist.linear.x;
                z_leg(1)=leg_back->twist.twist.linear.y;
            }
            
            /*** iterated state estimation ***/
            Nearest_Points.resize(feats_down_size);
            // kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
            //迭代量测更新KF
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en,
                                                  Sigma_leg,z_leg,useleg,Measures.leg_vaild,Measures.lidar_vaild);//kf.update_iterated_dyn_share_modified 是 esekf 类中的一个方法，它用于更新卡尔曼滤波器（或其他类似的估计方法）中的状态
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            //new added
            new_state_point = state_point;
            change_to_ldiar(new_state_point);//后端融合更新出的位姿是lidar到lidar原点的位姿

            /***后端融合***/
            getCurPose(new_state_point);//transformTobeMapped是lidar到lidar原点的位姿
            /*back end*/
            saveKeyFramesAndFactor(lidar_end_time,kf,state_point,new_state_point,feats_down_body,feats_undistort);
            correctPoses(state_point, ikdtree);

            /******* Publish odometry *******/
            //publish_odometry(pubOdomAftMapped);
            publish_fuse_odometry(pubLaserOdometryGlobal, pubLaserCloudInfo, cloudInfo);

            /*** add the feature points to map kdtree ***/
            feats_down_world->resize(feats_down_size);

            map_incremental();
            // cout<<"feats_down_size:"<<feats_down_size<<endl;
            // cout<<"ikdtree size:"<<ikdtree.validnum()<<endl;
            // cout<<"add points time:"<<time_add_map2 - time_add_map1<<endl;
            
            /******* Publish points *******/
            if (path_en)
            {
                publish_path(pubPath);
                publish_path_update(pubPathUpdate);             //   发布经过isam2优化后的路径
                publish_gnss_path(pubGnssPath);                        //   发布gnss轨迹
            }                         
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);

            //publish_map(pubLaserCloudMap);
            int num = robotID2Number(robot_id);
            publish_global_map(pubLaserCloudSurround, state_point, robot_id + "/" + odometryFrame, lidar_end_time, num);
            //输出航向角测试
            // Eigen::Vector3d eulerAngle = state_point.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
            // cout<<"heading:"<<eulerAngle(0)*180/M_PI<<endl;
            // double yaw = atan2(state_point.rot.matrix()(1,0),state_point.rot.matrix()(0,0)) * 180 / M_PI;
            // cout<<"heading:"<<yaw<<endl;
        }

        rate.sleep();
    }
    
    //输出轨迹
    if(0)
    {
        string file_name1 = "/home/ywb/NR_robot/na_mapping/src/na_mapping/Path/lidar_path.txt";
        string file_name2 = "/home/ywb/NR_robot/na_mapping/src/na_mapping/Path/rtk_path.txt";
        ofstream fout;
        fout.open(file_name1);
        cout<<"globalPath size:"<<globalPath.poses.size()<<endl;
        cout<<"keyframe_time size:"<<keyframe_time.size()<<endl;


        for(int i=0;i<globalPath.poses.size();i++)
        {   
            //计算rtk的位置
            V3D pos_imu,pos_rtk;
            Eigen::Quaterniond q_imu;
            
            pos_imu << globalPath.poses[i].pose.position.x,
                    globalPath.poses[i].pose.position.y,
                    globalPath.poses[i].pose.position.z;

            q_imu.w() = globalPath.poses[i].pose.orientation.w;
            q_imu.x() = globalPath.poses[i].pose.orientation.x;
            q_imu.y() = globalPath.poses[i].pose.orientation.y;
            q_imu.z() = globalPath.poses[i].pose.orientation.z;

            pos_rtk = pos_imu + q_imu.normalized().toRotationMatrix() * rtk_T_wrt_Lidar;

            fout << fixed << setprecision(9) << keyframe_time[i] << " "
                << fixed << setprecision(4) << pos_rtk[0] << " "
                << fixed << setprecision(4) << pos_rtk[1] << " "
                << fixed << setprecision(4) << pos_rtk[2] << endl;
        }
        fout.close();

        fout.open(file_name2);
        cout<<"rtk_time size:"<<rtk_time.size()<<endl;
        cout<<"gpspath size:"<<gps_path.poses.size()<<endl;
        for(int i=0;i<gps_path.poses.size();i++)
        {   
            fout << fixed << setprecision(9) << rtk_time[i] << " "
                << fixed << setprecision(4) << gps_path.poses[i].pose.position.x << " "
                << fixed << setprecision(4) << gps_path.poses[i].pose.position.y << " "
                << fixed << setprecision(4) << gps_path.poses[i].pose.position.z << endl;
        }
        fout.close();
    }


    startFlag = false;
    loopthread.join(); //  分离线程

    return 0;
}
