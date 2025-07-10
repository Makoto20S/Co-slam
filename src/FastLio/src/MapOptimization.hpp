
float transformTobeMapped[6]; //  当前帧的位姿(world系下)

/******函数声明******/
void getCurPose(state_ikfom cur_state);
void saveKeyFramesAndFactor(double &lidar_end_time,esekfom::esekf &kf,state_ikfom &state_point,state_ikfom &new_tate_point,PointCloudXYZI::Ptr feats_down_body,PointCloudXYZI::Ptr feats_undistort);
bool saveFrame();
void addOdomFactor();
void addGPSFactor(double &lidar_end_time, state_ikfom &state_point);
void addLoopFactor();
void updatePath(const PointTypePose &pose_in);
void correctPoses(state_ikfom &state_point, KD_TREE<PointType> &ikdtree);
PointTypePose change_to_odom(PointTypePose Pose6D, state_ikfom &sta_point);

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);
Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_);
Eigen::Affine3f trans2Affine3f(float transformIn[]);
gtsam::Pose3 trans2gtsamPose(float transformIn[]);
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);
float pointDistance(PointType p1, PointType p2);

void performLoopClosure(double &lidar_end_time, state_ikfom &state_point);
bool detectLoopClosureDistance(int *latestID, int *closestID, double &lidar_end_time);
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum,state_ikfom &state_point);
void loopClosureThread(double *lidar_end_time, state_ikfom *state_point);
void visualizeLoopClosure(double &lidar_end_time);

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn,state_ikfom &state_point);
pcl::PointCloud<PointType>::Ptr newtransformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn,state_ikfom &state_point);
void recontructIKdTree(state_ikfom &state_point, KD_TREE<PointType> &ikdtree);
void publish_global_map(const ros::Publisher & pubLaserCloudSurround, state_ikfom &state_point, string frame_id, double time_stamp, int num_);

/******变量定义******/
// gtsam
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         // 历史关键帧位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // 历史关键帧位姿
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
//new added
pcl::PointCloud<PointType>::Ptr newcloudKeyPoses3D(new pcl::PointCloud<PointType>());         // 历史关键帧lodo_lidar位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr newcloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // 历史关键帧lodo_lidar位姿
void saveVirtualLoop(const lio_sam::context_infoConstPtr& msgIn);
bool is_key_frame = false;

nav_msgs::Path globalPath;
// Surrounding map
float surroundingkeyframeAddingDistThreshold;  //  判断是否为关键帧的距离阈值
float surroundingkeyframeAddingAngleThreshold; //  判断是否为关键帧的角度阈值

//gps融合相关变量
string savemappath;    // 保存地图路径
string saveposepath;   // 保存起点位置
bool usertk;
bool use_rtk_heading;
string gnss_topic;
string gnss_heading_topic;
// double rtk_time_grift = 43.651;
double rtk_time_grift = 0.0;
double last_timestamp_gnss = -1.0 ;
deque<nav_msgs::Odometry> gnss_buffer;
geometry_msgs::PoseStamped msg_gnss_pose;
shared_ptr<GnssProcess> p_gnss(new GnssProcess());
GnssProcess gnss_data;
bool gnss_inited = false ;                        //  是否完成gnss初始化
double lat0,lon0,alt0;      //初始rtk的经纬高
double utm_x0,utm_y0,utm_z0;            //初始rtk的utm坐标系
nav_msgs::Path gps_path ;
float gpsCovThreshold;          //   gps方向角和高度差的协方差阈值
float poseCovThreshold;       //  位姿协方差阈值  from isam2
bool useGpsElevation;             //  是否使用gps高层优化
double rtk_heading; //rtk航向角
bool rtk_heading_vaild = false;
bool rtk_heading_init = false;
vector<double> rtk2Lidar_T(3, 0.0);
V3D rtk_T_wrt_Lidar(Zero3d);
bool rtk_p0_init = false;

//回环检测相关变量
bool startFlag = true;
bool aLoopIsClosed = false;
bool loopClosureEnableFlag;
float loopClosureFrequency; //   回环检测频率
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
map<int, int> loopIndexContainer; // from new to old
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());
float historyKeyframeSearchRadius;   // 回环检测 radius kdtree搜索半径
float historyKeyframeSearchTimeDiff; //  帧间时间阈值
float historyKeyframeFitnessScore;   // icp 匹配阈值
int historyKeyframeSearchNum;        // 回环时多少个keyframe拼成submap
ros::Publisher pubLoopConstraintEdge;
std::mutex mtx;
pcl::VoxelGrid<PointType> downSizeFilterICP;

// CPU Params
int numberOfCores = 4;

// global map visualization radius
float globalMapVisualizationSearchRadius;
float globalMapVisualizationPoseDensity;
float globalMapVisualizationLeafSize;

bool recontructKdTree = false;
int updateKdtreeCount = 0 ;        //  每100次更新一次
int kdtree_size_st = 0 ;

deque<sensor_msgs::Imu::ConstPtr> imu_rtk_buffer;
state_ikfom state_point_last;
double last_lidar_end_time = 0;

vector<double> keyframe_time; //存储关键帧对应的时间
vector<double> rtk_time; //存储rtk对应的时间

//  eulerAngle 2 Quaterniond
Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_)
{
    Eigen::Quaterniond q ;            //   四元数 q 和 -q 是相等的
    Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
    q = yaw * pitch * roll ;
    q.normalize();
    return q ;
}
/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}
/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}
/**
 * 位姿格式变换
 */
gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}
/**
 * 位姿格式变换
 */
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}
/**
 * 两点之间距离
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}
/**
 * 对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
 */
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn, state_ikfom &state_point)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
   // 注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释
  // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_b_lidar(state_point.offset_R_L_I.matrix());       //  获取  body2lidar  外参
    T_b_lidar.pretranslate(state_point.offset_T_L_I);        

    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body  
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();

    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  转换矩阵

    Eigen::Isometry3d transCur = T_w_lidar;        

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

pcl::PointCloud<PointType>::Ptr newtransformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn, state_ikfom &state_point)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f T_w_l_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_l ;          //   world2body
    T_w_l.matrix() = T_w_l_.matrix().cast<double>();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = T_w_l(0, 0) * pointFrom.x + T_w_l(0, 1) * pointFrom.y + T_w_l(0, 2) * pointFrom.z + T_w_l(0, 3);
        cloudOut->points[i].y = T_w_l(1, 0) * pointFrom.x + T_w_l(1, 1) * pointFrom.y + T_w_l(1, 2) * pointFrom.z + T_w_l(1, 3);
        cloudOut->points[i].z = T_w_l(2, 0) * pointFrom.x + T_w_l(2, 1) * pointFrom.y + T_w_l(2, 2) * pointFrom.z + T_w_l(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}


// 将更新的pose赋值到transformTobeMapped，transformTobeMapped数组最终存储的是当前位姿（位置和方向），通常用于SLAM或其他定位任务中的坐标变换或后端优化
void getCurPose(state_ikfom cur_state) //tate_ikfom 类型的对象中的位置和旋转信息 提取出来，并转换为一个变换矩阵
{
    //  欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
    // V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio 自带  roll pitch yaw  单位: 度，旋转顺序 zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     使用 SO3ToEuler 方法时，顺序是 rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw

    transformTobeMapped[0] = eulerAngle(2);                //  roll(x轴旋转)  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = eulerAngle(1);                //  pitch(y轴旋转)
    transformTobeMapped[2] = eulerAngle(0);                //  yaw(z轴旋转)
    // transformTobeMapped[2] = atan2(cur_state.rot.matrix()(1,0),cur_state.rot.matrix()(0,0));
    transformTobeMapped[3] = cur_state.pos(0);          //  x
    transformTobeMapped[4] = cur_state.pos(1);          //   y
    transformTobeMapped[5] = cur_state.pos(2);          // z
}

/**
 * 更新里程计轨迹
 */
void updatePath(const PointTypePose &pose_in)
{
    string odometryFrame = "camera_init";
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

/**
 * 将位姿还原到body到bodm
 */
PointTypePose change_to_odom(PointTypePose Pose6D, state_ikfom &sta_point) {
    Eigen::Quaterniond q_ = Eigen::Quaterniond(sta_point.offset_R_L_I.matrix());
    gtsam::Pose3 bodytolidar = gtsam::Pose3(gtsam::Rot3::Quaternion(q_.w(), q_.x(), q_.y(), q_.z()), gtsam::Point3(sta_point.offset_T_L_I.x(),
                                            sta_point.offset_T_L_I.y(), sta_point.offset_T_L_I.z()));
    gtsam::Pose3 lidmtoldiar = gtsam::Pose3(gtsam::Rot3::RzRyRx(Pose6D.roll, Pose6D.pitch, Pose6D.yaw),
                                           gtsam::Point3(Pose6D.x, Pose6D.y, Pose6D.z));
    gtsam::Pose3 lidmtobody = lidmtoldiar * bodytolidar.inverse();
    gtsam::Pose3 bdodtobody = bodytolidar * lidmtobody;

    tf::Quaternion q1(bdodtobody.rotation().toQuaternion().x(), bdodtobody.rotation().toQuaternion().y(),
                      bdodtobody.rotation().toQuaternion().z(), bdodtobody.rotation().toQuaternion().w());//四元数初始化参数顺序为x,y,z,w
    double roll, pitch, yaw;
    tf::Matrix3x3(q1).getRPY(roll, pitch, yaw);
    PointTypePose change_pose;
    change_pose.x = bdodtobody.translation().x();
    change_pose.y = bdodtobody.translation().y();
    change_pose.z = bdodtobody.translation().z();
    change_pose.intensity = Pose6D.intensity;
    change_pose.roll = roll;
    change_pose.pitch = pitch;
    change_pose.yaw = yaw;
    change_pose.time = Pose6D.time;
    return change_pose;
}

/**
 * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
 */
bool saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    // 前一帧位姿
    Eigen::Affine3f transStart = pclPointToAffine3f(newcloudKeyPoses6D->back());//ldodm-lidar
    // 当前帧位姿
    Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);
    // Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                    
    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿

    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;
    return true;
}

/**
 * 添加激光里程计因子
 */
void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // 第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e1, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));//ldodm-lidar
        // 变量节点设置初始值
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        // 添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(newcloudKeyPoses6D->points.back()); /// pre
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   // cur
        // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(newcloudKeyPoses3D->size() - 1, newcloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // 变量节点设置初始值
        initialEstimate.insert(newcloudKeyPoses3D->size(), poseTo);
    }
}
/**
 * 添加GPS因子
*/
void addGPSFactor(double &lidar_end_time, state_ikfom &state_point)
{
    if (usertk==false)
        return;

    if (gnss_buffer.empty())
        return;
    // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
    if (cloudKeyPoses3D->points.empty())
        return;
    else
    {
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < -1.0)
            return;
    }
    // 位姿协方差很小，没必要加入GPS数据进行校正
    if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
        return;

    static PointType lastGPSPoint;      // 最新的gps数据
    while (!gnss_buffer.empty())
    {
        // 删除当前帧0.1s之前的里程计
        if (gnss_buffer.front().header.stamp.toSec() < lidar_end_time - 0.1)//0.2
        {
            gnss_buffer.pop_front();
        }
        // 超过当前帧0.1s之后，退出
        else if (gnss_buffer.front().header.stamp.toSec() > lidar_end_time + 0.1)//0.2
        {
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();
            // GPS噪声协方差太大，不能用
            float noise_x = thisGPS.pose.covariance[0];         //  x 方向的协方差
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];      //   z(高层)方向的协方差
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold || noise_z > gpsCovThreshold)
                continue;
            /****************GPS杆臂处理***************/
            //计算当前的位置和姿态(利用imu补偿姿态)
            Sophus::SO3d rot_rtk = state_point_last.rot;
            V3D vel_rtk = state_point_last.vel;
            V3D pos_rtk = state_point_last.pos;
            // M3D matrix_begin = state_point.rot.matrix();
            // M3D matrix_end = state_point.rot.matrix();
            bool first_imu = true;
            double dt_rtk = 0.005;
            double imu_time_last = last_lidar_end_time;
            V3D angvel_avr,acc_avr;
            //丢掉上一个lidar时刻前的imu数据
            while(!imu_rtk_buffer.empty() && (last_lidar_end_time>0) && (imu_rtk_buffer.front()->header.stamp.toSec()<last_lidar_end_time))
            {
                imu_rtk_buffer.pop_front();
            }
            //更新姿态
            while(!imu_rtk_buffer.empty() && (last_lidar_end_time>0) && (imu_rtk_buffer.front()->header.stamp.toSec() <= thisGPS.header.stamp.toSec()))
            {
                //计算dt
                if(first_imu)
                {
                    dt_rtk = imu_rtk_buffer.front()->header.stamp.toSec() - last_lidar_end_time;
                    first_imu = false;
                    imu_time_last = imu_rtk_buffer.front()->header.stamp.toSec();
                }
                else
                {
                    dt_rtk = imu_rtk_buffer.front()->header.stamp.toSec() - imu_time_last;
                    imu_time_last = imu_rtk_buffer.front()->header.stamp.toSec();
                }

                
                angvel_avr << imu_rtk_buffer.front()->angular_velocity.x,
                              imu_rtk_buffer.front()->angular_velocity.y,
                              imu_rtk_buffer.front()->angular_velocity.z;

                acc_avr << imu_rtk_buffer.front()->linear_acceleration.x,
                           imu_rtk_buffer.front()->linear_acceleration.y,
                           imu_rtk_buffer.front()->linear_acceleration.z;

                rot_rtk = rot_rtk * Sophus::SO3d::exp(angvel_avr * dt_rtk);     
                acc_avr = rot_rtk * acc_avr + state_point_last.grav;
                vel_rtk += acc_avr * dt_rtk;
                pos_rtk += vel_rtk * dt_rtk;

                imu_rtk_buffer.pop_front();           
            }


            V3D dp;

            // dp=state_point.rot.matrix()*rtk_T_wrt_Lidar;
            dp = rot_rtk.matrix()*rtk_T_wrt_Lidar;

            float gps_x = thisGPS.pose.pose.position.x-dp[0];
            float gps_y = thisGPS.pose.pose.position.y-dp[1];
            float gps_z = thisGPS.pose.pose.position.z-dp[2];

            //丢掉rtk时刻前的imu数据
            while(!imu_rtk_buffer.empty() && (imu_rtk_buffer.front()->header.stamp.toSec()<thisGPS.header.stamp.toSec()))
            {
                imu_rtk_buffer.pop_front();
            }
            // //计算姿态和速度
            while(!imu_rtk_buffer.empty() && (imu_rtk_buffer.front()->header.stamp.toSec()<lidar_end_time))
            {
                dt_rtk = imu_rtk_buffer.front()->header.stamp.toSec() - imu_time_last;
                imu_time_last = imu_rtk_buffer.front()->header.stamp.toSec();

                angvel_avr << imu_rtk_buffer.front()->angular_velocity.x,
                              imu_rtk_buffer.front()->angular_velocity.y,
                              imu_rtk_buffer.front()->angular_velocity.z;

                acc_avr << imu_rtk_buffer.front()->linear_acceleration.x,
                           imu_rtk_buffer.front()->linear_acceleration.y,
                           imu_rtk_buffer.front()->linear_acceleration.z;

                rot_rtk = rot_rtk * Sophus::SO3d::exp(angvel_avr * dt_rtk); 
                acc_avr = rot_rtk * acc_avr + state_point_last.grav;
                vel_rtk += acc_avr * dt_rtk;   

                gps_x += vel_rtk[0] * dt_rtk;
                gps_y += vel_rtk[1] * dt_rtk;
                gps_z += vel_rtk[2] * dt_rtk;

                imu_rtk_buffer.pop_front();
            }

            if (!useGpsElevation)           //  是否使用gps的高度
            {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }

            // (0,0,0)无效数据
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                continue;
            // 每隔5m添加一个GPS里程计
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;

            if (pointDistance(curGPSPoint, lastGPSPoint) < 1.0)
                continue;
            else
                lastGPSPoint = curGPSPoint;

            // 添加GPS因子
            gtsam::Vector Vector3(3);
            // Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
            // Vector3 << max(2.0f*noise_x, 0.1f*0.1f), max(2.0f*noise_y, 0.1f*0.1f), max(2.0f*noise_z, 0.2f*0.2f);
            Vector3 << noise_x,noise_y,noise_z;
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
            gtSAMgraph.add(gps_factor);
            aLoopIsClosed = true;
            ROS_INFO("GPS Factor Added");
            break;
        }
    }
}

/**
 * 添加闭环因子
 */
void addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first; //   cur
        int indexTo = loopIndexQueue[i].second;  //    pre
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

//回环检测三大要素
// 1.设置最小时间差，太近没必要
// 2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
// 3.根据当前最小距离重新计算等待时间
bool detectLoopClosureDistance(int *latestID, int *closestID, double &lidar_end_time)
{
    // 当前关键帧帧
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  当前关键帧索引
    int loopKeyPre = -1;

    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;                    //  候选关键帧距离
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
//        if (abs(copy_cloudKeyPoses6D->points[id].z - copy_cloudKeyPoses6D->points[loopKeyCur].z) > 0.5){
//           continue;
//        }
        if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    // ROS_INFO("Find loop clousre frame ");
    return true;
}

/**
 * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
 */
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum,state_ikfom &state_point)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;

        // *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
        *nearKeyframes += *newtransformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear],state_point); //  fast-lio 没有进行特征提取，默认点云就是surf
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

//回环检测线程
void loopClosureThread(double *lidar_end_time_ptr, state_ikfom *state_point_ptr)
{
    if (loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   回环频率
    while (ros::ok() && startFlag)
    {
        rate.sleep();
        performLoopClosure(*lidar_end_time_ptr, *state_point_ptr);   //  回环检测
        visualizeLoopClosure(*lidar_end_time_ptr); // rviz展示闭环边
    }
}

/**
 * rviz展示闭环边
 */
void visualizeLoopClosure(double &lidar_end_time)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";

    if (loopIndexContainer.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void performLoopClosure(double &lidar_end_time, state_ikfom &state_point)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";

    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    mtx.lock();
//    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
//    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    *copy_cloudKeyPoses3D = *newcloudKeyPoses3D;//ldodm-lidar
    *copy_cloudKeyPoses6D = *newcloudKeyPoses6D;
    mtx.unlock();

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre,lidar_end_time) == false)
    {
        return;
    }

    // 提取
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //  cue keyframe
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //   history keyframe submap
    {
        // 提取当前关键帧特征点集合，降采样
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0, state_point); //  将cur keyframe 转换到world系下
        // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, state_point); //  选取historyKeyframeSearchNum个keyframe拼成submap
        // 如果特征点较少，返回
        // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        //     return;
        // 发布闭环匹配关键帧局部map 感觉用不到
        // if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        //     publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // scan-to-map，调用icp匹配
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // 未收敛，或者匹配不够好
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    std::cout << "icp  success  " << std::endl;

    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云 (感觉用不到)
    // if (pubIcpKeyFrames.getNumSubscribers() != 0)
    // {
    //     pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
    //     pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
    //     publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    // }

    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // 闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore() ; //  loop_clousre  noise from icp
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

    // 添加闭环因子需要的数据
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    loopIndexContainer[loopKeyCur] = loopKeyPre; //   使用hash map 存储回环对
}

void saveKeyFramesAndFactor(double &lidar_end_time,esekfom::esekf &kf,state_ikfom &state_point,state_ikfom &new_tate_point,PointCloudXYZI::Ptr feats_down_body, PointCloudXYZI::Ptr feats_undistort)
{
    //  计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (saveFrame() == false) {
        is_key_frame = false;
        return;
    }

    addOdomFactor(); //添加里程计因子
    
    // GPS因子 (UTM -> WGS84)
    addGPSFactor(lidar_end_time, state_point);
    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();
    // 执行优化
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed == true) // 有回环因子，多update几次（纠正累积的误差）
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0); //更新优化后，清空保存的因子图和初始估计
    initialEstimate.clear();

    PointType thisPose3D; //ldodm-lidar
    PointTypePose thisPose6D; // ldodm-lidar
    gtsam::Pose3 latestEstimate;

    // 优化结果
    isamCurrentEstimate = isam->calculateBestEstimate();
    // 当前帧位姿结果
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);

    // cloudKeyPoses3D加入当前帧位置
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    // 索引
    thisPose3D.intensity = cloudKeyPoses3D->size(); //  使用intensity作为该帧点云的index
    newcloudKeyPoses3D->push_back(thisPose3D);      //  ldodm-lidar

    // cloudKeyPoses6D加入当前帧位姿
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    newcloudKeyPoses6D->push_back(thisPose6D);      //  ldodm-lidar

    //new added
    Eigen::Vector3d pos_new(thisPose6D.x, thisPose6D.y, thisPose6D.z);//  ldodm-lidar
    Eigen::Quaterniond q_new = EulerToQuat(thisPose6D.roll, thisPose6D.pitch, thisPose6D.yaw);
    new_tate_point.pos = pos_new;
    new_tate_point.rot = q_new.toRotationMatrix();//  ldodm-lidar
    PointTypePose changePose6D = change_to_odom(thisPose6D, state_point);//bdodm-body,将当前位姿从 lidar 坐标系转换到 body 坐标系，通过 change_to_odom 函数实现坐标转换
    cloudKeyPoses6D->push_back(changePose6D);//change_to_odom() 函数将当前的位姿从激光坐标系（Lidar Frame）转换到体坐标系（Body Frame）
    PointType changePose3D;
    changePose3D.x = changePose6D.x;
    changePose3D.y = changePose6D.y;
    changePose3D.z = changePose6D.z;
    changePose3D.intensity = thisPose3D.intensity;
    cloudKeyPoses3D->push_back(changePose3D);         //  新关键帧帧放入队列中

    keyframe_time.push_back(lidar_end_time);
    is_key_frame = true;

    // 位姿协方差
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // ESKF状态和方差  更新
    state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
//    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
//    Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());
    Eigen::Vector3d pos(changePose6D.x, changePose6D.y, changePose6D.z);//bdodm-body
    Eigen::Quaterniond q = EulerToQuat(changePose6D.roll, changePose6D.pitch, changePose6D.yaw);

    //  更新状态量和协方差
    state_updated.pos = pos;
    state_updated.rot =  q.toRotationMatrix();
    state_point = state_updated; // 对state_point进行更新，state_point可视化用到
    // if(aLoopIsClosed == true )
    kf.change_x(state_updated);  //  对cur_pose 进行isam2优化后的修正

    // TODO:  P的修正有待考察，按照yanliangwang的做法，修改了p，会跑飞
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); // 获取当前的状态估计的协方差矩阵
    // P_updated.setIdentity();
    // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
    // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
    // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
    // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
    // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
    // kf.change_P(P_updated);

    // 当前帧激光角点、平面点，降采样集合
    // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());//将当前帧的特征点数据（经过降采样的点云）保存到关键帧队列中
    // pcl::copyPointCloud(*feats_undistort,  *thisCornerKeyFrame);

    // pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // 存储关键帧,没有降采样的点云
    pcl::copyPointCloud(*feats_down_body, *thisSurfKeyFrame); // 存储关键帧,降采样的点云

    // 保存特征点降采样集合
    // cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // cout<<"size of surfCloudKeyFrames:"<<sizeof(*(surfCloudKeyFrames[0]))*surfCloudKeyFrames.size()<<endl;

    updatePath(thisPose6D); //  可视化update后的path ldodm-lidar
    
}

//重构ikd树
void recontructIKdTree(state_ikfom &state_point, KD_TREE<PointType> &ikdtree){
    if(recontructKdTree  &&  updateKdtreeCount >  0){
        /*** if path is too large, the rvis will crash ***/
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);     //  subMap的pose集合
        // 降采样
        pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;//当前帧子地图包含的位姿降采样
        downSizeFilterSubMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
        downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS);         //  subMap poses  downsample
        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
        {
            // 距离过大
            if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                    continue;
            int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
            // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd],state_point); //  fast_lio only use  surfCloud
        }
        // 降采样，发布
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);

        std::cout << "subMapKeyFramesDS sizes  =  "   << subMapKeyFramesDS->points.size()  << std::endl;
        
        ikdtree.reconstruct(subMapKeyFramesDS->points);//删除当前所有的点云缓存，根据输入的点云，重新构造ikdtree
        updateKdtreeCount = 0;
        ROS_INFO("Reconstructed  ikdtree ");
        int featsFromMapNum = ikdtree.validnum();
        kdtree_size_st = ikdtree.size();
        std::cout << "featsFromMapNum  =  "   << featsFromMapNum   <<  "\t" << " kdtree_size_st   =  "  <<  kdtree_size_st  << std::endl;
    }
        updateKdtreeCount ++ ; 
}

void publish_global_map(const ros::Publisher & pubLaserCloudSurround, state_ikfom &state_point, string frame_id, double time_stamp, int num_)
{
    if (is_key_frame == false)
        return;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;

    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(newcloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(newcloudKeyPoses3D->back(), 1000.0, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(newcloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(5.0, 5.0, 5.0); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
        if (pointDistance(globalMapKeyPosesDS->points[i], newcloudKeyPoses3D->back()) > 1000.0)
            continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *newtransformPointCloud(surfCloudKeyFrames[thisKeyInd], &newcloudKeyPoses6D->points[thisKeyInd], state_point); //  fast_lio only use  surfCloud
    }

    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(1.0, 1.0, 1.0); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::PointCloud<DiscoType>::Ptr cloudOut(new pcl::PointCloud<DiscoType>());
    cloudOut->points.resize(globalMapKeyFramesDS->points.size());

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < globalMapKeyFramesDS->points.size(); ++i) {
        cloudOut->points[i].x = globalMapKeyFramesDS->points[i].x;
        cloudOut->points[i].y = globalMapKeyFramesDS->points[i].y;
        cloudOut->points[i].z = globalMapKeyFramesDS->points[i].z;
//        cloudOut->points[i].intensity = globalMapKeyFramesDS->points[i].intensity;
        cloudOut->points[i].intensity = num_ * 50;
    }

    pcl::toROSMsg(*cloudOut, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(time_stamp);
    laserCloudMap.header.frame_id = frame_id;
    pubLaserCloudSurround.publish(laserCloudMap);
}

void saveVirtualLoop(const lio_sam::context_infoConstPtr& msgIn){
    int indexFrom = msgIn->numRing;
    int indexTo = msgIn->numSector;
    gtsam::Pose3 poseBetween = gtsam::Pose3( gtsam::Rot3::RzRyRx(msgIn->poseRoll, msgIn->posePitch, msgIn->poseYaw),
                                     gtsam::Point3(msgIn->poseX, msgIn->poseY, msgIn->poseZ) );
    float noiseScore = msgIn->poseIntensity;
    gtsam::Vector Vector6(6);
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
        noiseScore;
    auto noiseBetween = gtsam::noiseModel::Diagonal::Variances(Vector6);

    mtx.lock();
    loopIndexQueue.push_back(make_pair(indexFrom, indexTo));
    loopPoseQueue.push_back(poseBetween);
    loopNoiseQueue.push_back(noiseBetween);
    mtx.unlock();
}

/**
 * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
 */
void correctPoses(state_ikfom &state_point, KD_TREE<PointType> &ikdtree)
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // 清空里程计轨迹
        globalPath.poses.clear();
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            newcloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            newcloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            newcloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            newcloudKeyPoses6D->points[i].x = newcloudKeyPoses3D->points[i].x;
            newcloudKeyPoses6D->points[i].y = newcloudKeyPoses3D->points[i].y;
            newcloudKeyPoses6D->points[i].z = newcloudKeyPoses3D->points[i].z;
            newcloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            newcloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            newcloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            //new added
            PointTypePose changePose6D = change_to_odom(newcloudKeyPoses6D->points[i], state_point);
            cloudKeyPoses6D->points[i] = changePose6D;
            cloudKeyPoses3D->points[i].x = changePose6D.x;
            cloudKeyPoses3D->points[i].y = changePose6D.y;
            cloudKeyPoses3D->points[i].z = changePose6D.z;
            cloudKeyPoses3D->points[i].intensity = changePose6D.intensity;

            // 更新里程计轨迹
            updatePath(newcloudKeyPoses6D->points[i]);
        }
        // 清空局部map， reconstruct  ikdtree submap
        recontructIKdTree(state_point, ikdtree);
        // ROS_INFO("ISMA2 Update");
        aLoopIsClosed = false;
    }
}

