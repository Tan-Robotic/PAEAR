#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_srvs/Trigger.h>
#include <visualization_msgs/MarkerArray.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <urdf/model.h>
#include <Eigen/Geometry>
#include <mutex>
#include <boost/bind.hpp>

struct VisualInfo
{
  std::string link_name;
  std::string mesh;
  Eigen::Vector3d scale;
  Eigen::Isometry3d origin;
};

class SnapshotGhost
{
public:
  SnapshotGhost(ros::NodeHandle& nh)
    : nh_(nh)
  {
    nh_.param("alpha", alpha_, 0.3);
    nh_.param("marker_topic", marker_topic_, std::string("visualization_marker_array"));
    nh_.param("use_embedded_materials", use_embedded_materials_, true);
    nh_.param("joint_states_topic", joint_states_topic_, std::string("/joint_states"));

    robot_model_loader::RobotModelLoader loader("robot_description");
    model_ = loader.getModel();
    state_.reset(new robot_state::RobotState(model_));
    state_->setToDefaultValues();

    urdf_model_.initParam("robot_description");
    for (const auto& kv : urdf_model_.links_)
    {
      const urdf::LinkSharedPtr& link = kv.second;
      if (!link || !link->visual || !link->visual->geometry)
        continue;
      urdf::Mesh* mesh = dynamic_cast<urdf::Mesh*>(link->visual->geometry.get());
      if (!mesh)
        continue;
      VisualInfo v;
      v.link_name = link->name;
      v.mesh = mesh->filename;
      v.scale = Eigen::Vector3d(1.0, 1.0, 1.0);
      if (mesh->scale.x != 0.0 && mesh->scale.y != 0.0 && mesh->scale.z != 0.0)
        v.scale = Eigen::Vector3d(mesh->scale.x, mesh->scale.y, mesh->scale.z);
      Eigen::Isometry3d o = Eigen::Isometry3d::Identity();
      const urdf::Pose& op = link->visual->origin;
      double rr, rp, ryaw;
      op.rotation.getRPY(rr, rp, ryaw);
      Eigen::AngleAxisd rx(rr, Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd ry(rp, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd rz(ryaw, Eigen::Vector3d::UnitZ());
      o.linear() = (rz * ry * rx).toRotationMatrix();
      o.translation() = Eigen::Vector3d(op.position.x, op.position.y, op.position.z);
      v.origin = o;
      visuals_.push_back(v);
    }

    pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 10);
    ros::NodeHandle nh_public;
    sub_ = nh_public.subscribe<sensor_msgs::JointState>(joint_states_topic_, 50, boost::bind(&SnapshotGhost::jointStateCb, this, _1));
    srv_ = nh_.advertiseService("freeze_snapshot", &SnapshotGhost::freezeSnapshot, this);
  }

  void jointStateCb(const sensor_msgs::JointState::ConstPtr& js)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < js->name.size() && i < js->position.size(); ++i)
    {
      if (model_->hasJointModel(js->name[i]))
        state_->setVariablePosition(js->name[i], js->position[i]);
    }
    state_->updateLinkTransforms();
  }

  bool freezeSnapshot(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    visualization_msgs::MarkerArray arr;
    int id = 0;
    for (const auto& v : visuals_)
    {
      Eigen::Isometry3d T = state_->getGlobalLinkTransform(v.link_name) * v.origin;
      Eigen::Quaterniond q(T.rotation());
      visualization_msgs::Marker mk;
      mk.header.frame_id = model_->getModelFrame();
      mk.header.stamp = ros::Time::now();
      mk.ns = std::string("snapshot_") + std::to_string(snapshot_count_);
      mk.id = id++;
      mk.type = visualization_msgs::Marker::MESH_RESOURCE;
      mk.action = visualization_msgs::Marker::ADD;
      mk.mesh_resource = v.mesh;
      mk.mesh_use_embedded_materials = use_embedded_materials_;
      mk.pose.position.x = T.translation().x();
      mk.pose.position.y = T.translation().y();
      mk.pose.position.z = T.translation().z();
      mk.pose.orientation.x = q.x();
      mk.pose.orientation.y = q.y();
      mk.pose.orientation.z = q.z();
      mk.pose.orientation.w = q.w();
      mk.scale.x = v.scale.x();
      mk.scale.y = v.scale.y();
      mk.scale.z = v.scale.z();
      if (use_embedded_materials_)
      {
        mk.color.r = 1.0;
        mk.color.g = 1.0;
        mk.color.b = 1.0;
      }
      else
      {
        double r = 1.0, g = 0.5, b = 0.0;
        if (v.link_name == "base_Link" || v.link_name == "s_Link" || v.link_name == "f1_Link" || v.link_name == "u_Link" || v.link_name == "w1_Link")
        { r = 0.1; g = 0.1; b = 0.1; }
        mk.color.r = r;
        mk.color.g = g;
        mk.color.b = b;
      }
      mk.color.a = alpha_;
      mk.lifetime = ros::Duration(0.0);
      arr.markers.push_back(mk);
    }
    pub_.publish(arr);
    snapshot_count_++;
    res.success = true;
    res.message = std::string("snapshot_") + std::to_string(snapshot_count_);
    return true;
  }

private:
  ros::NodeHandle nh_;
  double alpha_;
  std::string marker_topic_;
  bool use_embedded_materials_;
  std::string joint_states_topic_;
  robot_model::RobotModelPtr model_;
  boost::shared_ptr<robot_state::RobotState> state_;
  urdf::Model urdf_model_;
  std::vector<VisualInfo> visuals_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
  ros::ServiceServer srv_;
  std::mutex mtx_;
  int snapshot_count_ = 0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "snapshot_ghost");
  ros::NodeHandle nh("~");
  SnapshotGhost ghost(nh);
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}

