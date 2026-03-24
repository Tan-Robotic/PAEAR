
# Aubo机械臂ROS包：
```
roslaunch aubo_i10_moveit_config moveit_planning_execution.launch robot_ip:=127.0.0.1
```
```
roslaunch aubo_i10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.192.5
```
```
roslaunch aubo_gazebo aubo_i10_gazebo_control.launch
```

# Mech相机ROS包：

```
roslaunch mecheye_ros_interface start_camera.launch
```
```
rosservice call /capture_color_image
```
```
rosservice call /capture_depth_map
```
```
rosservice call /capture_point_cloud
```
```
rosservice call /capture_textured_point_cloud
```

# 分割与轨迹ROS包：
```
rosrun PAEAP_ros seg_main_ros.py _scale:=1.0 _voxel_size:=0.004 _r:=0.02 _labels_display:="[2]" _k_neighbors:=25
```
```
rosrun PAEAP_ros tra_main_ros.py
```




# 重置网络（虚拟机中）：
`sudo dpkg-reconfigure network-manager`

# 重启NetworkManager（真linux）——已解决（设置网关为192.168.192.1）
`sudo systemctl restart NetworkManager`



# IP说明：
`机械臂IP`：192.168.192.5
`PC端IP`：192.168.192.10
`Ubuntu端IP`：192.168.192.8
`Mecheye相机IP`：192.168.192.20




20251201-测试通讯

使用 roslaunch（推荐）
- 设置并启动接口：
roslaunch industrial_robot_client robot_interface_streaming.launch robot_ip:=192.168.192.5
- 调用测试服务（不会让机器人动）：
rosservice call /motion_streaming_interface/stop_motion
- 在启动该接口的终端查看日志，出现：
comm: send_and_reply stop_motion=0.123456s success=1


方式 A：仅设置关节名参数（最快）

- 加载 Aubo i10 的关节名：
  - rosparam load $(rospack find aubo_i10_moveit_config)/config/joint_names.yaml
- 验证：
  - rosparam get controller_joint_names