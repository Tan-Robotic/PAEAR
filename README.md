## PAEAR: A Method for Active Weld Seam Recognition

## Repository Description:
In this repository, the`aubo_robot`folder contains the ROS package used for simulation and real-time control in this paper; `industrial_core`contains dependencies related to robotic arm control; and the `mecheye_ros_interface` folder contains the ROS package for the Mech 3D camera. A complete simulation scenario featuring the 3D camera can be viewed at: https://github.com/Tan-Robotic/aubo_sim3dcamera. We have stored the initial development version of the PAEAR framework in `PAEAR_src`, but the final, complete version will be uploaded and made publicly available after the paper is published.


## Aubo Robot Arm ROS Package:

Launch the simulation environment:
```
roslaunch aubo_i10_moveit_config moveit_planning_execution.launch robot_ip:=127.0.0.1
```
Launch the physical environment:
```
roslaunch aubo_i10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.192.5
```
Launch the Gazebo simulation:
```
roslaunch aubo_gazebo aubo_i10_gazebo_control.launch
```


## Mech 3D Camera ROS Package
Start the camera：
```
roslaunch mecheye_ros_interface start_camera.launch
```
Capture point cloud：
```
rosservice call /capture_point_cloud
```
Capture a colored point cloud：
```
rosservice call /capture_textured_point_cloud
```

