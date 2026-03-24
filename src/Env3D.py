import gym
from gym import spaces
import numpy as np
import matplotlib
import open3d as o3d

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
import math
import os

class PointCloudEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 pcd_file='foregroud.pcd',
                 voxel_size=0.5,
                 initial_agent_radius=10.0,
                 action_distance=0.3,
                 k_neighbors=25,
                 curvature_threshold=10.0,
                 min_clusters=2,
                 max_clusters=3,
                 grid_size=0.5,
                 exploration_bonus=0.01,
                 penalty_factor=0.001,
                 min_angle=85.0,
                 min_proportion=0.1,
                 max_segments=10,
                 guided_steps_after_segment=10,
                 **kwargs):
        super(PointCloudEnv, self).__init__()

        self.pcd_file = pcd_file
        self.voxel_size = voxel_size
        self.initial_agent_radius = initial_agent_radius
        self.current_radius = self.initial_agent_radius
        self.radius_decrement = 0.05
        self.min_radius = 0.4
        self.action_distance = action_distance
        self.k_neighbors = k_neighbors
        self.curvature_threshold = curvature_threshold
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.grid_size = grid_size
        self.exploration_bonus = exploration_bonus
        self.penalty_factor = penalty_factor
        self.max_segments = max_segments
        self.segment_count = 0
        self.min_angle = min_angle
        self.min_proportion = min_proportion
        self.guided_steps_after_segment = guided_steps_after_segment
        self.guided_steps_remaining = 0
        self.current_guided_direction = None

        # 读取点云
        self.points = self.voxel_downsample(self.load_pcd(self.pcd_file), voxel_size=self.voxel_size)
        if len(self.points) == 0:
            raise ValueError(f"点云文件 {self.pcd_file} 为空或未读取到数据！")
        self.initial_points = self.points.copy()
        self.line_length = float(np.max(self.points, axis=0)[0] - np.min(self.points, axis=0)[0])
        self.num_points_per_line = len(self.points)
        self.bounding_box = self.compute_bounding_box()
        self.centroid = self.points.mean(axis=0)
        self.max_hull_volume = self.compute_max_hull_volume()
        self.max_num_inside = len(self.points)
        self.init_grid_counter()
        self.init_render()
        self.curvature_steps = 0
        self.required_consecutive_steps = 1
        self.episode_num = 0

        self.action_space = spaces.Box(low=np.array([-np.pi, -np.pi / 2], dtype=np.float32),
                                       high=np.array([np.pi, np.pi / 2], dtype=np.float32),
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.line_length,
                                            high=self.line_length,
                                            shape=(3,), dtype=np.float32)
        self.trajectory = []
        self.alpha = 0.1
        self.beta = -0.05
        self.curvature_reward = 1.0
        self.large_penalty = 0.001
        self.gamma = 0.3
        self.exploration_bonus = exploration_bonus
        self.penalty_factor = penalty_factor
        self.prev_num_inside = 0
        self.current_step = 0

    def load_pcd(self, filename):
        points = []
        with open(filename, 'r') as f:
            data_start = False
            for line in f:
                line = line.strip()
                if line.startswith('DATA'):
                    data_start = True
                    continue
                if not data_start or line.startswith('#') or len(line) == 0:
                    continue
                tokens = line.split()
                if len(tokens) >= 3:
                    try:
                        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
                        points.append([x, y, z])
                    except:
                        continue
        return np.array(points, dtype=np.float32)

    def voxel_downsample(self, points, voxel_size=0.5):
        if len(points) == 0:
            return points
        # 计算每个点所属体素格子的索引
        coords = np.floor(points / voxel_size).astype(np.int32)
        # 利用唯一体素索引作为分组依据
        voxel_dict = {}
        for idx, grid_idx in enumerate(map(tuple, coords)):
            if grid_idx not in voxel_dict:
                voxel_dict[grid_idx] = []
            voxel_dict[grid_idx].append(idx)
        # 对每个体素格子的点，选第一个（也可选中心或平均）
        downsampled = [points[indices[0]] for indices in voxel_dict.values()]
        return np.array(downsampled, dtype=np.float32)

    def init_grid_counter(self):
        min_xyz, max_xyz = self.bounding_box
        self.grid_x = int(math.ceil((max_xyz[0] - min_xyz[0]) / self.grid_size))
        self.grid_y = int(math.ceil((max_xyz[1] - min_xyz[1]) / self.grid_size))
        self.grid_z = int(math.ceil((max_xyz[2] - min_xyz[2]) / self.grid_size))
        self.grid_counter = np.zeros((self.grid_x, self.grid_y, self.grid_z), dtype=np.int32)

    def get_grid_index(self, position):
        """根据位置获取网格索引。
        Args:
            position (np.ndarray): 智能体的位置，形状为 (3,)

        Returns:
            tuple: (i, j, k) 网格索引
        """
        min_xyz, _ = self.bounding_box
        i = int((position[0] - min_xyz[0]) / self.grid_size)
        j = int((position[1] - min_xyz[1]) / self.grid_size)
        k = int((position[2] - min_xyz[2]) / self.grid_size)
        # 确保索引在范围内
        i = np.clip(i, 0, self.grid_x - 1)
        j = np.clip(j, 0, self.grid_y - 1)
        k = np.clip(k, 0, self.grid_z - 1)
        return (i, j, k)

    def generate_L_shape(self):
        plane_size = self.line_length

        x1 = np.linspace(0, plane_size, self.num_points_per_line)
        y1 = np.linspace(0, plane_size, self.num_points_per_line)
        x1_grid, y1_grid = np.meshgrid(x1, y1)
        z1_grid = np.zeros_like(x1_grid)
        plane1 = np.stack([x1_grid.flatten(), y1_grid.flatten(), z1_grid.flatten()], axis=1)

        y2 = np.linspace(0, plane_size, self.num_points_per_line)
        z2 = np.linspace(0, plane_size, self.num_points_per_line)
        y2_grid, z2_grid = np.meshgrid(y2, z2)
        x2_grid = np.zeros_like(y2_grid)
        plane2 = np.stack([x2_grid.flatten(), y2_grid.flatten(), z2_grid.flatten()], axis=1)

        noise1 = np.random.normal(0, self.noise_std, plane1.shape)
        noise2 = np.random.normal(0, self.noise_std, plane2.shape)
        plane1_noisy = plane1 + noise1
        plane2_noisy = plane2 + noise2
        points = np.vstack([plane1_noisy, plane2_noisy])

        return points

    def compute_bounding_box(self):
        """计算点云的轴对齐最小边界盒。"""
        if len(self.points) == 0:
            return (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        min_xyz = self.points.min(axis=0)
        max_xyz = self.points.max(axis=0)

        min_xyz -= 0.5
        max_xyz += 0.5
        return (min_xyz, max_xyz)  # 返回最小和最大坐标

    def compute_max_hull_volume(self):
        """估计最大可能的凸包体积用于归一化。"""
        if len(self.points) < 4:
            return 0.0
        try:
            hull = ConvexHull(self.points)
            return hull.volume  # 计算凸包体积
        except:
            return 0.0

    def init_render(self):
        """初始化渲染窗口和绘图元素。"""
        plt.ion()  
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                                       c='blue', s=30, alpha=0.6, label='Points')


        self.bounding_box_lines = []

        min_xyz, max_xyz = self.bounding_box
        self.ax.set_xlim(min_xyz[0] - 1, max_xyz[0] + 1)
        self.ax.set_ylim(min_xyz[1] - 1, max_xyz[1] + 1)
        self.ax.set_zlim(min_xyz[2] - 1, max_xyz[2] + 1)

        self.ax.set_title("3D Point Cloud Environment")
        self.ax.legend()

        self.draw_bounding_box()

        plt.show()

    def draw_bounding_box(self):
        """绘制边界盒线条。"""
        min_xyz, max_xyz = self.bounding_box
        corners = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ])

        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]

        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            line, = self.ax.plot([p1[0], p2[0]],
                                 [p1[1], p2[1]],
                                 [p1[2], p2[2]], 'k-', alpha=0.5)
            self.bounding_box_lines.append(line)

    def reset_render(self):
        """重置渲染窗口中的动态元素。"""
        if hasattr(self, 'agent_surface'):
            self.agent_surface.remove()

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.agent_pos[0] + self.current_radius * np.cos(u) * np.sin(v)
        y = self.agent_pos[1] + self.current_radius * np.sin(u) * np.sin(v)
        z = self.agent_pos[2] + self.current_radius * np.cos(v)
        self.agent_surface = self.ax.plot_surface(x, y, z, color='red', alpha=0.3)

        for line in self.bounding_box_lines:
            line.remove()
        self.bounding_box_lines = []

        self.draw_bounding_box()

    def render(self, mode='human'):
        """更新渲染窗口中的绘图元素。"""

        if hasattr(self, 'agent_surface'):
            self.agent_surface.remove()

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.agent_pos[0] + self.current_radius * np.cos(u) * np.sin(v)
        y = self.agent_pos[1] + self.current_radius * np.sin(u) * np.sin(v)
        z = self.agent_pos[2] + self.current_radius * np.cos(v)
        self.agent_surface = self.ax.plot_surface(x, y, z, color='red', alpha=0.3)

        for line in self.bounding_box_lines:
            line.remove()
        self.bounding_box_lines = []

        self.draw_bounding_box()

        self.ax.set_title(f"Step: {self.current_step}, Radius: {self.current_radius}")
        plt.draw()
        plt.pause(0.001)

    def reset(self):
        self.points = self.voxel_downsample(self.load_pcd(self.pcd_file), voxel_size=self.voxel_size)
        if len(self.points) == 0:
            raise ValueError(f"点云文件 {self.pcd_file} 为空或未读取到数据！")
        self.bounding_box = self.compute_bounding_box()
        self.centroid = self.points.mean(axis=0)
        self.current_radius = self.initial_agent_radius
        self.agent_pos = self.centroid.copy()
        self.episode_num += 1
        print(f"==========开始回合 {self.episode_num}==========")
        self.segment_count = 0
        self.reset_render()
        self.current_step = 0
        self.init_grid_counter()
        hull_volume, num_inside = self.compute_reward(return_num_inside=True)
        self.prev_num_inside = num_inside
        grid_idx = self.get_grid_index(self.agent_pos)
        self.grid_counter[grid_idx] += 1
        self.guided_steps_remaining = 0
        self.current_guided_direction = None
        return self.agent_pos.copy()


    def step(self, action):
        """执行一步动作并返回新的状态、奖励、是否完成和附加信息。"""

        original_action = action.copy()

        if self.guided_steps_remaining > 0 and self.current_guided_direction is not None:

            desired_direction = self.current_guided_direction

            azimuth = math.atan2(desired_direction[1], desired_direction[0])
            elevation = math.atan2(desired_direction[2], np.linalg.norm(desired_direction[:2]))
            action = np.array([azimuth, elevation], dtype=np.float32)
            self.guided_steps_remaining -= 1

            if self.guided_steps_remaining == 0:
                self.current_guided_direction = None

        azimuth, elevation = action

        movement = self.action_distance * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ])


        new_pos = self.agent_pos + movement

        reward = 0.0

        min_xyz, max_xyz = self.bounding_box
        within_bounds = np.all(new_pos >= min_xyz) and \
                        np.all(new_pos <= max_xyz)

        if within_bounds:
            self.agent_pos = new_pos
        else:
            direction_to_centroid = self.centroid - self.agent_pos
            distance = np.linalg.norm(direction_to_centroid)
            if distance > 0:
                movement = self.action_distance * (direction_to_centroid / distance)
                self.agent_pos += movement

            reward -= self.large_penalty

        self.current_step += 1

        grid_idx = self.get_grid_index(self.agent_pos)
        self.grid_counter[grid_idx] += 1

        hull_volume, num_inside = self.compute_reward(return_num_inside=True)

        delta = num_inside - self.prev_num_inside

        normalized_hull_volume = hull_volume / self.max_hull_volume if self.max_hull_volume > 0 else 0
        normalized_num_inside = num_inside / self.max_num_inside if self.max_num_inside > 0 else 0
        normalized_delta = delta / self.max_num_inside if self.max_num_inside > 0 else 0


        reward += normalized_hull_volume + self.alpha * normalized_num_inside + self.beta * normalized_delta

        distances = np.linalg.norm(self.points - self.agent_pos, axis=1)
        inside_points = self.points[distances <= self.current_radius]

        if len(inside_points) > self.k_neighbors:
            try:

                normals = self.compute_normals(inside_points)

                cluster_labels, n_clusters, cluster_centers = self.cluster_normals(normals)

                if not (self.min_clusters <= n_clusters <= self.max_clusters):
                    valid_curvature = False
                else:

                    angles = []
                    for i in range(len(cluster_centers)):
                        for j in range(i + 1, len(cluster_centers)):
                            dot_product = np.dot(cluster_centers[i], cluster_centers[j])
                            dot_product = np.clip(dot_product, -1.0, 1.0)  # 防止数值误差
                            angle = np.degrees(np.arccos(dot_product))
                            angles.append(angle)

                    if len(angles) == 0:
                        valid_curvature = False
                    else:
                        min_detected_angle = min(angles)
                        # valid_curvature = min_detected_angle >= self.min_angle
                        valid_curvature = True  

                        if valid_curvature:

                            label_counts = Counter(cluster_labels)
                            cluster_proportions = [count / len(normals) for label, count in label_counts.items() if
                                                   label != -1]

                            valid_proportions = [prop for prop in cluster_proportions if prop >= self.min_proportion]

                            if len(valid_proportions) >= self.min_clusters:
                                exploration_reward = self.exploration_bonus / math.sqrt(self.grid_counter[grid_idx])
                                reward += exploration_reward

                                reward += self.gamma * (n_clusters / self.max_clusters)  
                                self.curvature_steps += 1
                            else:
                                self.curvature_steps = 0
                        else:
                            self.curvature_steps = 0
                            if 'min_detected_angle' in locals():
                                print(f"曲率条件未满足，最小夹角 {min_detected_angle:.2f}°")
                            else:
                                print("曲率条件未满足。")
            except Exception as e:
                print(f"计算法向量或聚类时发生错误: {e}")
                self.curvature_steps = 0
        else:
            self.curvature_steps = 0

        # ------------------- 状态访问惩罚 -------------------
        # 对于已访问的网格区域施加惩罚，鼓励智能体离开频繁访问的区域
        penalty = self.penalty_factor * self.grid_counter[grid_idx]
        reward -= penalty

        reward = np.clip(reward, 0, 1)

        if self.curvature_steps >= self.required_consecutive_steps:
            if self.current_radius > self.min_radius:
                self.current_radius = max(self.current_radius - self.radius_decrement, self.min_radius)
                self.curvature_steps = 0 

                done = False  
            else:
                if self.segment_count >= self.max_segments:
                    done = True
                else:
                    done = False
                    covered_points = inside_points
                    if len(covered_points) > 0:
                        save_dir = 'covered_points'
                        os.makedirs(save_dir, exist_ok=True)
                        self.segment_count += 1
                        filename = os.path.join(save_dir, f"points_episode_{self.episode_num}_{self.segment_count}.txt")
                        np.savetxt(filename, covered_points, delimiter=',')
                        print(f"回合： {self.episode_num}，分割次数：{self.segment_count}: 已保存 {len(covered_points)} 点到 {filename}.")

                        mask = distances > self.current_radius
                        self.points = self.points[mask]

                        self.scatter.remove()
                        if len(self.points) > 0:
                            self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                                                           c='blue', s=30, alpha=0.6, label='Points')
                        else:
                            print("所有点已被覆盖。")
                        self.bounding_box = self.compute_bounding_box()
                        self.centroid = self.points.mean(axis=0) if len(self.points) > 0 else np.array([0.0, 0.0, 0.0])

                        self.init_grid_counter()

                        self.current_radius = self.initial_agent_radius

                        if len(cluster_centers) >= 2:
                            normal1 = cluster_centers[0]
                            normal2 = cluster_centers[1]

                            perpendicular_direction = np.cross(normal1, normal2)
                            norm = np.linalg.norm(perpendicular_direction)
                            if norm > 0:
                                perpendicular_direction /= norm 

                                self.current_guided_direction = perpendicular_direction

                                self.guided_steps_remaining = self.guided_steps_after_segment

                            else:
                                print("法向量平行，无法计算垂直方向。")
                        else:
                            print("聚类中心不足以计算垂直方向。")
                        # ------------------------------------------------------------

                    else:
                        print("半径满足，曲率满足，分割出的点云不足。")

                    if self.segment_count >= self.max_segments:
                        done = True

            reward = self.curvature_reward  
        else:
            done = False  


        self.prev_num_inside = num_inside
        info = {}

        return self.agent_pos.copy(), reward, done, info

    def compute_normals(self, points):
        """计算点云中每个点的法向量。

        Args:
            points (np.ndarray): 点云数据，形状为 (N, 3)。

        Returns:
            np.ndarray: 法向量数组，形状为 (N, 3)。
        """
        normals = []
        for point in points:
            distances = np.linalg.norm(points - point, axis=1)
            if len(distances) <= self.k_neighbors:
                neighbors = points
            else:
                neighbor_indices = distances.argsort()[1:self.k_neighbors + 1]
                neighbors = points[neighbor_indices]
            pca = PCA(n_components=3)
            pca.fit(neighbors)
            normal = pca.components_[-1]  
            if np.dot(normal, np.array([0, 0, 1])) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)

    def cluster_normals(self, normals):
        """使用DBSCAN聚类法向量，识别不同的平面。

        Args:
            normals (np.ndarray): 法向量数组，形状为 (N, 3)。

        Returns:
            tuple: 聚类标签、聚类数量和聚类中心。
        """
        scaler = StandardScaler()
        normals_scaled = scaler.fit_transform(normals)

        dbscan = DBSCAN(eps=0.2, min_samples=5)  
        labels = dbscan.fit_predict(normals_scaled)

        unique_labels = set(labels)
        unique_labels.discard(-1)  
        n_clusters = len(unique_labels)

        cluster_centers = []
        for label in unique_labels:
            cluster_normals = normals[labels == label]
            if len(cluster_normals) > 0:
                mean_normal = cluster_normals.mean(axis=0)
                norm = np.linalg.norm(mean_normal)
                if norm > 0:
                    mean_normal /= norm
                cluster_centers.append(mean_normal)

        return labels, n_clusters, cluster_centers

    def check_curvature(self):
        """检查凸包表面是否存在满足特定曲率条件的面。

        通过计算法向量的聚类数量、聚类中心之间的夹角以及法向量一致性来判断是否存在多个平面交接处。

        Returns:
            bool: 是否满足曲率条件。
        """
        distances = np.linalg.norm(self.points - self.agent_pos, axis=1)
        inside_points = self.points[distances <= self.current_radius]

        if len(inside_points) < self.k_neighbors:
            return False 

        try:
            normals = self.compute_normals(inside_points)

            cluster_labels, n_clusters, cluster_centers = self.cluster_normals(normals)

            if not (self.min_clusters <= n_clusters <= self.max_clusters):
                print(f"未满足曲率条件：检测到 {n_clusters} 个法向量聚类")
                return False

            angles = []
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    dot_product = np.dot(cluster_centers[i], cluster_centers[j])
                    dot_product = np.clip(dot_product, -1.0, 1.0)  # 防止数值误差
                    angle = np.degrees(np.arccos(dot_product))
                    angles.append(angle)

            if len(angles) == 0:
                return False

            min_detected_angle = min(angles)
            if min_detected_angle < self.min_angle:
                print(f"未满足曲率条件：聚类中心之间最小夹角 {min_detected_angle:.2f}° 小于阈值")
                return False

            label_counts = Counter(cluster_labels)
            cluster_proportions = [count / len(normals) for label, count in label_counts.items() if label != -1]

            valid_proportions = [prop for prop in cluster_proportions if prop >= self.min_proportion]

            if len(valid_proportions) >= self.min_clusters:
                print(
                    f"满足曲率条件：检测到 {n_clusters} 个聚类，且至少 {self.min_clusters} 个聚类的比例 >= {self.min_proportion * 100}%")
                return True
            else:
                print(f"未满足曲率条件：只有 {len(valid_proportions)} 个聚类的比例 >= {self.min_proportion * 100}%")
                return False

        except Exception as e:
            print(f"检查曲率时发生错误: {e}")
            return False

    def compute_reward(self, return_num_inside=False):
        """计算奖励。

        Args:
            return_num_inside (bool): 是否返回内部点数。

        Returns:
            float or tuple: 凸包体积，若 return_num_inside 为 True，则返回 (凸包体积, 内部点数)。
        """
        distances = np.linalg.norm(self.points - self.agent_pos, axis=1)
        inside_points = self.points[distances <= self.current_radius]

        num_inside = len(inside_points)

        if num_inside < 4:
            hull_volume = 0.0
        else:
            try:
                hull = ConvexHull(inside_points)
                hull_volume = hull.volume
            except:
                hull_volume = 0.0

        if return_num_inside:
            return hull_volume, num_inside
        else:
            return hull_volume

    def close(self):
        """关闭环境时关闭渲染窗口。"""
        plt.ioff()
        # plt.show()

    def __del__(self):
        """析构函数，确保渲染窗口关闭。"""
        plt.ioff()
        # plt.show()
