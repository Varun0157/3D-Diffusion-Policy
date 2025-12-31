import gym
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import pybullet as p
import pybullet_data
import cv2
from termcolor import cprint
from gym import spaces
import random


def compute_workspace_bounds(pc_xyz, n_std=2):
    """
    Compute workspace bounds using mean ± n_std * std

    Args:
        pc_xyz: (N, 3) numpy array of point cloud coordinates
        n_std: number of standard deviations for bounds

    Returns:
        WORK_SPACE: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    if pc_xyz.shape[0] == 0:
        return None

    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    WORK_SPACE = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],  # X
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],  # Y
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]]   # Z
    ]

    return WORK_SPACE


def crop_workspace(pc_xyz, workspace_bounds):
    """
    Crop point cloud to workspace bounds

    Args:
        pc_xyz: (N, 3) numpy array
        workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    Returns:
        mask: boolean array of valid points
    """
    if workspace_bounds is None:
        return np.ones(pc_xyz.shape[0], dtype=bool)

    mask = (
        (pc_xyz[:, 0] >= workspace_bounds[0][0]) & (pc_xyz[:, 0] <= workspace_bounds[0][1]) &
        (pc_xyz[:, 1] >= workspace_bounds[1][0]) & (pc_xyz[:, 1] <= workspace_bounds[1][1]) &
        (pc_xyz[:, 2] >= workspace_bounds[2][0]) & (pc_xyz[:, 2] <= workspace_bounds[2][1])
    )

    return mask


def downsample_with_fps(points: np.ndarray, num_points: int = 2500):
    """Fast point cloud sampling using torch3d"""
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points_tensor = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[..., :3], K=num_points_tensor)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

import numpy as np

def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, width=224, height=224,
                         fov=60, near=0.01, far=3.0, max_depth=2.5):
    """
    Convert a PyBullet depth buffer to a 3D point cloud in world coordinates,
    optionally filtering points beyond max_depth.

    Args:
        depth_buffer: depth buffer from PyBullet (H x W)
        view_matrix: PyBullet camera view matrix (16-element list)
        proj_matrix: PyBullet projection matrix (16-element list)
        width: image width
        height: image height
        fov: field of view (degrees)
        near: near clipping plane
        far: far clipping plane
        max_depth: maximum distance to keep points (in meters)
    """
    # Convert depth buffer to linear depth
    depth_img = far * near / (far - (far - near) * depth_buffer)

    # Camera intrinsics
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2

    # Pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 3D points in camera frame
    z = depth_img
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy  # negative to flip y-axis
    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Transform to world frame
    points_hom = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
    view_matrix_np = np.array(view_matrix).reshape(4, 4).T
    points_world = (np.linalg.inv(view_matrix_np) @ points_hom.T).T[:, :3]

    # Filter points beyond max_depth
    valid_mask = points_world[:, 2] < max_depth
    points_world = points_world[valid_mask]

    return points_world



class UR5Robotiq85:
    """UR5 Robot with Robotiq 85 Gripper"""
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

    def load(self):
        # self.id = p.loadURDF('/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.id = p.loadURDF('/home/cross-emb/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        from collections import namedtuple
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        import math
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def get_robot_state(self):
        """Get complete robot state: end-effector pose + joint angles"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        eef_pos = np.array(eef_state[0])
        eef_orn_quat = np.array(eef_state[1])
        eef_orn_euler = np.array(p.getEulerFromQuaternion(eef_orn_quat))

        joint_states = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_states.append(joint_state[0])

        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        gripper_angle = gripper_state[0]

        state = np.concatenate([
            eef_pos,
            eef_orn_euler,
            joint_states,
            [gripper_angle]

        ])

        return state

    def get_joint_positions(self):
        """Get current joint positions (6 arm joints + 1 gripper)"""
        joint_positions = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_positions.append(joint_state[0])

        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        joint_positions.append(gripper_state[0])

        return np.array(joint_positions)

    def get_joint_limits(self):
        """Get joint limits for arm joints and gripper"""
        limits_lower = self.arm_lower_limits + [self.gripper_range[0]]
        limits_upper = self.arm_upper_limits + [self.gripper_range[1]]
        return np.array(limits_lower), np.array(limits_upper)

    def set_joint_positions(self, joint_positions):
        """
        Set joint positions using high-gain position control
        """
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id, joint_id, 
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500,  # High force
                maxVelocity=10  # High velocity
            )

        if len(joint_positions) > self.arm_num_dofs:
            gripper_angle = joint_positions[self.arm_num_dofs]
            p.setJointMotorControl2(
                self.id, self.mimic_parent_id,
                p.POSITION_CONTROL,
                targetPosition=gripper_angle,
                force=100
            )
        # Ignoring physics and directly setting it !!
    # # Set arm joints DIRECTLY (no physics, instant teleport)
    #     for i, joint_id in enumerate(self.arm_controllable_joints):
    #         p.resetJointState(self.id, joint_id, joint_positions[i])

    #     # Set gripper
    #     if len(joint_positions) > self.arm_num_dofs:
    #         gripper_angle = joint_positions[self.arm_num_dofs]
    #         p.resetJointState(self.id, self.mimic_parent_id, gripper_angle)
            
    #         # Also set mimic joints
    #         for joint_id, multiplier in self.mimic_child_multiplier.items():
    #             p.resetJointState(self.id, joint_id, gripper_angle * multiplier)

    
class UR5PickPlaceEnv(gym.Env):
    """
    PyBullet UR5 Pick and Place Environment

    This environment follows the Gym interface and is compatible with
    MultiStepWrapper for multi-step action execution.

    Actions are interpreted as DELTAS in joint angles (6 arm joints + 1 gripper).
    """
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, use_gui=False, num_points=2500, image_size=224, use_workspace_crop=True, workspace_std=2.0):
        self.use_gui = use_gui
        self.num_points = num_points
        self.image_size = image_size
        self.max_steps = 350
        self.current_step = 0
        self.use_workspace_crop = use_workspace_crop
        self.workspace_std = workspace_std
        self.workspace_bounds = None  # Will be computed from first observation

        # Connect to PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.tray_pos = [0.5, 0.9, 0.6]
        self.tray_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.tray_id = p.loadURDF("tray/tray.urdf", self.tray_pos, self.tray_orn)

        # Load robot
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Camera parameters
        self.tp_cam_eye = [1.1, -0.6, 1.3]
        self.tp_cam_target = [0.5, 0.3, 0.7]
        self.tp_cam_up = [0, 0, 1]

        self.cube_id = None
        self.cube_start_pos = None

        # Define action and observation spaces
        # Action: 13D vector where:
        #   - First 6D (eef_pos(3) + eef_orn(3)): Not used in this version
        #   - Last 7D (arm_joints(6) + gripper(1)): DELTA joint angles
        # Note: MultiStepWrapper will repeat this to create multi-step actions
        self.action_space = spaces.Box(
            low=-0.1,  # Maximum delta per step
            high=0.1,
            shape=(13,),
            dtype=np.float32
        )

        # Observation space: Single timestep observations
        # Note: MultiStepWrapper will stack these to create multi-step observations
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(13,),
                dtype=np.float32
            ),
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.uint8
            ),
        })

        self.is_success_flag = False

    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.is_success_flag = False

        # Reset workspace bounds (will be recomputed)
        self.workspace_bounds = None

        # Remove old cube if exists
        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        # Reset robot to rest pose
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL,
                                   target_joint_positions[i])

        # Step simulation to stabilize
        for _ in range(100):
            p.stepSimulation()

        # Spawn cube at random location
        self.cube_start_pos = [
            random.uniform(0.3, 0.7),
            random.uniform(-0.1, 0.1),
            0.65
        ]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_pos, cube_start_orn)

        # Random cube color
        color = [random.random(), random.random(), random.random(), 1.0]
        p.changeVisualShape(self.cube_id, -1, rgbaColor=color)

        # Step simulation
        for _ in range(50):
            p.stepSimulation()

        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        Execute action and return observation.

        Action is 13D: [eef_pos(3), eef_orn(3), joint_deltas(6), gripper_delta(1)]
        - First 6D (eef_pos + eef_orn): Not used
        - Last 7D: DELTA joint angles applied to current joint positions

        Note: MultiStepWrapper will call this multiple times per step,
        so each call should execute a single atomic action.

        Args:
            action: 13D action vector (only last 7D are used as deltas)
        """
        self.current_step += 1

        joint_deltas = action[6:13]  # Last 7D: delta for 6 arm joints + 1 gripper

        # Get current joint positions
        current_joint_positions = self.robot.get_joint_positions()
        # print("Current joint positions:", current_joint_positions)

        # Apply delta to get target positions
        target_joint_positions = current_joint_positions + joint_deltas
        print("Target joint positions:", target_joint_positions)
        # Get joint limits
        joint_limits_lower, joint_limits_upper = self.robot.get_joint_limits()

        # # Clip to joint limits
        # target_joint_positions = np.clip(
        #     target_joint_positions,
        #     joint_limits_lower,
        #     joint_limits_upper
        # )

        
        self.robot.set_joint_positions(target_joint_positions)

        """
        We have 2 options here:
        1. Keep the sim time high and use physics (setJointMotorControl2)
        2. Lower the sim time step and directly set joint states (resetJointState)
        """

        for _ in range(50):
            p.stepSimulation()

        reached_pos = self.robot.get_joint_positions()
        # Compute difference
        joint_error = reached_pos - target_joint_positions
        joint_error_norm = np.linalg.norm(joint_error)

        # ---- Debug prints ----
        print("‖Joint error‖ (L2 norm)  :", joint_error_norm)
        print("-" * 50)

        obs = self._get_obs()

        # Check success: cube in tray
        reward = 0.0
        done = False

        if self.cube_id is not None:
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            # Check if cube is in tray region
            if (abs(cube_pos[0] - self.tray_pos[0]) < 0.2 and
                abs(cube_pos[1] - self.tray_pos[1]) < 0.2 and
                abs(cube_pos[2] - self.tray_pos[2]) < 0.1):
                reward = 1.0
                self.is_success_flag = True
                done = True

        # Episode timeout
        if self.current_step >= self.max_steps:
            done = True

        info = {'is_success': self.is_success_flag}

        return obs, reward, done, info

    def _get_obs(self):
        """Get current observation"""
        # Get robot state
        agent_pos = self.robot.get_robot_state()

        # Capture camera image and point cloud
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        width, height, rgb_img, depth_img, _ = p.getCameraImage(
            self.image_size, self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )

        rgb_img = np.array(rgb_img)[:, :, :3]  # (H, W, 3)
        depth_buffer = np.array(depth_img)

        # Convert to point cloud
        point_cloud = depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix,
                                          self.image_size, self.image_size)

        # Apply workspace cropping if enabled
        if self.use_workspace_crop:
            # Compute workspace bounds on first observation
            if self.workspace_bounds is None:
                self.workspace_bounds = compute_workspace_bounds(point_cloud, n_std=self.workspace_std)
                if self.workspace_bounds is not None:
                    print(f"Computed workspace bounds (mean ± {self.workspace_std}*std):")
                    print(f"  X: [{self.workspace_bounds[0][0]:.3f}, {self.workspace_bounds[0][1]:.3f}]")
                    print(f"  Y: [{self.workspace_bounds[1][0]:.3f}, {self.workspace_bounds[1][1]:.3f}]")
                    print(f"  Z: [{self.workspace_bounds[2][0]:.3f}, {self.workspace_bounds[2][1]:.3f}]")

            # Crop point cloud to workspace
            if self.workspace_bounds is not None:
                mask = crop_workspace(point_cloud, self.workspace_bounds)
                point_cloud = point_cloud[mask]

                # Debug: print how many points remain
                if self.current_step == 0:
                    print(f"  After workspace crop: {point_cloud.shape[0]} points")

        # Handle empty point cloud
        if point_cloud.shape[0] == 0:
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
        # Downsample point cloud
        elif point_cloud.shape[0] > self.num_points:
            point_cloud = downsample_with_fps(point_cloud, self.num_points)
        elif point_cloud.shape[0] < self.num_points:
            # Pad if fewer points
            padding = np.zeros((self.num_points - point_cloud.shape[0], 3))
            point_cloud = np.vstack([point_cloud, padding])

        # Transpose image to channel-first (3, H, W)
        rgb_img = rgb_img.transpose(2, 0, 1)

        obs_dict = {
            'point_cloud': point_cloud.astype(np.float32),
            'agent_pos': agent_pos.astype(np.float32),
            'image': rgb_img.astype(np.uint8),
        }

        return obs_dict

    def render(self, mode='rgb_array'):
        """Render the environment"""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        width, height, rgb_img, _, _ = p.getCameraImage(
            self.image_size, self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )

        rgb_img = np.array(rgb_img)[:, :, :3]
        return rgb_img.astype(np.uint8)

    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)

    def is_success(self):
        """Check if task was successful"""
        return self.is_success_flag

    def seed(self, seed=None):
        """Set random seed"""
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)