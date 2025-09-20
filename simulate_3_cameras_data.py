import numpy as np
import pandas as pd
import cv2
import json
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def simulate_apriltag_board(tag_rows=6, tag_cols=6, tag_size=0.13, tag_spacing=0.04):
    """Generate a flat list of 3D object points for an AprilTag board."""
    obj_points = []
    
    for row in range(tag_rows):
        for col in range(tag_cols):
            tag_x = col * (tag_size + tag_spacing)
            tag_y = row * (tag_size + tag_spacing)
            tag_z = 0  # Assume the board is in the XY plane

            # Define the four corner points in (X, Y, Z) and flatten into one list
            obj_points.extend([
                [tag_x, tag_y, tag_z],  # Top-left
                [tag_x + tag_size, tag_y, tag_z],  # Top-right
                [tag_x + tag_size, tag_y + tag_size, tag_z],  # Bottom-right
                [tag_x, tag_y + tag_size, tag_z]  # Bottom-left
            ])
    
    obj_points = np.array(obj_points, dtype=np.float32)  # Shape: (N, 3), where N = tag_rows * tag_cols * 4
    # print("obj_points function shape: ", obj_points.shape)
    return obj_points


def save_generated_data_to_csv(file_path, timestamp_ns, img_points, corner_ids, cam_id):
    """
    Save the generated image points to a CSV file, appending if the file exists.
    """
    if len(img_points) <= 10:
        return
    radii = np.random.normal(loc=2.0, scale=0.4, size=len(img_points))
    radii = np.round(radii, 2)
    img_points = np.round(img_points, 2)
    # print("img_points: ", img_points)


    data = {
        "timestamp_ns": [timestamp_ns] * len(img_points),
        "cam_id": [cam_id] * len(img_points),
        "corner_id": corner_ids,
        "corner_x": img_points[:, 0],
        "corner_y": img_points[:, 1],
        "radius": radii,
    }
    df = pd.DataFrame(data)

    # print("df =/n", df)
    # print("file_path = ", file_path)
    
    # Append to CSV file if it exists, otherwise create a new one with headers
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)


def generate_corner_ids(tag_rows, tag_cols):
    """Generates a flat list of unique corner IDs for an AprilTag board."""
    corner_ids = []
    tag_id = 0

    for _ in range(tag_rows):
        for _ in range(tag_cols):
            # Each AprilTag has 4 corners, stored sequentially
            corner_ids.extend([
                tag_id * 4 + 0,  # Top-left
                tag_id * 4 + 1,  # Top-right
                tag_id * 4 + 2,  # Bottom-right
                tag_id * 4 + 3   # Bottom-left
            ])
            tag_id += 1

    corner_ids = np.array(corner_ids, dtype=np.int32)  # Shape: (N,), where N = tag_rows * tag_cols * 4
    # print("corner_ids function shape: ", corner_ids.shape)
    return corner_ids


def generate_synthetic_cameras():
    """Generate synthetic intrinsics for three cameras."""
    K1 = np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]])
    K2 = np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]])
    K3 = np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]])
    dist1 = np.array([-0.04, 0.03, -0.04, 0.015]) 
    dist2 = np.array([-0.04, 0.03, -0.04, 0.015]) 
    dist3 = np.array([-0.04, 0.03, -0.04, 0.015]) 
    # dist1 = np.array([-0.45, 0.17, -0.03, 0.004]) #np.array([-0.28, 0.07, -0.009, 0.0001]) #[0.1, -0.05, 0.02, -0.01]) 
    # dist2 = np.array([-0.45, 0.17, -0.03, 0.004]) #[0.09, -0.04, 0.015, -0.008]) 
    # dist1 = np.array([-0.45, 0.17, -0.03, 0.004]) 
    # dist2 = np.array([-0.45, 0.17, -0.03, 0.004]) 
    return (K1, dist1), (K2, dist2), (K3, dist3)


def project_points_kannala_brandt(obj_points_cam, K, dist_coeffs):
    """Project 3D object points to 2D using the Kannala-Brandt model."""
    # transformed_pts = (R_matrix @ obj_points_world.T).T + tvec
    # X, Y, Z = transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2]
    X, Y, Z = obj_points_cam[:, 0], obj_points_cam[:, 1], obj_points_cam[:, 2]
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)
    # print("r: ", r)
    # print("Z: ", Z)
    # print("theta: ", theta)
    
    k1, k2, k3, k4 = dist_coeffs
    theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    # print("theta_d: ", theta_d)
    epsilon = 0.0001  # Small value to avoid division by zero
    if np.any(r <= epsilon):
        print("epsilon triggered for small r values")

    scale = np.where(r > epsilon, theta_d / r, 1.0)  # Default to 1.0 when r == 0
    # print("scale: ", scale)
    
    if (False):
        scale = 1/Z
        scale = np.where(Z > 0.001, 1/Z, 1000)  # Default to 1.0 when r == 0
        print("bypass distortion: scale = 1/Z\n")
        # print(scale)S
    x_distorted, y_distorted = X * scale, Y * scale
    # print("x_distorted: ", x_distorted)
    # print("y_distorted: ", y_distorted)
    # print("X: ", X)
    # print("Y: ", Y)
    
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # print("fx, fy, cx, cy: ", fx, fy, cx, cy)
    u, v = fx * x_distorted + cx, fy * y_distorted + cy
    return np.column_stack((u, v))



def add_noise(img_points, noise_std=1.0):
    """Add Gaussian noise to image points."""
    return img_points + np.random.normal(scale=noise_std, size=img_points.shape)


def plot_three_camera_pts(R_1, t_1, R_2, t_2, R_3, t_3, filtered_img_pts_1, filtered_ids_1, filtered_img_pts_2, filtered_ids_2, filtered_img_pts_3, filtered_ids_3, obj_corner_ids, obj_points_world):
    # # Define a board pose in front of the cameras
    # R_board = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    # t_board = np.array([0.0, 0.0, 1.0])  # 1m in front

    # obj_points_world = (R_board @ obj_points.T).T + t_board
    obj_points_cam_2 = (np.linalg.inv(R_2) @ (obj_points_world - t_2).T).T
    obj_points_cam_3 = (np.linalg.inv(R_3) @ (obj_points_world - t_3).T).T

    # R1_trans = R1 @ R_board
    # R2_trans = R2 @ R_board
    # t1_trans = R1 @ t_board + t1
    # t2_trans = R2 @ t_board + t2

    # # Project points for both cameras
    # img_pts_1, obj_pts_cam_1 = project_points(obj_points, K_1_gt, R1_trans, t1_trans)
    # img_pts_2, obj_pts_cam_2 = project_points(obj_points, K_1_gt, R2_trans, t2_trans)

    # Visualization
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 3D Plot
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    ax.scatter(obj_points_world[:, 0], obj_points_world[:, 1], obj_points_world[:, 2], c='b', label='obj_points_world')
    ax.scatter(obj_points_cam_2[:, 0], obj_points_cam_2[:, 1], obj_points_cam_2[:, 2], c='g', label='cam_2_points')
    ax.scatter(obj_points_cam_3[:, 0], obj_points_cam_3[:, 1], obj_points_cam_3[:, 2], c='m', label='cam_3_points')
    ax.scatter(t_1[0], t_1[1], t_1[2], c='r', marker='o', label='Camera 1')
    ax.scatter(t_2[0], t_2[1], t_2[2], c='g', marker='o', label='Camera 2')
    ax.scatter(t_3[0], t_3[1], t_3[2], c='m', marker='o', label='Camera 3')
    # print("t_1: ", t_1)
    # print("t_2: ", t_2)

    # Draw camera orientations
    cam_z_1 = R_1 @ np.array([0, 0, 2])
    cam_x_1 = R_1 @ np.array([2, 0, 0])
    ax.quiver(t_1[0], t_1[1], t_1[2], cam_z_1[0], cam_z_1[1], cam_z_1[2], color='r', length=0.1)
    ax.quiver(t_1[0], t_1[1], t_1[2], cam_x_1[0], cam_x_1[1], cam_x_1[2], color='m', length=0.1)
    cam_z_2 = R_2 @ np.array([0, 0, 2])
    cam_x_2 = R_2 @ np.array([2, 0, 0])
    ax.quiver(t_2[0], t_2[1], t_2[2], cam_z_2[0], cam_z_2[1], cam_z_2[2], color='g', length=0.1)
    ax.quiver(t_2[0], t_2[1], t_2[2], cam_x_2[0], cam_x_2[1], cam_x_2[2], color='c', length=0.1)
    cam_z_3 = R_3 @ np.array([0, 0, 2])
    cam_x_3 = R_3 @ np.array([2, 0, 0])
    ax.quiver(t_3[0], t_3[1], t_3[2], cam_z_3[0], cam_z_3[1], cam_z_3[2], color='k', length=0.1)
    ax.quiver(t_3[0], t_3[1], t_3[2], cam_x_3[0], cam_x_3[1], cam_x_3[2], color='y', length=0.1)

    for i, corner_id in enumerate(obj_corner_ids):
        ax.text(obj_points_world[i, 0], obj_points_world[i, 1], obj_points_world[i, 2], f'{corner_id}', size=10, color='black')
        ax.text(obj_points_cam_2[i, 0], obj_points_cam_2[i, 1], obj_points_cam_2[i, 2], f'{corner_id}', size=10, color='black')
        ax.text(obj_points_cam_3[i, 0], obj_points_cam_3[i, 1], obj_points_cam_3[i, 2], f'{corner_id}', size=10, color='black')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-0.5, 2)
    ax.set_ylim3d(-0.5, 2)
    ax.set_zlim3d(-0.5, 2)
    ax.set_title("3D Visualization of Cameras and Object Points")
    # ax.legend()

    # 2D Image Projections
    ax2 = fig.add_subplot(gs[0, 1])
    if len(filtered_img_pts_1) > 0:
        ax2.scatter(filtered_img_pts_1[:, 0], filtered_img_pts_1[:, 1], c='r', label='Cam 1 Projection')
        for i, corner_id in enumerate(filtered_ids_1):
            ax2.text(filtered_img_pts_1[i, 0], filtered_img_pts_1[i, 1], f'{corner_id}', size=10, color='black')
    ax2.set_xlim(0, 1280)
    ax2.set_ylim(0, 960)  # Invert y-axis for image coordinates
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title("2D Projections in Camera 1 Frame")
    ax2.set_xlabel("Image X")
    ax2.set_ylabel("Image Y")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    if len(filtered_img_pts_2) > 0:
        ax3.scatter(filtered_img_pts_2[:, 0], filtered_img_pts_2[:, 1], c='g', label='Cam 2 Projection')
        for i, corner_id in enumerate(filtered_ids_2):
            ax3.text(filtered_img_pts_2[i, 0], filtered_img_pts_2[i, 1], f'{corner_id}', size=10, color='black')
    ax3.set_xlim(0, 1280)
    ax3.set_ylim(0, 960)  # Invert y-axis for image coordinates
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title("2D Projections in Camera 2 Frame")
    ax3.set_xlabel("Image X")
    ax3.set_ylabel("Image Y")
    ax3.legend()

    ax4 = fig.add_subplot(gs[2, 1])
    if len(filtered_img_pts_3) > 0:
        ax4.scatter(filtered_img_pts_3[:, 0], filtered_img_pts_3[:, 1], c='m', label='Cam 3 Projection')
        for i, corner_id in enumerate(filtered_ids_3):
            ax4.text(filtered_img_pts_3[i, 0], filtered_img_pts_3[i, 1], f'{corner_id}', size=10, color='black')
    ax4.set_xlim(0, 1280)
    ax4.set_ylim(0, 960)  # Invert y-axis for image coordinates
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title("2D Projections in Camera 3 Frame")
    ax4.set_xlabel("Image X")
    ax4.set_ylabel("Image Y")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_filtered_points(obj_points_world, filtered_obj_pts, filtered_img_pts, corner_ids, filtered_corner_ids, img_size):
    """
    Visualizes the 3D points, color codes them by whether they are filtered, and then projects the filtered points 
    onto the image plane (2D sensor coordinates).
    """
    # 3D Plotting of points and color coding based on filtering
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # Plot the 3D space with the points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(obj_points_world[:, 0], obj_points_world[:, 1], obj_points_world[:, 2], c='r', label='All Points')
    ax.scatter(filtered_obj_pts[:, 0], filtered_obj_pts[:, 1], filtered_obj_pts[:, 2], c='g', label='Filtered Points')
    
    # Labeling the points with their corner IDs
    for i, corner_id in enumerate(corner_ids):
        ax.text(obj_points_world[i, 0], obj_points_world[i, 1], obj_points_world[i, 2], f'{corner_id}', size=10, color='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points with IDs (Red = All, Green = Filtered)')
    ax.legend()

    # 2D Projection of the filtered points onto the camera sensor
    ax2 = fig.add_subplot(122)
    
    # Plot filtered points projected onto the 2D image sensor
    ax2.scatter(filtered_img_pts[:, 0], filtered_img_pts[:, 1], c='g', label='Filtered Points')

    # Label the 2D projected points
    for i, corner_id in enumerate(filtered_corner_ids):
        ax2.text(filtered_img_pts[i, 0], filtered_img_pts[i, 1], f'{corner_id}', size=10, color='black')

    width, height = img_size
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_aspect('equal')

    ax2.set_xlabel('X (Image Sensor)')
    ax2.set_ylabel('Y (Image Sensor)')
    ax2.set_title('2D Projection of Filtered Points')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def filter_visible_points(obj_points_world, K, dist_coeffs, R_cam, t_cam, img_size, corner_ids):
    """
    Filters visible AprilTag corners by projecting them onto the image plane and removing points outside the image bounds.
    """
    width, height = img_size
    # print("img_size: ", img_size)
    # print("obj_points_world: ", obj_points_world)

    # Transform 3D points to the camera coordinate frame
    obj_pts_cam = (np.linalg.inv(R_cam) @ (obj_points_world - t_cam).T).T  # Shape: (N, 3)
    # print("obj_pts_cam: ", obj_pts_cam)

    # Filter out points behind the camera (Z <= 0)
    valid_z_indices = obj_pts_cam[:, 2] > 0.1
    # print("valid_z_indices shape: ", valid_z_indices.shape)
    # print ("valid_z_indices: ", valid_z_indices)
    # print("obj_pts_cam", obj_pts_cam)

    valid_obj_pts_cam = obj_pts_cam[valid_z_indices]
    valid_z_corner_ids = corner_ids[valid_z_indices]  #Ensure we keep only matching IDs


    if len(valid_obj_pts_cam) == 0:
        return np.array([]), np.array([]), []

    # Project points using Kannala-Brandt model
    img_pts = project_points_kannala_brandt(valid_obj_pts_cam, K, dist_coeffs)
    # print("img_pts: ", img_pts)
    
    # Filter points inside image bounds
    inside_x = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < width)
    inside_y = (img_pts[:, 1] >= 0) & (img_pts[:, 1] < height)
    inside_image = inside_x & inside_y

    filtered_obj_pts = valid_obj_pts_cam[inside_image]
    filtered_img_pts = img_pts[inside_image]
    filtered_corner_ids = valid_z_corner_ids[inside_image]  #Correctly filtered corner IDs
    # plot_filtered_points(obj_points_world, filtered_obj_pts, filtered_img_pts, corner_ids, filtered_corner_ids, img_size)


    return filtered_obj_pts, filtered_img_pts, filtered_corner_ids


def save_calibration_result(filepath,
                            K1, dist1,
                            K2, dist2,
                            K3, dist3,
                            R1, t1,
                            R2, t2,
                            R3, t3,
                            target_poses, timestamps):
    """
    Save calibration result in the JSON format expected by LoadCalibrationResult().
    """
    def intrinsics_to_vec(K):
        # fx, fy, cx, cy
        return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

    def rotation_to_quat(R_mat):
        # SciPy gives [x, y, z, w] → convert to [w, x, y, z]
        # print("R_mat in rotation_to_quat: ", R_mat)
        q = R.from_matrix(R_mat).as_quat()
        # print("q in rotation_to_quat: ", q)
        #if any of the values are within 1e-4 of 0.948516, print the matrix
        # if any(abs(v - 0.948516) < 1e-4 for v in q):
            # print("R_mat: ", R_mat)
            # print("q: ", q)
            # input()
        return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    # Camera intrinsics/distortion
    data = {
        "camera0": {
            "intrinsics": intrinsics_to_vec(K1),
            "distortion": [float(v) for v in dist1]
        },
        "camera1": {
            "intrinsics": intrinsics_to_vec(K2),
            "distortion": [float(v) for v in dist2]
        },
        "camera2": {
            "intrinsics": intrinsics_to_vec(K3),
            "distortion": [float(v) for v in dist3]
        },
        "target_poses": [],
        "inter_camera": {
            "camera1_to_camera0": {},
            "camera2_to_camera0": {}
        }
    }

    # --- Target poses (world to board) ---
    for i, (R_mat, t_vec) in enumerate(target_poses):
        qvec = rotation_to_quat(R_mat)
        data["target_poses"].append({
            "quaternion": qvec,
            "translation": [float(v) for v in t_vec],
            "timestamp": float(timestamps[i])
        })

    # --- Inter-camera transforms ---
    # cam1->cam0
    R_10 = R1.T @ R2
    t_10 = R1.T @ (t2 - t1)
    data["inter_camera"]["camera1_to_camera0"]["quaternion"] = rotation_to_quat(R_10)
    data["inter_camera"]["camera1_to_camera0"]["translation_vector"] = [float(v) for v in t_10]

    # cam2->cam0
    R_20 = R1.T @ R3
    t_20 = R1.T @ (t3 - t1)
    data["inter_camera"]["camera2_to_camera0"]["quaternion"] = rotation_to_quat(R_20)
    data["inter_camera"]["camera2_to_camera0"]["translation_vector"] = [float(v) for v in t_20]

    # Write JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def is_monotonic_kb(dist_coeffs, theta_max, samples=2000):
    k1, k2, k3, k4 = dist_coeffs
    theta = np.linspace(0, theta_max, samples)
    deriv = 1 + 3*k1*theta**2 + 5*k2*theta**4 + 7*k3*theta**6 + 9*k4*theta**8
    return np.all(deriv > 0), deriv.min()

def generate_synthetic_data(K_1, dist_coeffs_1, K_2, dist_coeffs_2, K_3, dist_coeffs_3, R1, t1, R2, t2, R3, t3, obj_points, img_size, rpy_range, xyz_range, num_samples, output_dir, board_size):
    headers = "timestamp_ns, corner_id, cam_id, corner_x, corner_y, radius\n"
    file_path = f"{output_dir}/synthetic_data_3_cams.csv"
    with open(file_path, 'w') as file:
        file.write(headers)

    tag_rows, tag_cols = board_size
    corner_ids = generate_corner_ids(tag_rows, tag_cols)

    target_poses = []
    timestamps = []

    # Example parameters
    radius_values = np.linspace(*xyz_range['r'], 3)
    start_angle_deg = 0.0    # x=0, z=+1
    stop_angle_deg = 90.0    # x=+1, z=0
    num_arc_points = 5
    y_values = np.linspace(*xyz_range['y'], 3)

    angles = np.radians(np.linspace(*xyz_range['t'], 3))
    # radius_values = [0.5, 1, 2]
    # y_values = [-0.6]
    # angles = [np.radians(30)]
    timestamp_ns = 0
    for radius in radius_values:
        # Arc sweep in x-z plane (y handled separately)
        x_values = radius * np.sin(angles)   # x=0 at 0°
        z_values = radius * np.cos(angles)   # z=radius at 0°

        for tx, tz in zip(x_values, z_values):
            for ty in y_values:

    # x_values = np.linspace(*xyz_range['x'], 3)
    # y_values = np.linspace(*xyz_range['y'], 3)
    # z_values = np.linspace(*xyz_range['z'], 2)
    # # y_values = np.array([0.0])
    # # z_values = np.array([2.0])
    # timestamp_ns = 0

    # for tx in x_values:
    #     for ty in y_values:
    #         for tz in z_values:
                roll = np.radians(5)
                pitch = np.radians(30)
                yaw = np.radians(5)
                # roll = np.radians(np.random.uniform(*rpy_range['roll']))
                # pitch = np.radians(np.random.uniform(*rpy_range['pitch']))
                # yaw = np.radians(np.random.uniform(*rpy_range['yaw']))
                R_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
                tvec = np.array([tx, ty, tz])
                # print("obj_points: ", obj_points)

                obj_points_world = (R_matrix @ obj_points.T).T + tvec
                # print obj_points_world points
                # print("obj_points_world: ", obj_points_world)
                q = R.from_matrix(R_matrix).as_quat()
                # print("q: ", q)
                # print("tvec: ", tvec)
                # print("R_matrix: ", R_matrix)

                # input()


                filtered_obj_pts_1, filtered_img_pts_1, filtered_corner_ids_1 = filter_visible_points(
                    obj_points_world, K_1, dist_coeffs_1, R1, t1, img_size, corner_ids
                )
                filtered_obj_pts_2, filtered_img_pts_2, filtered_corner_ids_2 = filter_visible_points(
                    obj_points_world, K_2, dist_coeffs_2, R2, t2, img_size, corner_ids
                )
                filtered_obj_pts_3, filtered_img_pts_3, filtered_corner_ids_3 = filter_visible_points(
                    obj_points_world, K_3, dist_coeffs_3, R3, t3, img_size, corner_ids
                )


                save_generated_data_to_csv(file_path, timestamp_ns, filtered_img_pts_1, filtered_corner_ids_1, 0)
                
                save_generated_data_to_csv(file_path, timestamp_ns, filtered_img_pts_2, filtered_corner_ids_2, 1)

                save_generated_data_to_csv(file_path, timestamp_ns, filtered_img_pts_3, filtered_corner_ids_3, 2)

                # plot_three_camera_pts(R1, t1, R2, t2, R3, t3, filtered_img_pts_1, filtered_corner_ids_1, filtered_img_pts_2, filtered_corner_ids_2, filtered_img_pts_3, filtered_corner_ids_3, corner_ids, obj_points_world)
               
                
                # only append timestamp if at least one camera has valid points
                if len(filtered_img_pts_1) > 0 or len(filtered_img_pts_2) > 0 or len(filtered_img_pts_3) > 0:
                    # print lenghth of filtered points
                    # print("Filtered points in camera 1: ", len(filtered_img_pts_1))
                    # print("Filtered points in camera 2: ", len(filtered_img_pts_2))
                    # print("Filtered points in camera 3: ", len(filtered_img_pts_3))
                    # print("timestamp_ns: ", timestamp_ns)
                    target_poses.append((R_matrix, tvec))
                    timestamps.append(timestamp_ns)

                timestamp_ns += 1

    file_path_json = f"{output_dir}/synthetic_calibration.json"
    save_calibration_result(
        file_path_json,
        K_1, dist_coeffs_1,
        K_2, dist_coeffs_2,
        K_3, dist_coeffs_3,
        R1, t1, R2, t2, R3, t3,
        target_poses, timestamps
    )


def generate_stereo_poses():
    """Generate synthetic extrinsics for the stereo setup."""
    R_cam1 = np.eye(3)
    t_cam1 = np.zeros(3)
    R_cam2 = R.from_euler('xyz', [0, 60, 0], degrees=True).as_matrix()
    t_cam2 = np.array([.1, .1, 0])  # Baseline of 10 cm
    R_cam3 = R.from_euler('xyz', [0, 120, 0], degrees=True).as_matrix()
    t_cam3 = np.array([.2, 0, 0])
    return (R_cam1, t_cam1), (R_cam2, t_cam2), (R_cam3, t_cam3)

# Example usage:
board_size = (6, 6)  # Example: 5 rows, 6 columns of AprilTags
rpy_range = {'roll': (-10, -10), 'pitch': (25, 35), 'yaw': (-30, 30)}  # Degrees
xyz_range = {'x': (-1, 2.4), 'y': (-0.6, 0.1), 'z': (1, 2)}  # Meters
rty_range = {'r': (0.5, 2), 't': (-30, 180), 'y': (-0.6, 0.1)}  # Meters
(K_1_gt, dist_1_coeffs_gt), (K_2_gt, dist_2_coeffs_gt), (K_3_gt, dist_3_coeffs_gt) = generate_synthetic_cameras()
(R1, t1), (R2, t2), (R3, t3) = generate_stereo_poses()
print("K_1_gt: ", K_1_gt)
obj_points = simulate_apriltag_board(board_size[0], board_size[1], 0.13, 0.04)

theta_max_test = np.deg2rad(120)  # largest possible half-FOV
ok_1, min_deriv_1 = is_monotonic_kb(dist_1_coeffs_gt, theta_max_test)
ok_2, min_deriv_2 = is_monotonic_kb(dist_2_coeffs_gt, theta_max_test)
ok_3, min_deriv_3 = is_monotonic_kb(dist_3_coeffs_gt, theta_max_test)
ok = ok_1 and ok_2 and ok_3
min_deriv = min(min_deriv_1, min_deriv_2, min_deriv_3)
if not ok:
    raise ValueError(f"Distortion polynomial is non-monotonic (min derivative={min_deriv}). "
                     "Choose smaller coefficients or reduce max FOV.")
generate_synthetic_data(K_1_gt, dist_1_coeffs_gt, K_2_gt, dist_2_coeffs_gt, K_3_gt, dist_3_coeffs_gt, R1, t1, R2, t2, R3, t3, obj_points, (1280, 960), rpy_range, rty_range, 10, "/home/jake/calibration_w_python", board_size)

