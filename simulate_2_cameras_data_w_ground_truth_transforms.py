import numpy as np
import pandas as pd
import cv2
import json
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
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


def save_generated_data_to_csv(file_path, timestamp_ns, img_points, corner_ids, cam_id, rvec, tvec):
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
        "corner_x": img_points[:, 0],
        "corner_y": img_points[:, 1],
        "corner_id": corner_ids,
        "radius": radii,
        "rvec_x": [rvec[0]] * len(img_points),
        "rvec_y": [rvec[1]] * len(img_points),
        "rvec_z": [rvec[2]] * len(img_points),
        "tvec_x": [tvec[0]] * len(img_points),
        "tvec_y": [tvec[1]] * len(img_points),
        "tvec_z": [tvec[2]] * len(img_points),
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
    """Generate synthetic intrinsics for two cameras."""
    K1 = np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]])
    K2 = np.array([[810, 0, 650], [0, 810, 470], [0, 0, 1]])
    dist1 = np.array([0.1, -0.05, 0.02, -0.01])
    dist2 = np.array([0.09, -0.04, 0.015, -0.008])
    return (K1, dist1), (K2, dist2)


def project_points_kannala_brandt(obj_points_cam, K, dist_coeffs):
    """Project 3D object points to 2D using the Kannala-Brandt model."""
    # transformed_pts = (R_matrix @ obj_points_world.T).T + tvec
    # X, Y, Z = transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2]
    X, Y, Z = obj_points_cam[:, 0], obj_points_cam[:, 1], obj_points_cam[:, 2]
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)
    
    k1, k2, k3, k4 = dist_coeffs
    theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    epsilon = 0.0001  # Small value to avoid division by zero
    if np.any(r <= epsilon):
        print("epsilon triggered for small r values")

    scale = np.where(r > epsilon, theta_d / r, 1.0)  # Default to 1.0 when r == 0
    
    if (True):
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
    u, v = fx * x_distorted + cx, fy * y_distorted + cy
    return np.column_stack((u, v))



def add_noise(img_points, noise_std=1.0):
    """Add Gaussian noise to image points."""
    return img_points + np.random.normal(scale=noise_std, size=img_points.shape)


def plot_two_camera_pts(R_1, t_1, R_2, t_2, filtered_img_pts_1, filtered_ids_1, filtered_img_pts_2, filtered_ids_2, obj_corner_ids, obj_points_world):
    # # Define a board pose in front of the cameras
    # R_board = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    # t_board = np.array([0.0, 0.0, 1.0])  # 1m in front

    # obj_points_world = (R_board @ obj_points.T).T + t_board
    obj_points_cam_2 = (np.linalg.inv(R_2) @ (obj_points_world - t_2).T).T

    # R1_trans = R1 @ R_board
    # R2_trans = R2 @ R_board
    # t1_trans = R1 @ t_board + t1
    # t2_trans = R2 @ t_board + t2

    # # Project points for both cameras
    # img_pts_1, obj_pts_cam_1 = project_points(obj_points, K_1_gt, R1_trans, t1_trans)
    # img_pts_2, obj_pts_cam_2 = project_points(obj_points, K_1_gt, R2_trans, t2_trans)

    # Visualization
    fig = plt.figure(figsize=(12, 6))

    # 3D Plot
    ax = fig.add_subplot(2, 2, (1,3), projection='3d')
    ax.scatter(obj_points_world[:, 0], obj_points_world[:, 1], obj_points_world[:, 2], c='b', label='obj_points_world')
    ax.scatter(obj_points_cam_2[:, 0], obj_points_cam_2[:, 1], obj_points_cam_2[:, 2], c='g', label='obj_points_world')
    ax.scatter(t_1[0], t_1[1], t_1[2], c='r', marker='o', label='Camera 1')
    ax.scatter(t_2[0], t_2[1], t_2[2], c='g', marker='o', label='Camera 2')
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

    for i, corner_id in enumerate(obj_corner_ids):
        ax.text(obj_points_world[i, 0], obj_points_world[i, 1], obj_points_world[i, 2], f'{corner_id}', size=10, color='black')
        ax.text(obj_points_cam_2[i, 0], obj_points_cam_2[i, 1], obj_points_cam_2[i, 2], f'{corner_id}', size=10, color='black')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-0.5, 2)
    ax.set_ylim3d(-0.5, 2)
    ax.set_zlim3d(-0.5, 2)
    ax.set_title("3D Visualization of Cameras and Object Points")
    ax.legend()

    # 2D Image Projections
    ax2 = fig.add_subplot(222)
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
    ax2.set_xlim(0, 1280)
    ax2.set_ylim(0, 960)
    

    ax3 = fig.add_subplot(224)
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
    ax3.set_xlim(0, 1280)
    ax3.set_ylim(0, 960)

    plt.tight_layout()
    plt.show()


def plot_filtered_points(obj_points_world, filtered_obj_pts, filtered_img_pts, corner_ids, filtered_corner_ids, img_size):
    """
    Visualizes the 3D points, color codes them by whether they are filtered, and then projects the filtered points 
    onto the image plane (2D sensor coordinates).
    """
    # 3D Plotting of points and color coding based on filtering
    fig = plt.figure(figsize=(14, 7))

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
    
    # Filter points inside image bounds
    inside_x = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < width)
    inside_y = (img_pts[:, 1] >= 0) & (img_pts[:, 1] < height)
    inside_image = inside_x & inside_y

    filtered_obj_pts = valid_obj_pts_cam[inside_image]
    filtered_img_pts = img_pts[inside_image]
    filtered_corner_ids = valid_z_corner_ids[inside_image]  #Correctly filtered corner IDs
    # plot_filtered_points(obj_points_world, filtered_obj_pts, filtered_img_pts, corner_ids, filtered_corner_ids, img_size)


    return filtered_obj_pts, filtered_img_pts, filtered_corner_ids



def generate_synthetic_data(K_1, dist_coeffs_1, K_2, dist_coeffs_2, R1, t1, R2, t2, obj_points, img_size, rpy_range, xyz_range, num_samples, output_dir, board_size):
    headers = "timestamp_ns,cam_id,corner_x,corner_y,corner_id,radius,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z\n"
    file_path = f"{output_dir}/synthetic_data.csv"
    with open(file_path, 'w') as file:
        file.write(headers)

    tag_rows, tag_cols = board_size
    corner_ids = generate_corner_ids(tag_rows, tag_cols)

    x_values = np.linspace(*xyz_range['x'], 4)
    y_values = np.linspace(*xyz_range['y'], 3)
    z_values = np.linspace(*xyz_range['z'], 3)
    # y_values = np.array([0.0])
    # z_values = np.array([2.0])
    timestamp_ns = 0

    for tx in x_values:
        for ty in y_values:
            for tz in z_values:
                roll = np.radians(np.random.uniform(*rpy_range['roll']))
                pitch = np.radians(np.random.uniform(*rpy_range['pitch']))
                yaw = np.radians(np.random.uniform(*rpy_range['yaw']))
                R_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
                # R_matrix = R.from_euler('xyz', [0, 30, 0], degrees=True).as_matrix()
                rot = R.from_matrix(R_matrix)
                rvec = rot.as_rotvec()
                tvec = np.array([tx, ty, tz])

                obj_points_world = (R_matrix @ obj_points.T).T + tvec

                filtered_obj_pts_1, filtered_img_pts_1, filtered_corner_ids_1 = filter_visible_points(
                    obj_points_world, K_1, dist_coeffs_1, R1, t1, img_size, corner_ids
                )
                filtered_obj_pts_2, filtered_img_pts_2, filtered_corner_ids_2 = filter_visible_points(
                    obj_points_world, K_2, dist_coeffs_2, R2, t2, img_size, corner_ids
                )

                save_generated_data_to_csv(file_path, timestamp_ns, filtered_img_pts_1, filtered_corner_ids_1, 0, rvec, tvec)
                
                save_generated_data_to_csv(file_path, timestamp_ns, filtered_img_pts_2, filtered_corner_ids_2, 1, rvec, tvec)

                # plot_two_camera_pts(R1, t1, R2, t2, filtered_img_pts_1, filtered_corner_ids_1, filtered_img_pts_2, filtered_corner_ids_2, corner_ids, obj_points_world)
                timestamp_ns += 1

def generate_stereo_poses():
    """Generate synthetic extrinsics for the stereo setup."""
    R_cam1 = np.eye(3)
    t_cam1 = np.zeros(3)
    R_cam2 = R.from_euler('xyz', [0, 60, 0], degrees=True).as_matrix()
    t_cam2 = np.array([.1, 0, 0])  # Baseline of 10 cm
    return (R_cam1, t_cam1), (R_cam2, t_cam2)

# Example usage:
board_size = (6, 6)  # Example: 5 rows, 6 columns of AprilTags
rpy_range = {'roll': (-10, -10), 'pitch': (25, 35), 'yaw': (-30, 30)}  # Degrees
xyz_range = {'x': (-0.5, 2.4), 'y': (-0.6, 0.1), 'z': (0.5, 1)}  # Meters
(K_1_gt, dist_1_coeffs_gt), (K_2_gt, dist_2_coeffs_gt) = generate_synthetic_cameras()
(R1, t1), (R2, t2) = generate_stereo_poses()
obj_points = simulate_apriltag_board(board_size[0], board_size[1], 0.13, 0.04)
generate_synthetic_data(K_1_gt, dist_1_coeffs_gt, K_2_gt, dist_2_coeffs_gt, R1, t1, R2, t2, obj_points, (1280, 960), rpy_range, xyz_range, 10, "/home/jake/atest", board_size)

