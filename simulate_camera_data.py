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
    print("obj_points function shape: ", obj_points.shape)
    return obj_points


def save_generated_data_to_csv(file_path, timestamp_ns, img_points, corner_ids):
    """
    Save the generated image points to a CSV file, appending if the file exists.
    """
    cam_id = 0  # Hardcoded cam_id
    radii = np.random.normal(loc=2.0, scale=0.4, size=len(img_points))
    radii = np.round(radii, 2)
    img_points = np.round(img_points, 2)


    data = {
        "timestamp_ns": [timestamp_ns] * len(img_points),
        "cam_id": [cam_id] * len(img_points),
        "corner_x": img_points[:, 0],
        "corner_y": img_points[:, 1],
        "corner_id": corner_ids,
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
    print("corner_ids function shape: ", corner_ids.shape)
    return corner_ids


def generate_synthetic_camera():
    """Generate synthetic camera intrinsics and distortion parameters."""
    K = np.array([[800, 0, 640],  # fx, 0, cx
                  [0, 800, 480],  # 0, fy, cy
                  [0, 0, 1]])      # 0, 0, 1
    # print( "K_inv = ", np.linalg.inv(K))
    
    dist_coeffs = np.array([0.1, -0.05, 0.02, -0.01])  # Example fisheye distortion coefficients
    return K, dist_coeffs

def project_points_kannala_brandt(obj_points, K, dist_coeffs, R_matrix, tvec):
    """Project 3D object points to 2D using the Kannala-Brandt model."""
    transformed_pts = (R_matrix @ obj_points.T).T + tvec
    X, Y, Z = transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2]
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

def plot_filtered_points(obj_points, filtered_obj_pts, filtered_img_pts, corner_ids, filtered_corner_ids, img_size):
    """
    Visualizes the 3D points, color codes them by whether they are filtered, and then projects the filtered points 
    onto the image plane (2D sensor coordinates).
    """
    # 3D Plotting of points and color coding based on filtering
    fig = plt.figure(figsize=(14, 7))

    # Plot the 3D space with the points
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c='r', label='All Points')
    ax.scatter(filtered_obj_pts[:, 0], filtered_obj_pts[:, 1], filtered_obj_pts[:, 2], c='g', label='Filtered Points')
    
    # Labeling the points with their corner IDs
    for i, corner_id in enumerate(corner_ids):
        ax.text(obj_points[i, 0], obj_points[i, 1], obj_points[i, 2], f'{corner_id}', size=10, color='black')
    
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

def filter_visible_points(obj_points, K, dist_coeffs, R_matrix, tvec, img_size, corner_ids):
    """
    Filters visible AprilTag corners by projecting them onto the image plane and removing points outside the image bounds.
    """
    width, height = img_size

    # Transform 3D points to the camera coordinate frame
    obj_pts_cam = (R_matrix @ obj_points.T).T + tvec  # Shape: (N, 3)

    # Filter out points behind the camera (Z <= 0)
    valid_z_indices = obj_pts_cam[:, 2] > 0
    # print("valid_z_indices shape: ", valid_z_indices.shape)
    # print ("valid_z_indices: ", valid_z_indices)
    # print("obj_pts_cam", obj_pts_cam)

    obj_pts_cam = obj_pts_cam[valid_z_indices]
    valid_z_corner_ids = corner_ids[valid_z_indices]  #Ensure we keep only matching IDs


    if len(obj_pts_cam) == 0:
        return np.array([]), np.array([]), []

    # Project points using Kannala-Brandt model
    img_pts = project_points_kannala_brandt(obj_pts_cam, K, dist_coeffs, R_matrix, tvec)
    
    # Filter points inside image bounds
    inside_x = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < width)
    inside_y = (img_pts[:, 1] >= 0) & (img_pts[:, 1] < height)
    inside_image = inside_x & inside_y

    filtered_obj_pts = obj_pts_cam[inside_image]
    filtered_img_pts = img_pts[inside_image]
    filtered_corner_ids = valid_z_corner_ids[inside_image]  #Correctly filtered corner IDs
    # plot_filtered_points(obj_points, filtered_obj_pts, filtered_img_pts, corner_ids, filtered_corner_ids, img_size)


    return filtered_obj_pts, filtered_img_pts, filtered_corner_ids




def generate_synthetic_data(K, dist_coeffs, obj_points, img_size, rpy_range, xyz_range, num_samples, output_dir, board_size):
    """
    Generate synthetic calibration data by randomly transforming the AprilTag board.

    Parameters:
        K (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Fisheye distortion coefficients (k1, k2, k3, k4).
        obj_points (np.ndarray): Nx3 array of AprilTag corner points in 3D.
        img_size (tuple): (width, height) of the camera image.
        rpy_range (dict): {'roll': (min, max), 'pitch': (min, max), 'yaw': (min, max)} in degrees.
        xyz_range (dict): {'x': (min, max), 'y': (min, max), 'z': (min, max)} in meters.
        num_samples (int): Number of random transformations to generate.
        output_dir (str): Directory to save the generated CSV files.
        board_size (tuple): (rows, cols) defining the AprilTag board.
    """
    headers = "timestamp_ns,cam_id,corner_x,corner_y,corner_id,radius\n"
    
    # Check if the file exists and write headers if not
    file_path = f"{output_dir}/synthetic_data.csv"
    with open(file_path, 'w') as file:
        file.write(headers)

    # Unpack board_size to match generate_corner_ids parameters
    tag_rows, tag_cols = board_size
    corner_ids = generate_corner_ids(tag_rows, tag_cols)
    # print("Checkpoint generate_synthetic_data\n")

    for i in range(num_samples):
        # print("Checkpoint generate_synthetic_data sample\n")
        # Sample random rotation (roll, pitch, yaw)
        roll = np.radians(np.random.uniform(*rpy_range['roll']))
        pitch = np.radians(np.random.uniform(*rpy_range['pitch']))
        yaw = np.radians(np.random.uniform(*rpy_range['yaw']))
        R_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

        # Sample random translation
        tx = np.random.uniform(*xyz_range['x'])
        ty = np.random.uniform(*xyz_range['y'])
        tz = np.random.uniform(*xyz_range['z'])
        tvec = np.array([tx, ty, tz])

        # R and tvec test
        # R_matrix = np.array([[1, 0, 0],
        #                      [0, 1, 0],
        #                      [0, 0, 1]])
        # R_matrix = R.from_euler('xyz', [np.radians(0), np.radians(0), np.radians(0)]).as_matrix()
        # tvec = np.array([-0.25, -0.15, 0.3])

        print("R and tvec\n" , R_matrix, tvec)
        print("obj_points shape: ", obj_points.shape)

        # Project points and filter visible ones
        filtered_obj_pts, filtered_img_pts, filtered_corner_ids = filter_visible_points(
            obj_points, K, dist_coeffs, R_matrix, tvec, img_size, corner_ids
        )
        
        # print("filtered_obj_pts shape: ", filtered_obj_pts.shape)
        # print("filtered_img_pts shape: ", filtered_img_pts.shape)
        # print("filtered_corner_ids length: ", len(filtered_corner_ids))

        if len(filtered_obj_pts) < 20:
            print(f"Skipping sample {i} (Not enough visible points)")
            continue  # Skip this sample if no points are visible

        # Save to CSV
        file_path = f"{output_dir}/synthetic_data.csv"
        save_generated_data_to_csv(file_path, i, filtered_img_pts, filtered_corner_ids)

# Example usage:
board_size = (6, 6)  # Example: 5 rows, 6 columns of AprilTags
rpy_range = {'roll': (-15, 15), 'pitch': (-10, 10), 'yaw': (-30, 30)}  # Degrees
xyz_range = {'x': (-0.4, 0.1), 'y': (-0.4, 0.1), 'z': (0.5, 1)}  # Meters
K_gt, dist_coeffs_gt = generate_synthetic_camera()
obj_points = simulate_apriltag_board(board_size[0], board_size[1], 0.13, 0.04)
generate_synthetic_data(K_gt, dist_coeffs_gt, obj_points, (1280, 960), rpy_range, xyz_range, 20, "/home/jake/atest", board_size)


