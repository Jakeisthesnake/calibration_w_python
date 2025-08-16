import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from collections import defaultdict
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def load_apriltag_board(json_file):
    """Load AprilTag board configuration from JSON and generate 3D object points."""
    with open(json_file, 'r') as f:
        board_config = json.load(f)

    tag_cols = board_config["tagCols"]
    tag_rows = board_config["tagRows"]
    tag_size = board_config["tagSize"]
    tag_spacing = board_config["tagSpacing"]
    
    obj_points = []  # List of 3D object points for the entire board
    
    for row in range(tag_rows):
        for col in range(tag_cols):
            # Compute the top-left corner of each tag
            tag_x = col * (tag_size + tag_spacing)
            tag_y = row * (tag_size + tag_spacing)
            tag_z = 0  # Assume the board is in the XY plane

            # Define the four corner points in (X, Y, Z)
            corners = np.array([
                [tag_x, tag_y, tag_z],  # Top-left
                [tag_x + tag_size, tag_y, tag_z],  # Top-right
                [tag_x + tag_size, tag_y + tag_size, tag_z],  # Bottom-right
                [tag_x, tag_y + tag_size, tag_z]  # Bottom-left
            ])
            obj_points.append(corners)

    return np.array(obj_points, dtype=np.float32)



def compute_homographies(obj_pts_list, img_pts_list, timestamp_list):
    """Compute homographies and return filtered object/image points."""
    homographies = []
    filtered_timestamp_list = []

    for obj_pts, img_pts, timestamp in zip(obj_pts_list, img_pts_list, timestamp_list):
        A = []
        for i in range(len(obj_pts)):
            X, Y = obj_pts[i][:2]  # Assume (X, Y, Z) but ignore Z (planar assumption)
            x, y = img_pts[i]
            A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
            A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])
        
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        # print("H un normalized\n", H)
        H /= H[-1, -1]  # Normalize
        
        # print("Singular values of A:", S)
        # print("H\n", H)
        
        if S[-1] > 1e-8:  # Only keep valid homographies
            homographies.append(H)
            filtered_timestamp_list.append(timestamp)
        else:
            print("Singular values are too small, skipping this homography.")

    return homographies, filtered_timestamp_list

def filter_data_by_timestamps(obj_pts_list, img_pts_list, corner_ids_list, timestamp_list, filtered_timestamps):
    """
    Filters the given lists based on the provided filtered timestamps.

    Parameters:
        obj_pts_list (list): List of object points.
        img_pts_list (list): List of image points.
        corner_ids_list (list): List of corner IDs.
        timestamp_list (list): List of timestamps.
        filtered_timestamps (list): List of timestamps to keep.

    Returns:
        tuple: Filtered versions of obj_pts_list, img_pts_list, corner_ids_list, and timestamp_list.
    """
    filtered_obj_pts = []
    filtered_img_pts = []
    filtered_corner_ids = []
    # print("filtered_timestamps\n", filtered_timestamps)

    for obj_pts, img_pts, corner_ids, timestamp in zip(obj_pts_list, img_pts_list, corner_ids_list, timestamp_list):
        # print("timestamp\n", timestamp)
        if timestamp in filtered_timestamps:
            # print("timestamp in filtered_timestamps\n", timestamp)
            filtered_obj_pts.append(obj_pts)
            filtered_img_pts.append(img_pts)
            filtered_corner_ids.append(corner_ids)

    return filtered_obj_pts, filtered_img_pts, filtered_corner_ids

def process_csv(file_path, target_cam_id):
    """Processes a CSV file and returns object and image points for each timestamp."""
    
    # Read CSV
    df = pd.read_csv(file_path)
    # print("df\n", df)
    
    # Filter data for the target camera
    df = df[df['cam_id'] == target_cam_id]
    
    # Group data by timestamp (each timestamp represents a different calibration image)
    grouped_data = defaultdict(lambda: {'obj_points': [], 'img_points': [], 'corner_ids': [], 'timestamps': []})
    # print("che1")
    timestamp_last = None
    for _, row in df.iterrows():
        # print("row\n:", row)
        timestamp = row['timestamp_ns']
        img_point = (row['corner_x'], row['corner_y'])
        obj_point = get_object_point(row['corner_id'])  # Assume predefined mapping
        grouped_data[timestamp]['img_points'].append(img_point)
        grouped_data[timestamp]['obj_points'].append(obj_point)
        grouped_data[timestamp]['corner_ids'].append(row['corner_id'])
        if timestamp != timestamp_last:
            grouped_data[timestamp]['timestamps'].append(timestamp)
            timestamp_last = timestamp
    # print("che2")
    # Convert to lists of NumPy arrays
    obj_pts_list = [np.array(data['obj_points'], dtype=np.float32) for data in grouped_data.values()]
    img_pts_list = [np.array(data['img_points'], dtype=np.float32) for data in grouped_data.values()]
    corner_ids_list = [data['corner_ids'] for data in grouped_data.values()]
    timestamp_list = [data['timestamps'][0] for data in grouped_data.values()]
    # print("obj_pts:\n", obj_pts_list)
    # print("timestamp_list\n", timestamp_list)
    
    return obj_pts_list, img_pts_list, corner_ids_list, timestamp_list

def get_object_point(corner_id, tag_rows=6, tag_cols=6, tag_size=0.13, tag_spacing=0.04):
    """Maps corner_id to object space coordinates, consistent with generate_corner_ids()."""
    tag_index = corner_id // 4  # Each tag has 4 corners
    local_corner = corner_id % 4  # Corner index within the tag

    row = tag_index // tag_cols  # Compute the row in the tag grid
    col = tag_index % tag_cols  # Compute the column in the tag grid

    # Compute tag origin in object space
    tag_x = col * (tag_size + tag_spacing)
    tag_y = row * (tag_size + tag_spacing)

    # Define the four corner offsets
    corner_offsets = [
        (0, 0),                # Top-left
        (tag_size, 0),         # Top-right
        (tag_size, tag_size),  # Bottom-right
        (0, tag_size)          # Bottom-left
    ]
    # print("local_corner", local_corner)

    offset_x, offset_y = corner_offsets[int(local_corner)]

    return (tag_x + offset_x, tag_y + offset_y)



def compute_intrinsic_params(H_list):
    """Compute intrinsic matrix K using homographies."""
    V = []
    for H in H_list:
        h1, h2 = H[:, 0], H[:, 1]
        V.append([h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], 
                  h1[1] * h2[1], h1[2] * h2[0] + h1[0] * h2[2], 
                  h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]])
        V.append([h1[0]**2 - h2[0]**2, 2 * (h1[0] * h1[1] - h2[0] * h2[1]), 
                  h1[1]**2 - h2[1]**2, 2 * (h1[0] * h1[2] - h2[0] * h2[2]), 
                  2 * (h1[1] * h1[2] - h2[1] * h2[2]), h1[2]**2 - h2[2]**2])
    
    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    _, S, _ = np.linalg.svd(V)
    print("Singular values of V:", S)
    print("b\n", b)

    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])

    v0 = (B[0,1]*B[0,2] - B[1,2]*B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lambda_ = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[1,2]*B[0,0])) / B[0,0]
    alpha = np.sqrt(lambda_ / B[0,0])
    beta = np.sqrt(lambda_ * B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -B[0,1] * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - B[0,2] * alpha**2 / lambda_
    print("alpha, beta, gamma, u0, v0\n", alpha, beta, gamma, u0, v0)

    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return K

def compute_extrinsic_params(H, K):
    """Compute extrinsic parameters (rotation and translation)."""
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    lambda_ = 1 / np.linalg.norm(K_inv @ h1)
    r1 = lambda_ * K_inv @ h1
    r2 = lambda_ * K_inv @ h2
    t = lambda_ * K_inv @ h3
    r3 = np.cross(r1, r2)

    R = np.column_stack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt  # Ensure it's a valid rotation matrix
    return R, t

def kannala_brandt_project(points, K, dist_coeffs):
    """Project 3D object points to 2D using the Kannala-Brandt model."""
    k1, k2, k3, k4 = dist_coeffs
    fx, fy, cx, cy = K[0], K[1], K[2], K[3]
    
    projected_points = []
    for X, Y, Z in points:
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(r, Z)
        # print("theta\n", theta)
        theta2, theta4, theta6, theta8 = theta**2, theta**4, theta**6, theta**8
        
        theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
        scale = theta_d / r if r > 1e-8 else 1.0
        if (False):
            scale = 1/Z
            scale = np.where(Z > 0.001, 1/Z, 1000)  # Default to 1.0 when r == 0
            # print("bypass distortion: scale = 1/Z\n")
            # print("Z ", Z)
            # print("scale: ", scale)
        
        x_distorted, y_distorted = X * scale, Y * scale
        u, v = fx * x_distorted + cx, fy * y_distorted + cy
        projected_points.append([u, v])
    
    return np.array(projected_points), theta

def skew(w):
    """Returns the skew-symmetric matrix of a 3D vector."""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def left_jacobian_inverse(omega):
    """Computes the inverse of the left Jacobian for SO(3)."""
    theta = np.linalg.norm(omega)
    if np.isclose(theta, 0.0):
        return np.eye(3)
    
    omega_hat = skew(omega)
    omega_hat_sq = omega_hat @ omega_hat

    A = 1 / 2
    B = (1 / theta**2) - ((1 + np.cos(theta)) / (2 * theta * np.sin(theta)))
    
    return np.eye(3) - A * omega_hat + B * omega_hat_sq

def log_se3(T):
    """
    Computes the logarithmic map of an SE(3) transformation.
    Returns a 6D vector: [omega (rotation), upsilon (translation)]
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Compute angle
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    
    if np.isclose(theta, 0.0):
        omega = np.zeros(3)
        J_inv = np.eye(3)
    else:
        omega = (theta / (2 * np.sin(theta))) * np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ])
        J_inv = left_jacobian_inverse(omega)

    upsilon = J_inv @ t

    return np.hstack((omega, upsilon))

def visualize_camera_data(obj_pts_list_0, img_pts_list_0, projected_pts_0,
                           obj_pts_list_1, img_pts_list_1, projected_pts_1):
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 3D object points for cam_0
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(obj_pts_list_0[:, 0], obj_pts_list_0[:, 1], obj_pts_list_0[:, 2], c='b', label='Object Points')
    ax1.set_title("Camera 0: Object Points")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim3d(-0.5, 2)
    ax1.set_ylim3d(-0.5, 2)
    ax1.set_zlim3d(-0.5, 2)
    ax1.legend()
    
    # Plot 2D image points and projected points for cam_0
    ax2 = fig.add_subplot(222)
    ax2.scatter(img_pts_list_0[:, 0], img_pts_list_0[:, 1], c='r', label='Image Points')
    ax2.scatter(projected_pts_0[:, 0], projected_pts_0[:, 1], c='g', marker='x', label='Projected Points')
    ax2.set_title("Camera 0: Image & Projected Points")
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.legend()
    
    # Plot 3D object points for cam_1
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(obj_pts_list_1[:, 0], obj_pts_list_1[:, 1], obj_pts_list_1[:, 2], c='b', label='Object Points')
    ax3.set_title("Camera 1: Object Points")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_xlim3d(-0.5, 2)
    ax3.set_ylim3d(-0.5, 2)
    ax3.set_zlim3d(-0.5, 2)
    ax3.legend()
    
    # Plot 2D image points and projected points for cam_1
    ax4 = fig.add_subplot(224)
    ax4.scatter(img_pts_list_1[:, 0], img_pts_list_1[:, 1], c='r', label='Image Points')
    ax4.scatter(projected_pts_1[:, 0], projected_pts_1[:, 1], c='g', marker='x', label='Projected Points')
    ax4.set_title("Camera 1: Image & Projected Points")
    ax4.set_xlabel("u")
    ax4.set_ylabel("v")
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


def fisheye_reprojection_error(params, obj_pts_list_0, img_pts_list_0, timestamp_list_0, corner_ids_list_0, obj_pts_list_1, img_pts_list_1, timestamp_list_1, corner_ids_list_1, all_timestamps):
    """Compute reprojection error for all images at once using the Kannala-Brandt model."""
    num_images_0 = len(timestamp_list_0)
    num_images_1 = len(timestamp_list_1)
    num_images = len(all_timestamps)

    # Extract parameters
    # print("params length\n", len(params))
    # print("num_images_0\n", num_images_0)
    # print("num_images_1\n", num_images_1)
    # print("param number: ", 4 + 4 + num_images_0 * 6 + 4 + num_images_1 * 6 + 3 + 3)
    K_0 = params[:4]  # fx, fy, cx, cy
    dist_coeffs_0 = params[4:8]  # 4 fisheye distortion parameters
    extrinsics_0 = params[8:(num_images_0*6 + 8)].reshape((num_images_0, 6))  # rvecs and tvecs stacked
    cam_0_param_length = 8 + num_images_0 * 6
    K_1 = params[cam_0_param_length:cam_0_param_length+4]  # fx, fy, cx, cy
    dist_coeffs_1 = params[cam_0_param_length+4:cam_0_param_length+8]  # 4 fisheye distortion parameters
    extrinsics_1 = params[cam_0_param_length+8:(cam_0_param_length + 8 + num_images_1*6)].reshape((num_images_1, 6))  # rvecs and tvecs stacked
    cam_1_param_length = cam_0_param_length + 8 + num_images_1 * 6
    rvec_cam_1 = params[cam_1_param_length:cam_1_param_length+3]
    tvec_cam_1 = params[cam_1_param_length+3:]
    R_matrix_cam_1 = R.from_rotvec(rvec_cam_1).as_matrix()

    # print("K_0\n", K_0)
    # print("dist_coeffs_0\n", dist_coeffs_0)
    # print("extrinsics_0\n", extrinsics_0)
    # print("K_1\n", K_1)
    # print("dist_coeffs_1\n", dist_coeffs_1)
    # print("extrinsics_1\n", extrinsics_1)
    # print("rvec_cam_1\n", rvec_cam_1)
    # print("tvec_cam_1\n", tvec_cam_1)
    R_matrix_0 = None
    R_matrix_1 = None


    total_error = []
    # all_theta = []
    
    for i in range(num_images):
        cam_0_index = timestamp_list_0.index(all_timestamps[i]) if all_timestamps[i] in timestamp_list_0 else None
        cam_1_index = timestamp_list_1.index(all_timestamps[i]) if all_timestamps[i] in timestamp_list_1 else None
        obj_pts_list_0_plot = np.array([[0, 0, 1]])
        img_pts_list_0_plot = np.array([[0, 0]])
        projected_pts_0_plot = np.array([[0, 0]])
        obj_pts_list_1_plot = np.array([[0, 0, 1]])
        img_pts_list_1_plot = np.array([[0, 0]])
        projected_pts_1_plot = np.array([[0, 0]])
        nan_flag = False
        if cam_0_index is not None:
            # print("img_pts_list_0[cam_0_index]:\n", img_pts_list_0[cam_0_index])
            # print("obj_pts_list_0_plot:\n", obj_pts_list_0_plot)
            
            rvec = extrinsics_0[cam_0_index, :3]  # Keep as rotation vector
            tvec_0 = extrinsics_0[cam_0_index, 3:6]

            # Convert to rotation matrix only for projection
            R_matrix_0 = R.from_rotvec(rvec).as_matrix()
            obj_pts_3d = np.hstack([obj_pts_list_0[cam_0_index], np.zeros((len(obj_pts_list_0[cam_0_index]), 1))])  # Assume Z=0
            
            # print("obj_pts_3d\n", obj_pts_3d)
            # print("R_matrix @ obj_pts_list[i].T\n", R_matrix @ obj_pts_3d.T)

            transformed_pts = (R_matrix_0 @ obj_pts_3d.T).T + tvec_0
            # if np.mean(transformed_pts[:, 2]) < 0:
            #     R_matrix_0 = -1*R_matrix_0
            #     t_vec_0 = -tvec_0
            #     transformed_pts = (R_matrix_0 @ obj_pts_3d.T).T + tvec_0
            
            # print("R_matrix_0\n", R_matrix_0)
            # print("tvec_0\n", tvec_0)
            
            projected_pts, theta_i = kannala_brandt_project(transformed_pts, K_0, dist_coeffs_0)
            error = (projected_pts - img_pts_list_0[cam_0_index]).ravel()
            # print("error_0\n", error)
            # print("error_0 sum\n", np.sum(error))
            total_error.append(error)
            if np.isnan(np.sum(error)):
                nan_flag = True
                print("error_0\n", error)
            # print("error_0\n", error)
            # all_theta.append(theta_i)
            # obj_pts_list_0_plot = obj_pts_3d
            obj_pts_list_0_plot = transformed_pts
            img_pts_list_0_plot = img_pts_list_0[cam_0_index]
            projected_pts_0_plot = projected_pts

            # print("obj_pts_list_0_plot:\n", obj_pts_list_0_plot)
        if cam_1_index is not None:
            # print("img_pts_list_1[cam_1_index]:\n", img_pts_list_1[cam_1_index])
            rvec = extrinsics_1[cam_1_index, :3]
            tvec_1 = extrinsics_1[cam_1_index, 3:6]
            # Convert to rotation matrix only for projection
            R_matrix_1 = R.from_rotvec(rvec).as_matrix()
            obj_pts_3d = np.hstack([obj_pts_list_1[cam_1_index], np.zeros((len(obj_pts_list_1[cam_1_index]), 1))])  # Assume Z=0
            # print("R_matrix\n", R_matrix)
            # print("tvec\n", tvec)
            # print("obj_pts_3d\n", obj_pts_3d)
            # print("obj_pts_3d - tvec\n", obj_pts_3d - tvec)
            # print("R_matrix @ obj_pts_list[i].T\n", R_matrix @ obj_pts_3d.T)

            transformed_pts = (R_matrix_1 @ obj_pts_3d.T).T + tvec_1
            # if np.mean(transformed_pts[:, 2]) < 0:
            #     R_matrix_1 = -1*R_matrix_1
            #     tvec_1 = -tvec_1
            #     transformed_pts = (R_matrix_1 @ obj_pts_3d.T).T + tvec_1

            # print("R_matrix_1\n", R_matrix_1)
            # print("tvec_1\n", tvec_1)
            # transformed_pts = (R_matrix_cam_1 @ obj_pts_3d.T).T + tvec_cam_1
            
            # transformed_pts = (np.linalg.inv(R_matrix_cam_1) @ (transformed_pts - tvec_cam_1).T).T
            # transformed_pts = (R_matrix_cam_1 @ transformed_pts.T).T + tvec_cam_1
            # transformed_pts = (R_matrix @ transformed_pts.T).T + tvec
            # transformed_pts = (np.linalg.inv(R_matrix_cam_1) @ (transformed_pts).T).T
            # transformed_pts = (R_matrix_cam_1 @ transformed_pts.T).T
            # print("transformed_pts\n", transformed_pts)
            projected_pts, theta_i = kannala_brandt_project(transformed_pts, K_1, dist_coeffs_1)
            error = (projected_pts - img_pts_list_1[cam_1_index]).ravel()
            # print("error_1\n", error)
            # print("error_1 sum\n", np.sum(error))
            if np.isnan(np.sum(error)):
                nan_flag = True
                print("error_1\n", error)
            total_error.append(error)
            # print("error_1\n", error)
            obj_pts_list_1_plot = transformed_pts

            # print("obj_pts_list_1_plot:\n", obj_pts_list_1_plot)
            img_pts_list_1_plot = img_pts_list_1[cam_1_index]
            projected_pts_1_plot = projected_pts
            # print("projected_pts_1_plot:\n", projected_pts_1_plot)
        if cam_0_index is not None and cam_1_index is not None:
            T_0 = np.eye(4)
            T_0[:3, :3] = R_matrix_0 #R.from_rotvec(rvec_0).as_matrix()
            T_0[:3, 3] = tvec_0

            T_1 = np.eye(4)
            T_1[:3, :3] = R_matrix_1 #R.from_rotvec(rvec_1).as_matrix()
            T_1[:3, 3] = tvec_1

            # Ground-truth inter-camera transformation
            T_01_obs = np.eye(4)
            R_matrix_01 = R.from_rotvec(rvec_cam_1).as_matrix()
            T_01_obs[:3, :3] = R_matrix_01
            T_01_obs[:3, 3] = tvec_cam_1

            # Estimated inter-camera transformation
            T_01_est = T_0 @ np.linalg.inv(T_1)
            R_matrix_01_maybe = R_matrix_0 @ np.linalg.inv(R_matrix_1)
            # print("R_matrix_01_maybe\n", R_matrix_01_maybe)
            tvec_01_est_maybe = tvec_0 - R_matrix_01_maybe @ tvec_1
            # print("tvec_01_est_maybe\n", tvec_01_est_maybe)
            R_matrix_01 = T_01_est[:3, :3]
            tvec_cam_1 = T_01_est[:3, 3]
            # print("R_matrix_01\n", R_matrix_01)
            # print("tvec_cam_1\n", tvec_cam_1)

            # Compute pose error using log(SE3)
            weight_cam_01 = 1.0
            pose_error = log_se3(T_01_obs @ np.linalg.inv(T_01_est)) * weight_cam_01
            if np.isnan(np.sum(pose_error)):
                nan_flag = True
                print("pose_error\n", pose_error)
            # print("pose_error\n", pose_error)
            total_error.append(pose_error)
            # print("pose_error\n", pose_error)
            # visualize_poses(R_matrix_0, tvec_0, R_matrix_1, tvec_1, R_matrix_01_maybe, tvec_01_est_maybe)
            # input()

        if nan_flag:
            visualize_camera_data(obj_pts_list_0_plot, img_pts_list_0_plot, projected_pts_0_plot, obj_pts_list_1_plot, img_pts_list_1_plot, projected_pts_1_plot)
            input()







            # all_theta.append(theta_i)
    print("total_error\n", np.sum(np.hstack(total_error)))
    # input("checkpoint")
    # print("max theta\n", np.max(all_theta))
    
    return np.hstack(total_error)

def optimize_fisheye_parameters(K_0, dist_coeffs_0, rvecs_0, tvecs_0, obj_pts_list_0, img_pts_list_0, timestamp_list_0, corner_ids_list_0, K_1, dist_coeffs_1, rvecs_1, tvecs_1, obj_pts_list_1, img_pts_list_1, timestamp_list_1, corner_ids_list_1, rvec_cam_1, tvec_cam_1):
    """Optimize all parameters (intrinsics, extrinsics, and distortion) using Levenberg-Marquardt."""
   
    all_timestamps = list(set(timestamp_list_0 + timestamp_list_1))
    
    # Flatten parameters into a single vector
    intrinsic_params_0 = np.array([K_0[0, 0], K_0[1, 1], K_0[0, 2], K_0[1, 2]])  # fx, fy, cx, cy
    extrinsic_params_0 = np.hstack([np.hstack([rvec.ravel(), tvec.ravel()]) for rvec, tvec in zip(rvecs_0, tvecs_0)])  # Keep rvec in vector form
    dist_coeffs_0 = dist_coeffs_0.ravel()
    intrinsic_params_1 = np.array([K_1[0, 0], K_1[1, 1], K_1[0, 2], K_1[1, 2]])  # fx, fy, cx, cy
    extrinsic_params_1 = np.hstack([np.hstack([rvec.ravel(), tvec.ravel()]) for rvec, tvec in zip(rvecs_1, tvecs_1)])  # Keep rvec in vector form
    dist_coeffs_1 = dist_coeffs_1.ravel()
    num_images_0 = len(timestamp_list_0)
    num_images_1 = len(timestamp_list_1)
    # print("dist_coeffs_0\n", dist_coeffs_0)
    # print("dist_coeffs_1\n", dist_coeffs_1)
    # print("rvec_cam_1\n", rvec_cam_1)
    # print("tvec_cam_1\n", tvec_cam_1)
    # print("extrinsic_params_0\n", extrinsic_params_0)
    # print("extrinsic_params_1\n", extrinsic_params_1)
    # print("K_0\n", K_0)
    # print("K_1\n", K_1)
    # print("rvec_cam_1\n", rvecs_0)
    # print("tvec_cam_1\n", tvecs_0)

    # print("intrinsic_params_0\n", intrinsic_params_0)
    # print("dist_coeffs_0\n", dist_coeffs_0)
    # print("extrinsic_params_0\n", extrinsic_params_0)
    # print("intrinsic_params_1\n", intrinsic_params_1)
    # print("dist_coeffs_1\n", dist_coeffs_1)
    # print("extrinsic_params_1\n", extrinsic_params_1)
    # print("rvec_cam_1\n", rvec_cam_1)
    # print("tvec_cam_1\n", tvec_cam_1)
    
    # print("len(intrinsic_params_0)\n", len(intrinsic_params_0))
    # print("len(dist_coeffs_0)\n", len(dist_coeffs_0))
    # print("len(extrinsic_params_0)\n", len(extrinsic_params_0))
    # print("len(intrinsic_params_1)\n", len(intrinsic_params_1))
    # print("len(dist_coeffs_1)\n", len(dist_coeffs_1))
    # print("len(extrinsic_params_1)\n", len(extrinsic_params_1))
    # print("len(rvec_cam_1)\n", len(np.array(rvec_cam_1)))
    # print("len(tvec_cam_1)\n", len(tvec_cam_1))
    # print("len(intrinsic_params_0) + len(dist_coeffs_0) + len(extrinsic_params_0) + len(intrinsic_params_1) + len(dist_coeffs_1) + len(extrinsic_params_1) + len(rvec_cam_1) + len(tvec_cam_1)\n", len(intrinsic_params_0) + len(dist_coeffs_0) + len(extrinsic_params_0) + len(intrinsic_params_1) + len(dist_coeffs_1) + len(extrinsic_params_1) + len(rvec_cam_1) + len(tvec_cam_1))
    



    params = np.hstack([intrinsic_params_0, dist_coeffs_0, extrinsic_params_0, intrinsic_params_1, dist_coeffs_1, extrinsic_params_1, rvec_cam_1, tvec_cam_1])
    # print("params\n", params)
    # print("params length\n", len(params))
    
    # print("params\n", params)
    # Check that we have enough data for LM (residuals ≥ parameters)
    # print("img_pts_list_0\n", img_pts_list_0)
    # print("img_pts_list_1\n", img_pts_list_1)
    num_points_0 = sum([pts.shape[0] for pts in img_pts_list_0])  # Total number of image points across all images
    num_points_1 = sum([pts.shape[0] for pts in img_pts_list_1])  # Total number of image points across all images
    num_points = num_points_0 + num_points_1
    num_parameters = len(params)

    if num_points * 2 < num_parameters:
        print("num_points", num_points)
        print("num_parameters", num_parameters)
        raise ValueError("Levenberg-Marquardt requires more residuals than parameters! Add more calibration images.")

    # Optimize using LM
    result = least_squares(
        fisheye_reprojection_error, params, args=(obj_pts_list_0, img_pts_list_0, timestamp_list_0, corner_ids_list_0, obj_pts_list_1, img_pts_list_1, timestamp_list_1, corner_ids_list_1, all_timestamps), method="lm"
    )
    
    # Extract optimized parameters
    refined_params = result.x
    refined_intrinsics_0 = np.array([[refined_params[0], 0, refined_params[2]], 
                                   [0, refined_params[1], refined_params[3]], 
                                   [0, 0, 1]])
    refined_dist_coeffs_0 = refined_params[4:8]
    refined_extrinsics_0 = refined_params[8:(num_images_0*6 + 8)].reshape((num_images_0, 6))
    refined_rvecs_0 = [extr[:3] for extr in refined_extrinsics_0]  # Keep as rotation vectors
    refined_tvecs_0 = [extr[3:] for extr in refined_extrinsics_0]
    cam_0_param_length = 8 + num_images_0 * 6
    refined_intrinsics_1 = np.array([[refined_params[cam_0_param_length], 0, refined_params[cam_0_param_length+2]],
                                   [0, refined_params[cam_0_param_length+1], refined_params[cam_0_param_length+3]],
                                   [0, 0, 1]])
    refined_dist_coeffs_1 = refined_params[cam_0_param_length+4:cam_0_param_length+8]
    refined_extrinsics_1 = refined_params[cam_0_param_length+8:(cam_0_param_length + 8 + num_images_1*6)].reshape((num_images_1, 6))
    refined_rvecs_1 = [extr[:3] for extr in refined_extrinsics_1]  # Keep as rotation vectors
    refined_tvecs_1 = [extr[3:] for extr in refined_extrinsics_1]
    cam_1_param_length = cam_0_param_length + 8 + num_images_1 * 6
    refined_rvec_cam_1 = refined_params[cam_1_param_length:cam_1_param_length+3]
    refined_tvec_cam_1 = refined_params[cam_1_param_length+3:cam_1_param_length+6]

    print("refined_intrinsics_0\n", refined_intrinsics_0)
    print("refined_dist_coeffs_0\n", refined_dist_coeffs_0)
    # print("refined_rvecs_0\n", refined_rvecs_0)
    # print("refined_tvecs_0\n", refined_tvecs_0)
    print("refined_intrinsics_1\n", refined_intrinsics_1)
    print("refined_dist_coeffs_1\n", refined_dist_coeffs_1)
    # print("refined_rvecs_1\n", refined_rvecs_1)
    # print("refined_tvecs_1\n", refined_tvecs_1)
    print("refined_rvec_cam_1\n", refined_rvec_cam_1)
    print("refined_tvec_cam_1\n", refined_tvec_cam_1)
    
    return refined_intrinsics_0, refined_dist_coeffs_0, refined_rvecs_0, refined_tvecs_0, refined_intrinsics_1, refined_dist_coeffs_1, refined_rvecs_1, refined_tvecs_1, refined_rvec_cam_1, refined_tvec_cam_1


def calibrate_fisheye_camera(data_file):
    """
    Perform full camera calibration using Zhang's method and the Kannala-Brandt fisheye model.
    """
    obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0 = process_csv(data_file, target_cam_id=0)
    obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1 = process_csv(data_file, target_cam_id=1)
    
    # print("img_pts_list_0\n", img_pts_list_0)
    # print("img_pts_list_1\n", img_pts_list_1)


    # Step 1: Compute homographies
    H_list_0, filtered_timestamp_list_0 = compute_homographies(obj_pts_list_0, img_pts_list_0, timestamp_list_0)
    H_list_1, filtered_timestamp_list_1 = compute_homographies(obj_pts_list_1, img_pts_list_1, timestamp_list_1)
    # print("H_list_0\n", H_list_0)
    # print("H_list_1\n", H_list_1)
    print("filtered_timestamp_list_0\n", filtered_timestamp_list_0)


    obj_pts_list_0, img_pts_list_0, corner_ids_list_0 = filter_data_by_timestamps(
        obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0, filtered_timestamp_list_0)

    obj_pts_list_1, img_pts_list_1, corner_ids_list_1 = filter_data_by_timestamps(
        obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1, filtered_timestamp_list_1)

    
    print("Checkpoint2\n")
    # print("H_list_0\n", H_list_0)
    # print("H_list_1\n", H_list_1)
    # print("img_pts_list_0\n", img_pts_list_0)
    # print("img_pts_list_1\n", img_pts_list_1)
    # print("corner_ids_list_0\n", corner_ids_list_0)
    # print("corner_ids_list_1\n", corner_ids_list_1)
    # print("timestamp_list_0\n", timestamp_list_0)
    # print("timestamp_list_1\n", timestamp_list_1)
    # print("obj_pts_list_0\n", obj_pts_list_0)
    # print("obj_pts_list_1\n", obj_pts_list_1)
    # print("filtered_timestamp_list_0\n", filtered_timestamp_list_0)
    # print("filtered_timestamp_list_1\n", filtered_timestamp_list_1)
    
    # Step 2: Compute intrinsic parameters
    K_0 = compute_intrinsic_params(H_list_0)
    K_1 = compute_intrinsic_params(H_list_1)
    # K_0 = np.array([[ 8.00091117e+02, -3.53677890e-02,  6.39855292e+02],
    #                [ 0.00000000e+00,  8.00130558e+02,  4.79944330e+02],
    #                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # K_1 = np.array([[ 8.09103963e+02, -1.70293870e-01,  6.48625336e+02],
    #                [ 0.00000000e+00,  8.08718280e+02,  4.70333165e+02],
    #                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    print("K_0\n", K_0)
    print("K_1\n", K_1)
    # input("check point 3")

    # Step 3: Compute extrinsic parameters
    extrinsics_0 = [compute_extrinsic_params(H, K_0) for H in H_list_0]
    extrinsics_1 = [compute_extrinsic_params(H, K_1) for H in H_list_1]
    rvecs_0, tvecs_0 = zip(*[(cv2.Rodrigues(R)[0], t) for R, t in extrinsics_0])
    rvecs_1, tvecs_1 = zip(*[(cv2.Rodrigues(R)[0], t) for R, t in extrinsics_1])

    # Step 4: Optimize parameters
    dist_coeffs_0 = np.zeros(4)  # Fisheye model uses 4 parameters
    dist_coeffs_1 = np.zeros(4)  # Fisheye model uses 4 parameters
    # dist_coeffs = np.array([0.1, -0.05, 0.02, -0.01])  # Fisheye model uses 4 parameters
    rvec_cam_1 = np.zeros(3)  # Initial guess for rotation vector of camera 1
    tvec_cam_1 = np.zeros(3)  # Initial guess for translation vector of camera 1

    K_0 = np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]])
    K_1 = np.array([[810, 0, 650], [0, 810, 470], [0, 0, 1]])
    dist_coeffs_0 = np.array([0.1, -0.05, 0.02, -0.01])
    dist_coeffs_1 = np.array([0.09, -0.04, 0.015, -0.008])
    R_cam1 = R.from_euler('xyz', [0, 60, 0], degrees=True).as_matrix()
    rot = R.from_matrix(R_cam1)
    rvec_cam_1 = rot.as_rotvec()
    t_cam2 = np.array([.1, 0, 0])  # Baseline of 10 cm
    intrinsic_refined_0, dist_coeffs_refined_0, rvecs_refined_0, tvecs_refined_0, intrinsic_refined_1, dist_coeffs_refined_1, rvecs_refined_1, tvecs_refined_1, cam_1_rvec, cam_1_tvec = optimize_fisheye_parameters(
        K_0, dist_coeffs_0, rvecs_0, tvecs_0, obj_pts_list_0, img_pts_list_0, filtered_timestamp_list_0, corner_ids_list_0, K_1, dist_coeffs_1, rvecs_1, tvecs_1, obj_pts_list_1, img_pts_list_1, filtered_timestamp_list_1, corner_ids_list_1, rvec_cam_1, tvec_cam_1
    )
    print("intrinsic_refined\n", intrinsic_refined)
    print("dist_coeffs_refined\n", dist_coeffs_refined)
    print("rvecs_refined\n", rvecs_refined)
    print("tvecs_refined\n", tvecs_refined)

    return K, dist_coeffs_refined, rvecs_refined, tvecs_refined


calibrate_fisheye_camera("/home/jake/atest/synthetic_data.csv")
# print("K\n", K)
# print("dist_coeffs_refined\n", dist_coeffs_refined)
# print("rvecs_refined\n", rvecs_refined)
# print("tvecs_refined\n", tvecs_refined)
# print("Reprojection error:", result.cost)
# print("Optimization success:", result.success)
# print("Optimized parameters:", result.x)
# print("Optimization message:", result.message)
# print("Optimization time:", result.optimization_time)
# print("Number of iterations:", result.nfev)
# print("Number of function evaluations:", result.njev)
# print("Number of Jacobian evaluations:", result.njev)
# print("Number of linear system iterations:", result.nit)
# print("Optimization status code:", result.status)