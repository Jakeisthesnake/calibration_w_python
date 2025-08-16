import json
import numpy as np
import pandas as pd
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



def compute_homographies(obj_pts_list, img_pts_list):
    """Compute homographies and return filtered object/image points."""
    homographies = []
    filtered_obj_pts_list = []
    filtered_img_pts_list = []

    for obj_pts, img_pts in zip(obj_pts_list, img_pts_list):
        A = []
        for i in range(len(obj_pts)):
            X, Y = obj_pts[i][:2]  # Assume (X, Y, Z) but ignore Z (planar assumption)
            x, y = img_pts[i]
            A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
            A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])
        
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        print("H un normalized\n", H)
        H /= H[-1, -1]  # Normalize
        
        print("Singular values of A:", S)
        print("H\n", H)
        
        if S[-1] > 1e-8:  # Only keep valid homographies
            homographies.append(H)
            filtered_obj_pts_list.append(obj_pts)
            filtered_img_pts_list.append(img_pts)
        else:
            print("Singular values are too small, skipping this homography.")

    return homographies, filtered_obj_pts_list, filtered_img_pts_list

def process_csv(file_path, target_cam_id):
    """Processes a CSV file and returns object and image points for each timestamp."""
    
    # Read CSV
    df = pd.read_csv(file_path)
    # print("df\n", df)
    
    # Filter data for the target camera
    df = df[df['cam_id'] == target_cam_id]
    
    # Group data by timestamp (each timestamp represents a different calibration image)
    grouped_data = defaultdict(lambda: {'obj_points': [], 'img_points': []})
    # print("che1")
    for _, row in df.iterrows():
        # print("row\n:", row)
        timestamp = row['timestamp_ns']
        img_point = (row['corner_x'], row['corner_y'])
        obj_point = get_object_point(row['corner_id'])  # Assume predefined mapping
        grouped_data[timestamp]['img_points'].append(img_point)
        grouped_data[timestamp]['obj_points'].append(obj_point)
    # print("che2")
    # Convert to lists of NumPy arrays
    obj_pts_list = [np.array(data['obj_points'], dtype=np.float32) for data in grouped_data.values()]
    img_pts_list = [np.array(data['img_points'], dtype=np.float32) for data in grouped_data.values()]
    # print("obj_pts:\n", obj_pts_list)
    
    return obj_pts_list, img_pts_list

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
    # print("Singular values of V:", S)

    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])

    v0 = (B[0,1]*B[0,2] - B[1,2]*B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lambda_ = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[1,2]*B[0,0])) / B[0,0]
    alpha = np.sqrt(lambda_ / B[0,0])
    beta = np.sqrt(lambda_ * B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -B[0,1] * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - B[0,2] * alpha**2 / lambda_

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
        
        x_distorted, y_distorted = X * scale, Y * scale
        u, v = fx * x_distorted + cx, fy * y_distorted + cy
        projected_points.append([u, v])
    
    return np.array(projected_points), theta

def fisheye_reprojection_error(params, obj_pts_list, img_pts_list, num_images):
    """Compute reprojection error for all images at once using the Kannala-Brandt model."""
    
    # Extract parameters
    K = params[:4]  # fx, fy, cx, cy
    dist_coeffs = params[4:8]  # 4 fisheye distortion parameters
    extrinsics = params[8:].reshape((num_images, 6))  # rvecs and tvecs stacked

    total_error = []
    all_theta = []
    
    for i in range(num_images):
        rvec = extrinsics[i, :3]  # Keep as rotation vector
        tvec = extrinsics[i, 3:6]

        # Convert to rotation matrix only for projection
        R_matrix = R.from_rotvec(rvec).as_matrix()
        obj_pts_3d = np.hstack([obj_pts_list[i], np.zeros((len(obj_pts_list[i]), 1))])  # Assume Z=0
        # print("R_matrix\n", R_matrix)
        # print("tvec\n", tvec)
        # print("obj_pts_3d\n", obj_pts_3d)
        # print("R_matrix @ obj_pts_list[i].T\n", R_matrix @ obj_pts_3d.T)

        transformed_pts = (R_matrix @ obj_pts_3d.T).T + tvec
        
        projected_pts, theta_i = kannala_brandt_project(transformed_pts, K, dist_coeffs)
        error = (projected_pts - img_pts_list[i]).ravel()
        total_error.append(error)
        all_theta.append(theta_i)
        print("error lenght\n", len(error))
    # print("total_error\n", np.sum(np.hstack(total_error)))
    # print("max theta\n", np.max(all_llLtheta))
    
    return np.hstack(total_error)

def optimize_fisheye_parameters(K, dist_coeffs, rvecs, tvecs, obj_pts_list, img_pts_list):
    """Optimize all parameters (intrinsics, extrinsics, and distortion) using Levenberg-Marquardt."""
    
    num_images = len(obj_pts_list)
    
    # Flatten parameters into a single vector
    intrinsic_params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])  # fx, fy, cx, cy
    extrinsic_params = np.hstack([np.hstack([rvec.ravel(), tvec.ravel()]) for rvec, tvec in zip(rvecs, tvecs)])  # Keep rvec in vector form
    dist_coeffs = dist_coeffs.ravel()

    params = np.hstack([intrinsic_params, dist_coeffs, extrinsic_params])
    
    # Check that we have enough data for LM (residuals ≥ parameters)
    num_points = sum([pts.shape[0] for pts in img_pts_list])  # Total number of image points across all images
    num_parameters = len(params)

    if num_points * 2 < num_parameters:
        raise ValueError("Levenberg-Marquardt requires more residuals than parameters! Add more calibration images.")

    # Optimize using LM
    result = least_squares(
        fisheye_reprojection_error, params, args=(obj_pts_list, img_pts_list, num_images), method="lm"
    )

    # Extract optimized parameters
    refined_params = result.x
    refined_intrinsics = np.array([[refined_params[0], 0, refined_params[2]], 
                                   [0, refined_params[1], refined_params[3]], 
                                   [0, 0, 1]])
    refined_dist_coeffs = refined_params[4:8]
    refined_extrinsics = refined_params[8:].reshape((num_images, 6))

    refined_rvecs = [extr[:3] for extr in refined_extrinsics]  # Keep as rotation vectors
    refined_tvecs = [extr[3:] for extr in refined_extrinsics]

    return refined_intrinsics, refined_dist_coeffs, refined_rvecs, refined_tvecs


def calibrate_fisheye_camera(data_file):
    """
    Perform full camera calibration using Zhang's method and the Kannala-Brandt fisheye model.
    """
    obj_pts_list, img_pts_list = process_csv(data_file, target_cam_id=0)
    print("Checkpoint\n")
    print("obj_pts_list\n", obj_pts_list)
    print("img_pts_list\n", img_pts_list)


    # Step 1: Compute homographies
    H_list, filtered_obj_pts_list, filtered_img_pts_list = compute_homographies(obj_pts_list, img_pts_list)
    # print("H_list\n", H_list)
    print("Checkpoint2\n")
    
    # Step 2: Compute intrinsic parameters
    K = compute_intrinsic_params(H_list)
    print("K\n", K)
    input("check point 3")

    # Step 3: Compute extrinsic parameters
    extrinsics = [compute_extrinsic_params(H, K) for H in H_list]
    rvecs, tvecs = zip(*[(cv2.Rodrigues(R)[0], t) for R, t in extrinsics])

    # Step 4: Optimize parameters
    # dist_coeffs = np.zeros(4)  # Fisheye model uses 4 parameters
    dist_coeffs = np.array([0.1, -0.05, 0.02, -0.01])  # Fisheye model uses 4 parameters
    intrinsic_refined, dist_coeffs_refined, rvecs_refined, tvecs_refined = optimize_fisheye_parameters(
        K, dist_coeffs, rvecs, tvecs, filtered_obj_pts_list, filtered_img_pts_list
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