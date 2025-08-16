import json
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict
from scipy.optimize import least_squares

import scipy.linalg as linalg

def normalize_points(pts):
    """Normalize 2D points for better numerical stability in homography estimation."""
    mean = np.mean(pts, axis=0)  # Compute centroid
    std = np.std(pts, axis=0)  # Compute standard deviation

    scale = np.sqrt(2) / np.mean(std)  # Scale factor to make average distance sqrt(2)
    
    # Construct normalization matrix
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation
    pts_h = np.column_stack((pts, np.ones(len(pts))))  # Convert to homogeneous coordinates
    pts_norm = (T @ pts_h.T).T  # Apply transformation
    
    return pts_norm[:, :2], T  # Return normalized points and transformation matrix


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_points(obj_pts, img_pts, H=None):
    """Visualize 3D object points, 2D image points, and optionally reprojection via homography."""
    
    fig = plt.figure(figsize=(12, 6))
    
    # 3D Plot for Object Points
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(obj_pts[:, 0], obj_pts[:, 1], np.zeros(len(obj_pts)), c='b', marker='o')
    
    for i, (x, y) in enumerate(obj_pts):
        ax1.text(x, y, 0, str(i), color='red', fontsize=12)  # Label with index

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z (Set to 0)")
    ax1.set_title("3D Object Points")
    
    # 2D Plot for Image Points
    ax2 = fig.add_subplot(122)
    ax2.scatter(img_pts[:, 0], img_pts[:, 1], c='g', marker='x', label='Image Points')
    
    # Apply homography if provided
    if H is not None:
        obj_pts_h = np.hstack((obj_pts, np.ones((obj_pts.shape[0], 1))))  # Convert to homogeneous coordinates
        img_pts_reprojected_h = (H @ obj_pts_h.T).T  # Apply homography
        img_pts_reprojected = img_pts_reprojected_h[:, :2] / img_pts_reprojected_h[:, 2, np.newaxis]  # Normalize 
        print("obj_pts_h\n", obj_pts_h)
        print("img_pts\n", img_pts)
        print("img_pts_reprojected\n", img_pts_reprojected)
        
        ax2.scatter(img_pts_reprojected[:, 0], img_pts_reprojected[:, 1], c='r', marker='o', label='Reprojected Points')
        for i, (u, v) in enumerate(img_pts_reprojected):
            ax2.text(u, v, str(i), color='red', fontsize=12)
    
    for i, (u, v) in enumerate(img_pts):
        ax2.text(u, v, str(i), color='blue', fontsize=12)  # Label with index
    
    ax2.set_xlabel("Image X")
    ax2.set_ylabel("Image Y")
    ax2.set_title("2D Image Points")
    ax2.legend()
    
    plt.show()



def compute_homography(obj_pts, img_pts):
    """Compute homography using Direct Linear Transformation (DLT) with normalization."""
    obj_pts_norm, T_obj = normalize_points(obj_pts)
    img_pts_norm, T_img = normalize_points(img_pts)
    # obj_pts_norm = obj_pts
    # img_pts_norm = img_pts
    
    A = []
    for i in range(len(obj_pts_norm)):
        X, Y = obj_pts_norm[i]
        x, y = img_pts_norm[i]
        A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])
    
    A = np.array(A)
    # print("A\n", A)
    U, S, Vt = np.linalg.svd(A)
    # print("U\n", U)
    # print("S\n", S)
    # print("Vt\n", Vt)
    H_norm = Vt[-1].reshape(3, 3)  # Unnormalized homography

    # Denormalize: H = T_img^-1 * H_norm * T_obj
    H = np.linalg.inv(T_img) @ H_norm @ T_obj
    # H = H_norm

    # Compute reprojection error
    obj_pts_h = np.column_stack((obj_pts, np.ones(len(obj_pts))))
    img_pts_est = (H @ obj_pts_h.T).T
    img_pts_est /= img_pts_est[:, 2][:, np.newaxis]  # Normalize
    print("obj_pts_h\n", obj_pts_h)
    print("img_pts\n", img_pts)
    print("img_pts_est\n", img_pts_est)

    error = np.linalg.norm(img_pts_est[:, :2] - img_pts, axis=1).mean()
    print("Homography Reprojection Error:", error)
    visualize_points(np.array(obj_pts), np.array(img_pts), (H / H[-1, -1]))
    print("H\n", H)
    
    return H / H[-1, -1]  # Normalize

def homography(xy_src, xy_dst):
    '''
    xy_src  : Nx2 Matrix corrsponding to source points
                N is equal to the number of points
                Each row contains [x, y]
                
    xy_dst  : Nx2 Matrix corrsponding to destination points
                N is equal to the number of points
                Each row contains [x, y]  

    returns : 3x3 homography matrix such that [ xy_dst = H * xy_src ]
    
    Reference: https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
    '''
    
    src = np.asarray(xy_src, dtype=np.float32)
    dst = np.asarray(xy_dst, dtype=np.float32)
    
    if src.shape != dst.shape:
        raise Exception('Source and Destination dimensions must be same')
    if src.shape[0] < 4:
        raise Exception('At least 4 set of points is required to compute homography')
    if src.shape[1] != 2:
        raise Exception('Each row in Source and Destination matrices should contain [x, y] points')
        
    n_points = src.shape[0]
    
    # Form matrix A
    A = np.zeros((n_points*2, 9), dtype=np.float32)
    for i in range(n_points):
        # A[i] = [-x1, -y1, -1, 0, 0, 0, x2x1, x2y1, x2]
        A[2*i] = [-1.0*src[i][0], -1.0*src[i][1], -1, 0, 0, 0, dst[i][0]*src[i][0], dst[i][0]*src[i][1], dst[i][0]]
        # A[i+1] = [0, 0, 0, -x1, -y1, -1, y2x1, y2y1, y2]
        A[2*i+1] = [0, 0, 0, -1.0*src[i][0], -1.0*src[i][1], -1, dst[i][1]*src[i][0], dst[i][1]*src[i][1], dst[i][1]]
    # print("A_thirdparty\n", A)    
    U, Sigma, V_transpose = linalg.svd(A)
    # print("U_3rd\n", U)
    # print("Sigma_3rd\n", Sigma)
    # print("V_transpose_3rd\n", V_transpose)
    
    ## Form homography matrix
    # homography matrix corresponds to the column of V
    # corresponding to the smallest value of Sigma.
    # linalg.svd() returns Sigma in decreasing order
    # hence homography matrix will can be chosesn as
    # the last column of V or last row of V_transpose
    H = np.reshape(V_transpose[-1], (3,3))
    print("H_thirdparty\n", H)
    
    # Compute inverse of homography
    H_inverse = linalg.inv(H)
    # visualize_points(np.array(xy_src), np.array(xy_dst), H)
    
    return H, H_inverse


def process_csv(file_path, target_cam_id):
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Filter data for the target camera
    df = df[df['cam_id'] == target_cam_id]
    
    # Group data by timestamp
    grouped_data = defaultdict(lambda: {'obj_points': [], 'img_points': []})
    for _, row in df.iterrows():
        timestamp = row['timestamp_ns']
        img_point = (row['corner_x'], row['corner_y'])
        obj_point = get_object_point(row['corner_id'])  # Assume predefined mapping
        grouped_data[timestamp]['img_points'].append(img_point)
        grouped_data[timestamp]['obj_points'].append(obj_point)
    
    # Compute homographies
    H_list = [compute_homography(data['obj_points'], data['img_points']) 
              for data in grouped_data.values()]
    H_list_3rd = [homography(data['obj_points'], data['img_points'])[0] 
              for data in grouped_data.values()]
    
    
    
    
    return H_list_3rd

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

file_path = '/home/jake/atest/synthetic_data.csv'
target_cam_id = 0
H_list = process_csv(file_path, target_cam_id)
# print("H_list\n", H_list)
K = np.array([[800, 0, 640],  # fx, 0, cx
                  [0, 800, 480],  # 0, fy, cy
                  [0, 0, 1]])      # 0, 0, 1
K_inv =  np.linalg.inv(K)
R = K_inv @ H_list[0]
# print("R\n", R)
det_R = np.linalg.det(R)
# print("det_R\n", det_R)

