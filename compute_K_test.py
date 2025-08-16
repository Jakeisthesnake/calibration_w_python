import json
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict
from scipy.optimize import least_squares

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
    # print("V\n", V)
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


def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Rotation matrix around the X-axis (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation matrix around the Y-axis (pitch)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation matrix around the Z-axis (yaw)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Total rotation matrix by combining R_z, R_y, and R_x
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

# Example usage
rotation_matrix_1 = rpy_to_rotation_matrix(0, 0, 0)
translation_vector_1 = np.array([0, 0, 0])
rotation_matrix_2 = rpy_to_rotation_matrix(0, 0, 30)
translation_vector_2 = np.array([0, -1, 0])
rotation_matrix_3 = rpy_to_rotation_matrix(0, 30, 0)
translation_vector_3 = np.array([0, 0, -1])
rotation_matrix_4 = rpy_to_rotation_matrix(0, 0, -30)
translation_vector_4 = np.array([0, 1, 0])
rotation_matrix_5 = rpy_to_rotation_matrix(0, -30, 0)
translation_vector_5 = np.array([0, 0, 1])

# print("Rotation Matrix 1:\n", rotation_matrix_1)
# print("Rotation Matrix 2:\n", rotation_matrix_2)
# print("Rotation Matrix 3:\n", rotation_matrix_3)

# print("test")

r_t_1 = (np.array([rotation_matrix_1[0, :],
                  rotation_matrix_1[1, :],
                  translation_vector_1])).transpose()

r_t_2 = (np.array([rotation_matrix_2[0, :],
                  rotation_matrix_2[1, :],
                  translation_vector_2])).transpose()

r_t_3 = (np.array([rotation_matrix_3[0, :],
                  rotation_matrix_3[1, :],
                  translation_vector_3])).transpose()
r_t_4 = (np.array([rotation_matrix_4[0, :],
                  rotation_matrix_4[1, :],
                  translation_vector_4])).transpose()
r_t_5 = (np.array([rotation_matrix_5[0, :],
                  rotation_matrix_5[1, :],
                  translation_vector_5])).transpose()
# print("r_t_1: \n", r_t_1)
# print("r_t_2: \n", r_t_2)
# print("r_t_3: \n", r_t_3)
# print("r_t_4: \n", r_t_4)
# print("r_t_5: \n", r_t_5)

K_expected = np.array([
    [800,   0, 320],  # fx,  0, cx
    [  0, 800, 240],  #  0, fy, cy
    [  0,   0,   1]   #  0,  0,  1
])

H_1 = K_expected @ r_t_1
H_2 = K_expected @ r_t_2
H_3 = K_expected @ r_t_3
H_4 = K_expected @ r_t_4
H_5 = K_expected @ r_t_5

print("H_1: \n", H_1)
print("H_2: \n", H_2)
print("H_3: \n", H_3)
print("H_4: \n", H_4)
print("H_5: \n", H_5)

H_list = [H_1, H_2, H_3, H_4, H_5]
K_computed = compute_intrinsic_params(H_list)
K_computed /= K_computed[2,2]
# print("Computed Intrinsic Matrix:\n", K_computed)

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

R_expected, t_expected = compute_extrinsic_params(H_2, K_expected)
print("Expected Rotation Matrix:\n", R_expected)
print("Expected Translation Vector:\n", t_expected)

R_expected, t_expected = compute_extrinsic_params(H_3, K_expected)
print("Expected Rotation Matrix:\n", R_expected)
print("Expected Translation Vector:\n", t_expected)

R_expected, t_expected = compute_extrinsic_params(H_4, K_expected)
print("Expected Rotation Matrix:\n", R_expected)
print("Expected Translation Vector:\n", t_expected)
R_expected, t_expected = compute_extrinsic_params(H_5, K_expected)
print("Expected Rotation Matrix:\n", R_expected)
print("Expected Translation Vector:\n", t_expected)