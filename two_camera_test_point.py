import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define rotation matrix for 30 degrees around Y-axis
theta = np.radians(30)
R_y = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

# Define translations
t_cam1 = np.array([0, 0, 0])  # Camera 1 at origin
t_cam2 = np.array([1, 0, 0])  # Camera 2 at (1,0,0)

# Define world point
P_world = np.array([0, 0, 2])

# Transform into camera frames
P_cam1 = P_world  # Identity transform
P_cam2 = np.linalg.inv(R_y) @ (P_world - t_cam2)
print(f"World Point: {P_world}")
print(f"Point in Cam1 Frame: {P_cam1}")
print(f"Point in Cam2 Frame: {P_cam2}")

# Create figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot world frame axes
ax.quiver(0, 0, 0, 1, 0, 0, color='r', linestyle='dashed', label="World X-axis")
ax.quiver(0, 0, 0, 0, 1, 0, color='g', linestyle='dashed', label="World Y-axis")
ax.quiver(0, 0, 0, 0, 0, 1, color='b', linestyle='dashed', label="World Z-axis")

# Plot camera 1 axes
ax.quiver(*t_cam1, 1, 0, 0, color='r', label="Cam1 X-axis")
ax.quiver(*t_cam1, 0, 1, 0, color='g', label="Cam1 Y-axis")
ax.quiver(*t_cam1, 0, 0, 1, color='b', label="Cam1 Z-axis")

# Plot camera 2 axes (rotated frame)
rotated_x = R_y @ np.array([1, 0, 0])
rotated_z = R_y @ np.array([0, 0, 1])
ax.quiver(*t_cam2, *rotated_x, color='r', linestyle='dotted', label="Cam2 X-axis")
ax.quiver(*t_cam2, 0, 1, 0, color='g', linestyle='dotted', label="Cam2 Y-axis")
ax.quiver(*t_cam2, *rotated_z, color='b', linestyle='dotted', label="Cam2 Z-axis")

# Plot points
ax.scatter(*P_world, color='black', label="World Point (0,0,2)", s=100)
ax.scatter(*P_cam1, color='purple', label="Point in Cam1 Frame", s=100)
ax.scatter(*P_cam2, color='orange', label="Point in Cam2 Frame", s=100)

# Labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Camera Transformations Visualization")
ax.legend()
ax.view_init(elev=20, azim=30)  # Adjust viewing angle

plt.show()
