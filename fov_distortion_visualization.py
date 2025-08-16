import numpy as np
import matplotlib.pyplot as plt

def kannala_brandt_distortion(theta, coeffs):
    """Compute distorted angle using the Kannala-Brandt model."""
    k1, k2, k3, k4 = coeffs
    theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    return theta_d

def plot_angle_distortion(coeffs):
    """Plot undistorted angle vs. distorted angle."""
    theta = np.linspace(0, np.pi/2, 100)  # From 0 to 90 degrees
    theta_d = kannala_brandt_distortion(theta, coeffs)
    theta_pinhole = np.tan(theta)
    
    plt.figure(figsize=(6, 4))
    plt.plot(np.degrees(theta), np.degrees(theta_d), label='Distorted Angle')
    plt.plot(np.degrees(theta), np.degrees(theta), linestyle='dashed', label='Ideal (No Distortion)')
    plt.plot(np.degrees(theta), np.degrees(theta_pinhole), linestyle='dotted', label='Pinhole Camera')
    plt.xlabel('Undistorted Angle (degrees)')
    plt.ylabel('Distorted Angle (degrees)')
    plt.title('Kannala-Brandt Distortion Effect')
    plt.xlim(0, 100)
    plt.ylim(0, 150)
    plt.legend()
    plt.grid()
    plt.show()

def generate_fisheye_grid(coeffs, size=5, resolution=21):
    """Generate a grid and apply Kannala-Brandt distortion."""
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    theta = np.arctan2(r, 1.0)  # Assume focal length = 1
    theta_d = kannala_brandt_distortion(theta, coeffs)
    scale = theta_d / (theta + 1e-8)
    Xd, Yd = X * scale, Y * scale
    
    plt.figure(figsize=(6, 6))
    plt.plot(Xd, Yd, 'k', alpha=0.6)  # Vertical lines
    plt.plot(Xd.T, Yd.T, 'k', alpha=0.6)  # Horizontal lines
    plt.xlim(-size, size)
    plt.ylim(-size, size)
    plt.title('Fisheye Grid Distortion')
    plt.axis('equal')
    plt.show()

# Given distortion coefficients
coeffs = [0.33301244, 0.13635289, 0.04196859, 0.04174295]

# Plot results
plot_angle_distortion(coeffs)
# generate_fisheye_grid(coeffs)
