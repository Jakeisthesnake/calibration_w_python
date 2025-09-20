import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def kb_projection(theta, k1, k2, k3, k4):
    """Kannala–Brandt projection function."""
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta4 * theta2
    theta8 = theta4**2
    theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
    return theta_d

def kb_derivative(theta, k1, k2, k3, k4):
    """Derivative of theta_d wrt theta (for monotonicity check)."""
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta4 * theta2
    theta8 = theta4**2
    return (1 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8)

# Initial coefficients
init_k1, init_k2, init_k3, init_k4 = 0.0, 0.0, 0.0, 0.0
fov_deg = 180
theta = np.linspace(0, np.deg2rad(fov_deg), 1000)

# --- Set up figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(left=0.25, bottom=0.35)

# Plot θ→θ_d curve
theta_d = kb_projection(theta, init_k1, init_k2, init_k3, init_k4)
line1, = ax1.plot(np.rad2deg(theta), theta_d, lw=2, label=r"$\theta_d(\theta)$")
ax1.plot(np.rad2deg(theta), theta, 'k--', alpha=0.5, label="Identity (θ_d=θ)")
ax1.set_xlabel("Incident angle θ [deg]")
ax1.set_ylabel("Distorted angle θ_d [rad]")
ax1.set_title("Kannala–Brandt Mapping")
ax1.grid(True)
ax1.legend()

# Circle projection subplot
circle_angles = np.linspace(0, 2*np.pi, 40)
circle_points = np.stack([np.cos(circle_angles), np.sin(circle_angles)], axis=1)

# Start with no distortion (identity)
r = 1.0
circle_proj = circle_points * r
scat2 = ax2.scatter(circle_proj[:,0], circle_proj[:,1], c='blue', s=10)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("Projected Circle in Image Plane")
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.grid(True)

# Define slider axes
ax_k1 = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_k2 = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_k3 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_k4 = plt.axes([0.25, 0.10, 0.65, 0.03])

# Create sliders
s_k1 = Slider(ax_k1, 'k1', -1.0, 1.0, valinit=init_k1, valstep=0.01)
s_k2 = Slider(ax_k2, 'k2', -1.0, 1.0, valinit=init_k2, valstep=0.01)
s_k3 = Slider(ax_k3, 'k3', -1.0, 1.0, valinit=init_k3, valstep=0.01)
s_k4 = Slider(ax_k4, 'k4', -1.0, 1.0, valinit=init_k4, valstep=0.01)

def update(val):
    k1, k2, k3, k4 = s_k1.val, s_k2.val, s_k3.val, s_k4.val
    
    # --- θ_d curve update ---
    theta_d = kb_projection(theta, k1, k2, k3, k4)
    line1.set_ydata(theta_d)

    dtheta_d = kb_derivative(theta, k1, k2, k3, k4)
    bad = dtheta_d <= 0
    for coll in ax1.collections:
        coll.remove()
    if np.any(bad):
        ax1.scatter(np.rad2deg(theta[bad]), theta_d[bad], color='red', s=10, label="Non-monotonic")
    ax1.relim()
    ax1.autoscale_view()

    # --- Circle projection update ---
    # Define circle in terms of azimuth φ and fixed θ
    phi = np.linspace(-1*np.pi, np.pi, 20)
    
    theta_d_circle = kb_projection(phi, k1, k2, k3, k4)

    # In normalized image plane, radial distance = tan(theta_d)
    
    x = np.cos(theta_d_circle)
    y = np.sin(theta_d_circle)

    circle_proj = np.stack([x, y], axis=1)
    scat2.set_offsets(circle_proj)

    ax2.set_xlim(circle_proj[:,0].min()-0.1, circle_proj[:,0].max()+0.1)
    ax2.set_ylim(circle_proj[:,1].min()-0.1, circle_proj[:,1].max()+0.1)

    fig.canvas.draw_idle()

# Connect sliders
s_k1.on_changed(update)
s_k2.on_changed(update)
s_k3.on_changed(update)
s_k4.on_changed(update)

plt.show()
