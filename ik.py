import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


L1 = 0.2
L2 = 2.0
L3 = 2.0
L4 = 0.5
key_axis_rotation = 0


def inverse_kinematics(x, y, z, L1, L2, L3, elbow_up=False):
    q1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    Xp = r
    Zp = z - L1

    D = np.sqrt(Xp**2 + Zp**2)
    if D > (L2 + L3) or D < abs(L2 - L3):
        raise ValueError("Target not reachable")

    cos_q3 = (L2**2 + L3**2 - D**2)/(2*L2*L3)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)

    # Choose elbow configuration
    if elbow_up:
        q3 = np.pi + np.arccos(cos_q3)
    else:
        q3 = np.pi - np.arccos(cos_q3)

    alpha = np.arctan2(Zp, Xp)
    beta = np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))
    q2 = alpha - beta

    return q1, q2, q3


def frame_vectors(q1, q2, q3):
    dx = np.cos(q1)
    dy = np.sin(q1)
    x_prime = np.array([dx, dy, 0.0])
    y_prime = np.array([-dy, dx, 0.0])
    z_prime = np.array([0.0, 0.0, 1.0])

    theta = q2 + q3
    x_double_prime = x_prime * np.cos(theta) + z_prime * np.sin(theta)
    z_double_prime = -x_prime * np.sin(theta) + z_prime * np.cos(theta)

    return x_double_prime, y_prime, z_double_prime

def compute_q4_q5(q1, q2, q3):
    x_final, y_final, z_final = frame_vectors(q1, q2, q3)

    # Compute q4 to point tool straight down (-Z)
    xz = x_final[2]   # z-component of x_final
    zz = z_final[2]   # z-component of z_final
    # q4 = atan2(-zz, -xz)
    q4 = np.arctan2(-zz, -xz)

    # Compute q5 to align horizontal axis with global X.
    # Simple approach: q5 = -q1
    q5 = -q1+key_axis_rotation

    return q4, q5


def forward_kinematics(q1, q2, q3, q4, q5, L1, L2, L3, L4):
    O0 = np.array([0.0, 0.0, 0.0])
    O1 = np.array([0.0, 0.0, L1])  # top of the vertical link

    dx = np.cos(q1)
    dy = np.sin(q1)
    x_dir = np.array([dx, dy, 0.0])
    z_dir = np.array([0.0, 0.0, 1.0])

    # Joint 2 position
    O2 = O1 + L2 * (np.cos(q2)*x_dir + np.sin(q2)*z_dir)
    # Joint 3 position
    O3 = O2 + L3 * (np.cos(q2+q3)*x_dir + np.sin(q2+q3)*z_dir)

    # After q4, the tool points down: tool_z = [0,0,-1].
    O4 = O3 + L4 * np.array([0,0,-1])

    # q5 rotates about the tool's down axis- correcting for q1
    q1_q5 = q1+q5
    tool_x_dir = np.array([np.cos(q1_q5), np.sin(q1_q5), 0.0])

    O5 = O4 + 0.5 * tool_x_dir  # 0.5 is an arbitrary length for visualization

    return O0, O1, O2, O3, O4, O5


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

arm_lines = []
for _ in range(5):
    line, = ax.plot([], [], [], 'o-', lw=2)
    arm_lines.append(line)

target_pt = ax.scatter([], [], [], c='r', s=50, label='Target')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('5-DOF Arm')
ax.legend()

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 3])

def target_trajectory(frame):
    angle = 2 * np.pi * (frame / 100)
    x_d = 1.0 + 0.5*np.cos(angle)
    y_d = 1.0 + 0.5*np.sin(angle)
    z_d = 1.0 + 0.5*np.sin(angle*2)
    return x_d, y_d, z_d

def update(frame):
    x_d, y_d, z_d = target_trajectory(frame)

    try:
        q1, q2, q3 = inverse_kinematics(x_d, y_d, z_d, L1, L2, L3, elbow_up=True)
        q4, q5 = compute_q4_q5(q1, q2, q3)
        print(f"Q1: {q1}")
        print(f"Q5: {q5}",)
    except ValueError:
        q1, q2, q3, q4, q5 = 0,0,0,0,0

    O0, O1, O2, O3, O4, O5 = forward_kinematics(q1, q2, q3, q4, q5, L1, L2, L3, L4)

    arm_points = [(O0,O1), (O1,O2), (O2,O3), (O3,O4), (O4,O5)]
    for i, seg in enumerate(arm_points):
        arm_lines[i].set_data([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]])
        arm_lines[i].set_3d_properties([seg[0][2], seg[1][2]])

    target_pt._offsets3d = ([x_d], [y_d], [z_d])

    return arm_lines + [target_pt]

anim = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()
