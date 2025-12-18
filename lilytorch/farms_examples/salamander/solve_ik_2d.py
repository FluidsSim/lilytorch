

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch

def ComputeIK2D(elbowID, x, y, d1, d2):
    """
    elbowID=1 - elbow down
    elbowID=-1 - elbow up
    """
    D=((x**2+y**2-d1**2-d2**2)/(2*d1*d2)).clip(-1,1)
    elbow=elbowID*torch.acos(D)
    k1=d1+d2*torch.cos(elbow)
    k2=d2*torch.sin(elbow)
    thigh=torch.atan2(y,x)-torch.atan2(k2,k1)
    return (thigh, elbow)


def angle2coordinates(theta):
    return torch.tensor([torch.cos(theta), torch.sin(theta)])

def rot_mat2d(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.tensor([[c, -s],
                         [s, c]])



# Generate a trajectory for the end effector (e.g., a circle in 2D)
t_vals = torch.linspace(0, 10*2*torch.pi, 1000)
dt = t_vals[1] - t_vals[0]
radius = 0.003
center = torch.tensor([0.008, 0.0])
x_traj = center[0] + radius * torch.sin(t_vals)
y_traj = center[1] + radius * torch.cos(t_vals)

l1 = torch.tensor(0.006)
l2 = torch.tensor(0.006)

fig, ax = plt.subplots()

line, = ax.plot([], [], 'bo-', label='Links', markersize=8)
ee_scatter = ax.scatter([], [], color='r', s=100, label='End Effector')
target_scatter = ax.scatter([], [], color='k', s=50, marker='s', label='Target')

ax.set_xlim(-0.001, 0.013)
ax.set_ylim(-0.007, 0.007)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D 2-Link Inverse Kinematics Animation')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')


def update(frame):
    x = x_traj[frame]
    y = y_traj[frame]
    thigh, elbow = ComputeIK2D(1, x, y, l1, l2)

    # Forward kinematics to get joint positions
    joint1 = torch.tensor([0.0, 0.0])
    joint2 = joint1 + l1 * angle2coordinates(thigh)
    ee = joint2 + l2 * angle2coordinates(thigh + elbow)

    # Update link lines
    line.set_data([joint1[0], joint2[0], ee[0]], [joint1[1], joint2[1], ee[1]])

    # Update end effector and target
    ee_scatter.set_offsets([[ee[0], ee[1]]])
    target_scatter.set_offsets([[x, y]])

    return line, ee_scatter, target_scatter

ani = FuncAnimation(fig, update, frames=len(t_vals), interval=50, blit=False)

ani.save('figures/ik_animation.mp4', writer='ffmpeg', fps=30)

