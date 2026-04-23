

from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc., if installed
import matplotlib.pyplot as plt
import torch
import numpy as np

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


# ...existing code...
# Generate a trajectory for the end effector with y-dependent speed

sampling_rate = 1000
tstop = 4
times = torch.arange(0, tstop, 1 / sampling_rate)
N = times.shape[0]

freq = 1
radius = 0.006
center = torch.tensor([0.01, 0.0])

# theta_base = 2 * np.pi * freq * times
# cos_tb = np.cos(theta_base)
# dy = np.concatenate([[0.0], np.diff(cos_tb)])  # length N

# k = 10
# smoothness = 25.0
# sf = 1.0 + k * np.tanh(smoothness * dy)  # length N

# s = np.cumsum(sf)
# s = (s - s[0]) / (s[-1] - s[0])  # normalize to [0,1]
# s_uniform = np.linspace(0.0, 1.0, N)
# t_vals = np.interp(s_uniform, s, theta_base)  # length N

# # convert to torch tensors with same length as times
# t_torch = torch.from_numpy(t_vals.astype(np.float32))
# x_traj = center[0] + torch.tensor(radius, dtype=t_torch.dtype) * torch.sin(t_torch)
# y_traj = center[1] + torch.tensor(radius, dtype=t_torch.dtype) * torch.cos(t_torch)


# theta      = 2 * np.pi * freq * times 

# dy         = -np.sin(theta)
# k          = 2.0
# smoothness = 40.0
# sf         = times + k * np.tanh( smoothness * dy)

# s          = np.cumsum(sf)
# s          = s / s[-1]
# s_uniform  = np.linspace(0.0, 1.0, sf.shape[0])
# t_vals     = np.interp(s_uniform, s, theta)
# t_torch = torch.from_numpy(t_vals.astype(np.float32))
# x_traj = center[0] + torch.tensor(radius, dtype=t_torch.dtype) * torch.sin(t_torch)
# y_traj = center[1] + torch.tensor(radius, dtype=t_torch.dtype) * torch.cos(t_torch)
# dt = 1 / sampling_rate
# s = torch.zeros_like(times)
# k = 0.9
# dsdt = 2 * np.pi * freq * (1 - k * torch.sin(s))
# s = torch.cat([torch.tensor([0.0]), torch.cumsum(dsdt[:-1] * dt, dim=0)])

s    = torch.zeros_like(times)
s[0] = 0.0
k    = 0.8
for i in range(1, N):
    dsdt = 2*np.pi*freq * (1 + k * np.sin(s[i-1]))
    s[i] = s[i-1] + dsdt/sampling_rate

# omega = 2 * np.pi * freq
# dt = 1 / sampling_rate
# s = torch.zeros_like(times)
# for _ in range(10):  # Iterate to approximate the solution
#     dsdt = omega * (1 + k * torch.sin(s))
#     s = torch.cat([torch.tensor([0.0]), torch.cumsum(dsdt[:-1] * dt, dim=0)])



x_traj = center[0] + torch.tensor(radius) * torch.sin(s)
y_traj = center[1] + torch.tensor(radius) * torch.cos(s)


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
    thigh, elbow = ComputeIK2D(-1, x, y, l1, l2)

    # Forward kinematics to get joint positions
    joint1 = torch.tensor([0.0, 0.0])
    joint2 = joint1 + l1 * angle2coordinates(thigh)
    ee     = joint2 + l2 * angle2coordinates(thigh + elbow)

    # Update link lines
    line.set_data([joint1[0], joint2[0], ee[0]], [joint1[1], joint2[1], ee[1]])

    # Update end effector and target
    ee_scatter.set_offsets([[ee[0], ee[1]]])
    target_scatter.set_offsets([[x, y]])

    return line, ee_scatter, target_scatter

fps = 60
step = max(1, int(sampling_rate / fps))
frames = range(0, N, step)
interval = int(1000 / fps)  # ms per frame

ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

plt.show()

# # Create figures directory if it doesn't exist
# os.makedirs('figures', exist_ok=True)

# ani.save('figures/ik_animation.mp4', writer='ffmpeg', fps=30)

