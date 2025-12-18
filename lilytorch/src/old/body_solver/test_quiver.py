
import matplotlib.pyplot as plt
import numpy as np

# Create grid
x, y = np.meshgrid(np.linspace(-100,100,2**5),np.linspace(-100,100,2**5))

dt=0.1
# Define the vector components (u and v are the x and y components of the vectors)
u = dt*np.sin(x)
v = dt*np.cos(y)

# Create a quiver plot
plt.scatter(x,y)
plt.quiver(x, y, u, v, scale=1, scale_units='xy')
plt.scatter(x+u,y+v)

plt.show()