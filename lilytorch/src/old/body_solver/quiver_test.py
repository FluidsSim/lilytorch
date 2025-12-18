from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt


plt.figure()
# Contour Plot
X, Y = np.meshgrid(
    np.linspace(0,2,100),
    np.linspace(-4,4,100)
)

Z = X*np.exp(-(X**2 + Y**2))
cp = plt.contourf(X, Y, Z,cmap="Greys")
cb = plt.colorbar(cp)

# Vector Field
X, Y = np.meshgrid(
    np.linspace(0,2,20),
    np.linspace(-4,4,20)
)

U =(1 - 2*(X**2))*np.exp(-((X**2)+(Y**2)))
V = -2*X*Y*np.exp(-((X**2)+(Y**2)))
speed = np.sqrt(U**2 + V**2)
UN = U/speed
VN = V/speed
quiv = plt.quiver(X, Y, UN, VN,  # assign to var
           color='Teal')

plt.figure()
plt.contourf(X,Y,UN,cmap="Greys")
plt.colorbar()

plt.show()