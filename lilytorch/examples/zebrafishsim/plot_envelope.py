import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 1000)

c1 = +0.05,
c2 = -0.13,
c3 = +0.28
y  = c1+c2*x+c3*x**2

# y = 1 + 0.6 * (x - 1) + 0.31 * (x ** 2 - 1)

plt.plot(x, y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = 1 + 0.7(x - 1) + 0.31(x² - 1)')
plt.show()