
import jax.numpy as jnp
import jax.scipy.sparse.linalg

N = 50
L = 1.0
h = L / (N + 1)
x = jnp.linspace(h, L - h, N)
y = jnp.linspace(h, L - h, N)
X, Y = jnp.meshgrid(x, y, indexing='ij')

def matvec(u_flat):
    u = u_flat.reshape(N, N)
    u_new = -4*u
    u_new += jnp.roll(u, 1, axis=0)
    u_new += jnp.roll(u, -1, axis=0)
    u_new += jnp.roll(u, 1, axis=1)
    u_new += jnp.roll(u, -1, axis=1)
    u_new.at[0,0].set(0)  # Dirichlet BC at x=0
    return (u_new / h**2).flatten()


f = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
f_flat = f.flatten()

u_flat, exit_code = jax.scipy.sparse.linalg.gmres(matvec, f_flat)


# u_flat, exit_code = jax.scipy.linalg.solve(matvec, f_flat)



import matplotlib.pyplot as plt

u = u_flat.reshape(N, N)

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of 2D Poisson Equation')
plt.show()
