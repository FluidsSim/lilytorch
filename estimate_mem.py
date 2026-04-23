def estimate_memory():
    # 3D Grid dimensions (including the +2 ghost layers added by FluidSolver)
    nx = 1024 + 2
    ny = 128 + 2
    nz = 128 + 2
    cells = nx * ny * nz
    
    # 64-bit precision (8 bytes per float)
    bytes_per_float = 8
    grid_bytes = cells * bytes_per_float
    
    print(f"Total cells: {cells:e}")
    print(f"Bytes per 3D grid: {grid_bytes / 1024**2:.2f} MB")
    
    # Estimate typical number of large grids required:
    # Variables: u0, u1, v0, v1, w0, w1, p, div, sdf_val (at least 9 core arrays)
    # Plus intermediate computations, forces, right-hand-sides for Poisson, etc.
    # A conservative estimate is 20-30 large arrays overall to solve BDIM.
    
    num_grids = 25
    total_mb = (grid_bytes * num_grids) / 1024**2
    total_gb = total_mb / 1024
    
    print(f"Estimated baseline state memory (for ~{num_grids} grids): {total_gb:.2f} GB")
    
    # The FFT solver precomputes/caches wave numbers and transforms which in complex128
    # takes roughly 2x space for forward/inverse transforms.
    # PyTorch memory overhead usually doubles this peak memory during backprop or intermediate ops.
    print(f"Expected peak VRAM usage: ~{total_gb * 2.5:.2f} to {total_gb * 3.5:.2f} GB")

estimate_memory()
