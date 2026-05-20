import h5py, numpy as np, os, glob

ns_data = "/data/andreaferrario/ns_data"
# Find most recent directory
dirs = sorted(glob.glob(f"{ns_data}/2026-*"))
if not dirs:
    print("No simulation data found")
    exit()
latest = dirs[-1]
print(f"Latest: {latest}")

# Try to find and read animat data
import glob as g
h5files = g.glob(f"{latest}/**/*.h5", recursive=True) + g.glob(f"{latest}/**/*.hdf5", recursive=True)

if h5files:
    # Look for files that might contain link positions
    target_file = None
    for f_path in h5files:
        if "links" in f_path.lower() or "sensor" in f_path.lower():
             target_file = f_path
             break
    
    if not target_file:
        for f_path in h5files:
            if "ocean" not in f_path.lower():
                target_file = f_path
                break

    if not target_file:
        target_file = h5files[0]
        
    print(f"Opening: {target_file}")
    with h5py.File(target_file, 'r') as f:
        # Looking for things like 'link_0' or 'position'
        # We'll print datasets to find the right one
        def find_and_analyze(name, obj):
            if isinstance(obj, h5py.Dataset):
                if 'link_0' in name and 'pos' in name.lower():
                    data = obj[...]
                    if len(data.shape) >= 2:
                        # Assuming data is [time, coords] or similar
                        pos = data[:, :3] # x, y, z
                        diff = np.diff(pos, axis=0)
                        # We need time to compute speed. Let's look for a time dataset or assume 1 unit
                        # If we don't have time, we show displacement per step
                        dist = np.sqrt(np.sum(diff**2, axis=1))
                        avg_speed = np.mean(dist)
                        direction = pos[-1] - pos[0]
                        print(f"Dataset: {name}")
                        print(f"  Shape: {data.shape}")
                        print(f"  Average displacement per step: {avg_speed:.6f}")
                        print(f"  Total displacement: {np.linalg.norm(direction):.6f}")
                        print(f"  Start pos: {pos[0]}")
                        print(f"  End pos: {pos[-1]}")
                        print(f"  Direction vector: {direction}")
                elif 'head' in name.lower() and 'pos' in name.lower():
                    print(f"Found potential head position: {name} {obj.shape}")

        f.visititems(find_and_analyze)
else:
    print("No HDF5 files found.")
