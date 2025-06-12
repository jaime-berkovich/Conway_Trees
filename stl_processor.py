import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from functools import partial
import time

def estimate_triangles_from_size(file_path):
    """Estimate number of triangles in an STL file based on its size."""
    file_size = os.path.getsize(file_path)
    if file_size < 1000:  # ASCII STL
        return file_size // 100
    else:  # Binary STL
        return (file_size - 84) // 50

def load_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (normals, triangles)."""
    path = Path(path)
    raw = path.read_bytes()

    # Check if ASCII
    if raw[:5].lower() == b"solid" and b"facet normal" in raw[:200]:
        normals = []
        tris = []
        for line in raw.decode("ascii", errors="ignore").splitlines():
            tokens = line.strip().split()
            if tokens[:2] == ["facet", "normal"]:
                normals.append(list(map(float, tokens[2:])))
            elif tokens[:1] == ["vertex"]:
                tris.append(list(map(float, tokens[1:])))
        tris = np.asarray(tris, np.float32).reshape(-1, 3, 3)
        normals = np.repeat(np.asarray(normals, np.float32)[:, None, :], 3, 1)[:, 0]
        return normals, tris

    # Binary STL
    header = raw[:80]
    n_tri = struct.unpack("<I", raw[80:84])[0]
    dt = np.dtype([
        ("normals", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("v3", np.float32, (3,)),
        ("attr", np.uint16),
    ])
    body = np.frombuffer(raw, count=n_tri, offset=84, dtype=dt)
    normals = body["normals"]
    tris = np.stack([body["v1"], body["v2"], body["v3"]], axis=1)
    return normals, tris

def _ray_cast_chunk(chunk_points, tris, normals):
    """Process a chunk of points using a robust vectorized ray casting algorithm."""
    is_inside_mask = np.zeros(len(chunk_points), dtype=bool)
    ray_dir = np.array([1.0, 0.0, 0.0])

    # Pre-calculate triangle edges
    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0

    for i, point in enumerate(chunk_points):
        # --- Start of Möller–Trumbore algorithm, vectorized for all triangles ---
        T = point - v0
        P = np.cross(ray_dir, edge2)
        det = np.sum(edge1 * P, axis=1)

        # Find triangles that are not parallel to the ray
        non_parallel_mask = np.abs(det) > 1e-6
        if not np.any(non_parallel_mask):
            continue

        # Filter all arrays to only include non-parallel triangles
        inv_det = 1.0 / det[non_parallel_mask]
        T_filter = T[non_parallel_mask]
        P_filter = P[non_parallel_mask]
        edge1_filter = edge1[non_parallel_mask]
        edge2_filter = edge2[non_parallel_mask]
        normals_filter = normals[non_parallel_mask]
        
        # Calculate u and v parameters to find points within the triangle
        u = np.sum(T_filter * P_filter, axis=1) * inv_det
        Q = np.cross(T_filter, edge1_filter)
        v = np.sum(ray_dir * Q, axis=1) * inv_det

        # Find intersections that are within the triangle's bounds (u,v check)
        intersection_mask = (u >= 0) & (v >= 0) & (u + v <= 1)
        if not np.any(intersection_mask):
            continue

        # Filter for intersections within bounds
        Q_intersect = Q[intersection_mask]
        edge2_intersect = edge2_filter[intersection_mask]
        inv_det_intersect = inv_det[intersection_mask]
        normals_intersect = normals_filter[intersection_mask]

        # Calculate t parameter for valid intersections to check ray direction
        t = np.sum(edge2_intersect * Q_intersect, axis=1) * inv_det_intersect

        # Final check: t > epsilon (in front of ray origin) and correct normal direction
        final_mask = (t > 1e-6) & (np.sum(normals_intersect * ray_dir, axis=1) < 0)
        
        # The number of final intersections determines if the point is inside
        intersections = np.sum(final_mask)
        is_inside_mask[i] = (intersections % 2 == 1)

    return is_inside_mask

def stl_to_voxel_array(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Convert STL to voxel array using parallel ray casting."""
    start_time = time.time()
    print("Loading STL file...")
    normals, tris = load_stl(path)
    
    # Get bounding box
    all_points = tris.reshape(-1, 3)
    x_min, y_min, z_min = np.floor(all_points.min(axis=0))
    x_max, y_max, z_max = np.ceil(all_points.max(axis=0))
    
    # Ensure we capture the full 36mm x 30mm x 30mm structure
    x_min = min(x_min, 0)
    x_max = max(x_max, 35)
    y_min = min(y_min, 0)
    y_max = max(y_max, 29)
    z_min = min(z_min, 0)
    z_max = max(z_max, 35)  # <-- 36mm tall
    
    # Create voxel grid
    grid_shape = (int(z_max - z_min + 1), 
                 int(y_max - y_min + 1), 
                 int(x_max - x_min + 1))
    grid = np.zeros(grid_shape, dtype=bool)
    print(f"Grid shape: {grid_shape}")
    
    # Generate all voxel centers using arange for exact count
    z_coords = np.arange(z_min + 0.5, z_max + 1, 1)
    y_coords = np.arange(y_min + 0.5, y_max + 1, 1)
    x_coords = np.arange(x_min + 0.5, x_max + 1, 1)
    Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    print(f"Number of points: {len(points)} (should match grid size: {grid.size})")
    
    # Split points into chunks for parallel processing
    n_cores = mp.cpu_count()
    chunks = np.array_split(points, n_cores)
    
    print(f"Processing {len(points)} voxels using {n_cores} cores...")
    
    # Process chunks in parallel
    with mp.Pool(n_cores) as pool:
        non_empty_chunks = [chunk for chunk in chunks if len(chunk) > 0]
        results = pool.map(partial(_ray_cast_chunk, tris=tris, normals=normals), non_empty_chunks)

    # Combine results
    results = np.concatenate(results)
    print(f"Grid size: {grid.size}, Results size: {results.size}")
    if grid.size != results.size:
        print("WARNING: grid.size != results.size! Aborting assignment to prevent IndexError.")
        print(f"Grid shape: {grid_shape}, grid.size: {grid.size}, results.size: {results.size}")
        return None, None
    grid.ravel()[:] = results
    
    # Remove loading plates (3mm at top and bottom)
    mask = np.zeros_like(grid)
    mask[3:33] = True  # Keep only the middle 30mm (removes 3mm from top and bottom)
    grid &= mask
    
    # Create rotated grid
    filled_idx = np.column_stack(np.nonzero(grid)).astype(np.int32)
    
    if len(filled_idx) == 0:
        rotated_grid = np.zeros_like(grid)
        voxel_centers = np.array([], dtype=np.int32).reshape(0, 3)
    else:
        z, y, x = filled_idx.T
        
        # Rotate 90 degrees counterclockwise about z-axis
        new_x = y
        new_y = -x
        
        # Shift to be non-negative
        new_x -= np.min(new_x)
        new_y -= np.min(new_y)
        
        # Create new grid with appropriate size
        rotated_shape = (grid.shape[0], np.max(new_y) + 1, np.max(new_x) + 1)
        rotated_grid = np.zeros(rotated_shape, dtype=bool)
        
        # Fill the new grid
        rotated_grid[z, new_y, new_x] = True
        
        voxel_centers = np.column_stack(np.nonzero(rotated_grid)).astype(np.int32)

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    return voxel_centers, rotated_grid

def process_stl_file(stl_path, output_dir):
    """Process a single STL file and save its voxelized representation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Estimate triangles
    estimated_triangles = estimate_triangles_from_size(stl_path)
    print(f"Estimated triangles: {estimated_triangles:,}")
    
    # Convert STL to voxel array
    centers, matrix = stl_to_voxel_array(stl_path)
    
    # Get filename without extension
    filename = Path(stl_path).stem
    
    # Save the numpy array
    np.save(os.path.join(output_dir, f"{filename}.npy"), matrix)
    
    # Create interactive 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the voxels
    ax.voxels(matrix, facecolors='white', edgecolors='black', linewidth=0.3)
    
    # Set up the plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    plt.title(f"Voxelized Structure: {filename}\nEstimated triangles: {estimated_triangles:,}")
    
    # Enable interactive features
    ax.mouse_init()
    
    # Show the plot
    plt.show()

def process_directory(input_dir, output_dir):
    """Process all STL files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    stl_files = list(Path(input_dir).glob("**/*.stl"))
    print(f"Found {len(stl_files)} STL files in {input_dir}")
    
    for stl_file in stl_files:
        print(f"Processing {stl_file.name}...")
        process_stl_file(str(stl_file), output_dir) 