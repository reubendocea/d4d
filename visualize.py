#!/usr/bin/env python3
"""
Minimal Open3D visualization for D4D clips.
Shows Zivid and endoscope pointclouds overlaid in the same coordinate frame.
"""

import numpy as np
import open3d as o3d
import cv2
from pathlib import Path


def create_pointcloud_from_depth(depth_map, left_image, camera_params):
    """
    Create pointcloud from depth map and camera parameters.

    Args:
        depth_map: HxW numpy array of depth values (meters)
        left_image: HxWx3 BGR image
        camera_params: dict with keys 'fx', 'fy', 'cx', 'cy'

    Returns:
        o3d.geometry.PointCloud in camera frame
    """
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']

    H, W = depth_map.shape

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Back-project to 3D (camera frame)
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Stack into Nx3 array
    points = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)

    # Get colors from image (convert BGR to RGB and normalize)
    if left_image is not None:
        image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0
        colors = image_rgb.reshape(-1, 3)
    else:
        colors = np.ones_like(points) * 0.5  # Gray if no image

    # Filter valid points (depth > 0 and < 0.5m)
    valid_mask = (points[:, 2] > 0) & (points[:, 2] < 0.5)
    points = points[valid_mask]
    colors = colors[valid_mask]

    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def load_zivid_pointcloud(zivid_ply_path):
    """
    Load Zivid pointcloud from PLY file.

    Args:
        zivid_ply_path: Path to .ply file

    Returns:
        o3d.geometry.PointCloud
    """
    import trimesh

    zivid_path = Path(zivid_ply_path)
    if not zivid_path.exists():
        print(f"Zivid pointcloud not found: {zivid_path}")
        return None

    # Load with trimesh
    pcd = trimesh.load(str(zivid_path))

    # Extract pointcloud
    zivid_pcd = None
    if isinstance(pcd, trimesh.PointCloud):
        zivid_pcd = pcd
    elif isinstance(pcd, trimesh.Scene):
        for geom in pcd.geometry.values():
            if isinstance(geom, trimesh.PointCloud):
                zivid_pcd = geom
                break

    if zivid_pcd is None or not hasattr(zivid_pcd, 'vertices'):
        return None

    # Convert to Open3D
    vertices = np.asarray(zivid_pcd.vertices).astype(np.float64)
    colors = None
    if hasattr(zivid_pcd, 'colors') and zivid_pcd.colors is not None:
        colors = np.asarray(zivid_pcd.colors).astype(np.float64)

        # Handle RGBA -> RGB conversion (take only first 3 channels)
        if colors.ndim == 2 and colors.shape[1] == 4:
            colors = colors[:, :3]

        # Normalize to [0, 1] if needed
        if colors.max() > 1.0:
            colors = colors / 255.0

    # Subsample if too large
    if len(vertices) > 200000:
        indices = np.random.choice(len(vertices), size=200000, replace=False)
        vertices = vertices[indices]
        if colors is not None:
            colors = colors[indices]

    # Create Open3D pointcloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(vertices)
    if colors is not None and len(colors) == len(vertices) and colors.shape[1] == 3:
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default blue color for Zivid
        o3d_pcd.paint_uniform_color([0.3, 0.3, 0.8])

    return o3d_pcd


def create_camera_frustum(pose, color=[1, 0, 0], scale=0.02):
    """
    Create a camera frustum for visualization.

    Args:
        pose: 4x4 camera pose matrix (camera in world frame)
        color: RGB color [r, g, b]
        scale: Size of the frustum

    Returns:
        o3d.geometry.LineSet
    """
    # Define frustum points in camera frame
    points = np.array([
        [0, 0, 0],                          # Camera center
        [-scale, -scale, 2*scale],          # Top-left
        [scale, -scale, 2*scale],           # Top-right
        [scale, scale, 2*scale],            # Bottom-right
        [-scale, scale, 2*scale],           # Bottom-left
    ])

    # Transform to world frame
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    points_world = (pose @ points_hom.T).T[:, :3]

    # Define lines
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle
    ]

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

    return line_set


def visualize_clip_state(zivid_ply_path, endoscope_left_image_path, endoscope_depth_path,
                          endoscope_camera_params, endoscope_pose,
                          window_name="Clip Visualization"):
    """
    Visualize the initial state of a clip with Zivid and endoscope pointclouds.

    Args:
        zivid_ply_path: Path to Zivid .ply file (already in world/Zivid frame)
        endoscope_left_image_path: Path to left endoscope image
        endoscope_depth_path: Path to endoscope depth map (.npy file)
        endoscope_camera_params: dict with 'fx', 'fy', 'cx', 'cy' for endoscope
        endoscope_pose: 4x4 numpy array - endoscope camera pose in world/Zivid frame
        window_name: Name of the visualization window
    """
    geometries = []

    # 1. Load Zivid pointcloud (already in world frame)
    print(f"Loading Zivid pointcloud from {zivid_ply_path}...")
    zivid_pcd = load_zivid_pointcloud(zivid_ply_path)
    if zivid_pcd is not None:
        print(f"  Loaded {len(zivid_pcd.points)} Zivid points")
        geometries.append(zivid_pcd)
    else:
        print("  Failed to load Zivid pointcloud")

    # 2. Load endoscope depth and image
    print(f"Loading endoscope depth from {endoscope_depth_path}...")
    depth_map = np.load(endoscope_depth_path)
    left_image = cv2.imread(endoscope_left_image_path)

    if left_image is None:
        print(f"  Warning: Could not load image from {endoscope_left_image_path}")

    print(f"  Depth map shape: {depth_map.shape}")
    if left_image is not None:
        print(f"  Image shape: {left_image.shape}")

    # 3. Create endoscope pointcloud in camera frame
    print("Creating endoscope pointcloud...")
    endoscope_pcd_camera = create_pointcloud_from_depth(depth_map, left_image, endoscope_camera_params)
    print(f"  Created {len(endoscope_pcd_camera.points)} endoscope points")

    # 4. Transform endoscope pointcloud to world frame
    print("Transforming endoscope pointcloud to world frame...")
    endoscope_pcd_world = o3d.geometry.PointCloud(endoscope_pcd_camera)
    endoscope_pcd_world.transform(endoscope_pose)
    geometries.append(endoscope_pcd_world)

    # 5. Create camera frustum
    print("Creating camera frustum...")
    frustum = create_camera_frustum(endoscope_pose, color=[1, 0, 0], scale=0.02)
    geometries.append(frustum)

    # 6. Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # 7. Visualize
    if geometries:
        print(f"\nVisualizing {len(geometries)} geometries...")
        print("Legend:")
        print("  Blue/colored: Zivid pointcloud (ground truth)")
        print("  Colored: Endoscope pointcloud (stereo reconstruction)")
        print("  Red frustum: Endoscope camera pose")
        print("  RGB axes: Coordinate frame (X=red, Y=green, Z=blue)")
        print("\nPress Q to close the window")

        # Create visualizer with custom view
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720, left=100, top=100)

        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)

        # Set rendering options for point clouds (use points instead of squares)
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        # Set camera view to match endoscope perspective
        view_control = vis.get_view_control()

        # Get camera intrinsics
        fx = endoscope_camera_params.get('fx', 571.5)
        fy = endoscope_camera_params.get('fy', 572.5)
        cx = endoscope_camera_params.get('cx', 312.0)
        cy = endoscope_camera_params.get('cy', 249.0)
        width = endoscope_camera_params.get('width', 640)
        height = endoscope_camera_params.get('height', 512)

        # Create intrinsic matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        # The endoscope_pose is camera-to-world transform
        # Open3D expects world-to-camera (extrinsic), which is the inverse
        extrinsic = np.linalg.inv(endoscope_pose)

        # Create camera parameters
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = intrinsic
        camera_params.extrinsic = extrinsic

        # Convert from pinhole camera parameters
        view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

        # Run visualizer
        vis.run()
        vis.destroy_window()
    else:
        print("No geometries to visualize!")
