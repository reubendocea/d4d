# D4D - The Dresden Dataset for 4D Reconstruction

Dataset loader for **D4D - The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**.

Hierarchical dataset loader for surgical stereo reconstruction with Open3D visualization.

## Installation

```bash
pip install -e .
```

## Usage

```python
from d4d.loader import D4D
from visualize import visualize_clip_state

# Load dataset
d4d = D4D("/path/to/preprocessed_restructured")

# Iterate: Dataset → Specimen → Session → Clip
for specimen in d4d:
    for session in specimen:
        for clip in session:
            print(f"{clip.name}: {len(clip.left_img_paths)} images, {clip.duration:.1f}s")

# Access specific clip
specimen = next(iter(d4d))
session = next(iter(specimen))
clip = next(iter(session))

# Clip properties
clip.left_img_paths          # List of left image paths
clip.right_img_paths         # List of right image paths
clip.stereo_depth_paths      # List of depth map paths
clip.pointclouds             # Dict with 'start'/'end' Zivid PLY paths
clip.endoscope_params        # Endoscope camera parameters (fx, fy, cx, cy, baseline, width, height)
clip.zivid_params            # Zivid camera parameters
clip.poses                   # Dict with 'start'/'end' curated camera poses (4x4 matrices)

# Visualize with curated poses
if clip.pointclouds.get('start') and clip.left_img_paths and clip.stereo_depth_paths:
    visualize_clip_state(
        clip.pointclouds['start'],
        clip.left_img_paths[0],
        clip.stereo_depth_paths[0],
        clip.endoscope_params,
        clip.poses['start']
    )
```

## Dataset Structure

```
d4d_dataset/
└── specimen_1/
    └── 2025_03_06-16_49_40/
        ├── clips.json
        ├── camera_info/
        ├── pointcloud/
        └── clips/
            └── Clip_1/
                ├── left_images_rect/
                ├── right_images_rect/
                ├── stereo_depth/
                └── camera_info/
```

## Dependencies

numpy, opencv-python, PyYAML, open3d, trimesh, tqdm 