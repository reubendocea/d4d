#!/usr/bin/env python3

import os
import json
import yaml
import glob
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Iterator
from pathlib import Path
from tqdm import tqdm


@dataclass
class CameraInfo:
    """Camera calibration information."""
    width: int
    height: int
    K: np.ndarray  # Camera matrix
    D: np.ndarray  # Distortion parameters
    R: np.ndarray  # Rectification matrix
    P: np.ndarray  # Projection matrix


@dataclass
class StereoParameters:
    """Stereo camera parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


def _extract_matrix(data: dict, short_name: str, long_name: str, shape: tuple, default: np.ndarray) -> np.ndarray:
    """Helper to extract matrix from YAML data with multiple naming conventions."""
    for key in [short_name, long_name]:
        if key in data:
            value = data[key]
            if isinstance(value, dict) and 'data' in value:
                value = value['data']
            return np.array(value).reshape(shape) if shape else np.array(value)
    return default


def load_camera_info(yaml_path: str) -> CameraInfo:
    """Load camera calibration from yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    width = data.get('width', data.get('image_width', 0))
    height = data.get('height', data.get('image_height', 0))

    K = _extract_matrix(data, 'K', 'camera_matrix', (3, 3), np.eye(3))
    D = _extract_matrix(data, 'D', 'distortion_coefficients', None, np.zeros(5))
    R = _extract_matrix(data, 'R', 'rectification_matrix', (3, 3), np.eye(3))
    P = _extract_matrix(data, 'P', 'projection_matrix', (3, 4), None)

    if P is None:
        P = np.zeros((3, 4))
        P[:3, :3] = K

    return CameraInfo(width=width, height=height, K=K, D=D, R=R, P=P)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return img


class Clip:
    """Represents a single clip with all its associated data."""

    def __init__(self, clip_path: Path, session: 'Session'):
        self.path = Path(clip_path)
        self.session = session
        self.name = self.path.name

        # Metadata from session's clips.json
        self._metadata = None
        self._left_camera_info = None
        self._right_camera_info = None

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load clip metadata from session's clips.json."""
        clips_json_path = self.session.path / 'clips.json'

        if not clips_json_path.exists():
            return

        try:
            with open(clips_json_path, 'r') as f:
                data = json.load(f)

            # Find this clip's metadata
            for clip_data in data.get('clips', []):
                if clip_data.get('name') == self.name:
                    self._metadata = clip_data
                    break
        except Exception as e:
            print(f"Warning: Could not load clip metadata: {e}")

    @property
    def metadata(self) -> Optional[Dict]:
        """Get clip metadata from clips.json."""
        return self._metadata

    @property
    def start_timestamp(self) -> Optional[float]:
        """Get start timestamp from metadata."""
        if self._metadata:
            return self._metadata.get('start', {}).get('timestamp')
        return None

    @property
    def end_timestamp(self) -> Optional[float]:
        """Get end timestamp from metadata."""
        if self._metadata:
            return self._metadata.get('end', {}).get('timestamp')
        return None

    @property
    def duration(self) -> Optional[float]:
        """Get clip duration in seconds."""
        if self.start_timestamp is not None and self.end_timestamp is not None:
            return self.end_timestamp - self.start_timestamp
        return None

    def _get_camera_info(self, side: str) -> Optional[CameraInfo]:
        """Helper to get camera calibration info for left or right camera."""
        cache_attr = f'_{side}_camera_info'
        cached = getattr(self, cache_attr)

        if cached is None:
            # Try clip-level camera_info first (various naming patterns)
            cam_info_dir = self.path / 'camera_info'
            if cam_info_dir.exists():
                patterns = [f'{side}_rect*.yaml', f'{side}.yaml']
                for pattern in patterns:
                    yaml_files = list(cam_info_dir.glob(pattern))
                    if yaml_files:
                        cached = load_camera_info(str(yaml_files[0]))
                        break

            # Fall back to session-level camera_info if not found
            if cached is None:
                session_yaml = self.session.path / 'camera_info' / f'{side}.yaml'
                if session_yaml.exists():
                    cached = load_camera_info(str(session_yaml))

            setattr(self, cache_attr, cached)
        return cached

    @property
    def left_camera_info(self) -> Optional[CameraInfo]:
        """Get left camera calibration info."""
        return self._get_camera_info('left')

    @property
    def right_camera_info(self) -> Optional[CameraInfo]:
        """Get right camera calibration info."""
        return self._get_camera_info('right')

    def get_stereo_parameters(self) -> Optional[StereoParameters]:
        """Extract stereo camera parameters."""
        if not self.left_camera_info or not self.right_camera_info:
            return None

        left_P = self.left_camera_info.P
        right_P = self.right_camera_info.P

        fx = left_P[0, 0]
        fy = left_P[1, 1]
        cx = left_P[0, 2]
        cy = left_P[1, 2]
        baseline = abs(right_P[0, 3] / right_P[0, 0] - left_P[0, 3] / left_P[0, 0])

        return StereoParameters(fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)

    def get_image_paths(self, rectified: bool = True) -> Tuple[List[str], List[str]]:
        """
        Get lists of left and right image paths.

        Args:
            rectified: If True, get rectified images; otherwise get raw images

        Returns:
            Tuple of (left_image_paths, right_image_paths)
        """
        suffix = '_rect' if rectified else ''
        paths = []

        for side in ['left', 'right']:
            img_dir = self.path / f'{side}_images{suffix}'
            # Fall back to session-level for non-rectified if clip-level doesn't exist
            if not rectified and not img_dir.exists():
                img_dir = self.session.path / f'{side}_images'

            images = sorted(img_dir.glob('*.png')) if img_dir.exists() else []
            paths.append([str(p) for p in images])

        return paths[0], paths[1]

    def load_images(self, rectified: bool = True, progress: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load all images in the clip.

        Args:
            rectified: If True, load rectified images; otherwise load raw images
            progress: If True, show progress bar

        Returns:
            Tuple of (left_images_list, right_images_list)
        """
        left_paths, right_paths = self.get_image_paths(rectified=rectified)

        left_images_list = []
        right_images_list = []

        min_images = min(len(left_paths), len(right_paths))

        iterator = range(min_images)
        if progress:
            iterator = tqdm(iterator, desc=f"Loading images for {self.name}", unit="images")

        for i in iterator:
            left_img = load_image(left_paths[i])
            right_img = load_image(right_paths[i])

            left_images_list.append(left_img)
            right_images_list.append(right_img)

        return left_images_list, right_images_list

    def load_depth_maps(self) -> List[np.ndarray]:
        """Load stereo depth maps if available."""
        depth_dir = self.path / 'stereo_depth'
        if not depth_dir.exists():
            return []

        depth_files = sorted(depth_dir.glob('*.npy'))
        return [np.load(str(f)) for f in depth_files]

    def load_masks(self, mask_type: str = 'zivid') -> Dict[str, np.ndarray]:
        """
        Load masks for the clip.

        Args:
            mask_type: Type of masks to load ('zivid', 'rectified', or 'original')

        Returns:
            Dictionary with mask names as keys and mask arrays as values
        """
        mask_dirs = {
            'zivid': self.path / 'zivid_masks',
            'rectified': self.path / 'left_images_rect_masks',
            'original': self.session.path / 'masks' / 'original_masks' / f'{self.name}_masks',
        }

        mask_dir = mask_dirs.get(mask_type)
        if not mask_dir or not mask_dir.exists():
            return {}

        masks = {}
        for mask_file in sorted(mask_dir.glob('*.png')):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[mask_file.stem] = mask

        return masks

    def load_zivid_images(self) -> Dict[str, np.ndarray]:
        """Load Zivid color images (start and end)."""
        zivid_dir = self.path / 'zivid_images'
        images = {}

        if zivid_dir.exists():
            for img_file in zivid_dir.glob('*.png'):
                img = cv2.imread(str(img_file))
                if img is not None:
                    images[img_file.stem] = img

        return images

    def load_pose_bounds(self) -> Optional[np.ndarray]:
        """Load pose bounds array."""
        pose_bounds_file = self.path / 'pose_bounds.npy'
        if pose_bounds_file.exists():
            return np.load(str(pose_bounds_file))
        return None

    def load_camera_poses(self) -> Dict[str, np.ndarray]:
        """
        Load curated camera poses.

        Returns:
            Dictionary with 'start' and 'end' poses
        """
        poses = {}

        for position in ['start', 'end']:
            pose_file = self.path / f'curated_camera_pose_{position}.txt'
            if pose_file.exists():
                poses[position] = np.loadtxt(str(pose_file))

        return poses

    # Convenience properties for easy data access
    @property
    def left_img_paths(self) -> List[str]:
        """Get list of left image paths (rectified)."""
        left_paths, _ = self.get_image_paths(rectified=True)
        return left_paths

    @property
    def right_img_paths(self) -> List[str]:
        """Get list of right image paths (rectified)."""
        _, right_paths = self.get_image_paths(rectified=True)
        return right_paths

    @property
    def stereo_depth_paths(self) -> List[str]:
        """Get list of stereo depth map paths."""
        depth_dir = self.path / 'stereo_depth'
        if depth_dir.exists():
            return [str(p) for p in sorted(depth_dir.glob('*.npy'))]
        return []

    @property
    def pointclouds(self) -> Dict[str, str]:
        """
        Get Zivid pointcloud paths for start and end.
        Reads from session's clips.json (start_geometry and end_geometry fields).
        """
        pointclouds = {}

        if self._metadata:
            for key in ['start', 'end']:
                geom = self._metadata.get(f'{key}_geometry')
                if geom:
                    path = self.session.path / geom
                    if path.exists():
                        pointclouds[key] = str(path)

        return pointclouds

    @property
    def endoscope_params(self) -> Optional[Dict[str, float]]:
        """Get endoscope camera parameters from stereo projection matrix."""
        stereo_params = self.get_stereo_parameters()
        left_cam_info = self.left_camera_info
        if stereo_params and left_cam_info:
            return {
                'fx': stereo_params.fx,
                'fy': stereo_params.fy,
                'cx': stereo_params.cx,
                'cy': stereo_params.cy,
                'baseline': stereo_params.baseline,
                'width': left_cam_info.width,
                'height': left_cam_info.height,
            }
        return None

    @property
    def endoscope_intrinsics(self) -> Optional[Dict[str, float]]:
        """Get endoscope camera intrinsics directly from left camera K matrix."""
        left_cam_info = self.left_camera_info
        if left_cam_info:
            stereo_params = self.get_stereo_parameters()
            return {
                'fx': left_cam_info.K[0, 0],
                'fy': left_cam_info.K[1, 1],
                'cx': left_cam_info.K[0, 2],
                'cy': left_cam_info.K[1, 2],
                'baseline': stereo_params.baseline if stereo_params else 0.0,
                'width': left_cam_info.width,
                'height': left_cam_info.height,
            }
        return None

    @property
    def zivid_params(self) -> Optional[Dict[str, float]]:
        """Get Zivid camera parameters from session-level color_camera_info.yaml."""
        color_camera_yaml = self.session.path / 'camera_info' / 'color_camera_info.yaml'

        if not color_camera_yaml.exists():
            return None

        try:
            camera_info = load_camera_info(str(color_camera_yaml))
            return {
                'fx': camera_info.K[0, 0],
                'fy': camera_info.K[1, 1],
                'cx': camera_info.K[0, 2],
                'cy': camera_info.K[1, 2],
                'width': camera_info.width,
                'height': camera_info.height,
            }
        except Exception as e:
            print(f"Warning: Could not load Zivid camera parameters: {e}")
            return None

    @property
    def poses(self) -> Dict[str, np.ndarray]:
        """Get curated camera poses for start and end."""
        return self.load_camera_poses()

    def __repr__(self) -> str:
        return f"Clip(name='{self.name}', duration={self.duration:.2f}s)" if self.duration else f"Clip(name='{self.name}')"


class Session:
    """Represents a single recording session."""

    def __init__(self, session_path: Path, specimen: 'Specimen'):
        self.path = Path(session_path)
        self.specimen = specimen
        self.name = self.path.name
        self.clips_json_path = self.path / 'clips.json'
        self._clips_metadata = None

    @property
    def clips_metadata(self) -> Dict:
        """Get clips metadata from clips.json."""
        if self._clips_metadata is None and self.clips_json_path.exists():
            with open(self.clips_json_path, 'r') as f:
                self._clips_metadata = json.load(f)
        return self._clips_metadata or {}

    def get_clip_names(self) -> List[str]:
        """Get list of clip names from clips.json."""
        if not self.clips_json_path.exists():
            return []

        try:
            with open(self.clips_json_path, 'r') as f:
                data = json.load(f)
            return [clip['name'] for clip in data.get('clips', [])]
        except Exception as e:
            print(f"Warning: Could not parse clips.json: {e}")
            return []

    def __iter__(self) -> Iterator[Clip]:
        """Iterate over clips in this session."""
        clips_dir = self.path / 'clips'

        if not clips_dir.exists():
            return

        # Get clip names from clips.json for proper ordering
        clip_names = self.get_clip_names()

        # If no clips.json, fall back to directory listing
        if not clip_names:
            clip_names = [d.name for d in sorted(clips_dir.iterdir())
                         if d.is_dir() and d.name.startswith('Clip_')]

        for clip_name in clip_names:
            clip_path = clips_dir / clip_name
            if clip_path.exists() and clip_path.is_dir():
                yield Clip(clip_path, self)

    def __repr__(self) -> str:
        num_clips = len(self.get_clip_names())
        return f"Session(name='{self.name}', clips={num_clips})"


class Specimen:
    """Represents a single specimen."""

    def __init__(self, specimen_path: Path, dataset: 'D4D'):
        self.path = Path(specimen_path)
        self.dataset = dataset
        self.name = self.path.name

    def get_session_names(self) -> List[str]:
        """Get list of session directory names."""
        return sorted([d.name for d in self.path.iterdir() if d.is_dir()])

    def __iter__(self) -> Iterator[Session]:
        """Iterate over sessions for this specimen."""
        for session_dir in sorted(self.path.iterdir()):
            if session_dir.is_dir():
                yield Session(session_dir, self)

    def __repr__(self) -> str:
        num_sessions = len(self.get_session_names())
        return f"Specimen(name='{self.name}', sessions={num_sessions})"


class D4D:
    """
    Main interface for the D4D dataset.

    D4D - The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes.

    Usage:
        d4d = D4D('/path/to/dataset')
        for specimen in d4d:
            for session in specimen:
                for clip in session:
                    # Work with clip data
                    left_imgs, right_imgs = clip.load_images()
    """

    def __init__(self, dataset_path: str):
        """
        Initialize D4D dataset loader.

        Args:
            dataset_path: Path to the root of the restructured dataset
        """
        self.path = Path(dataset_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    def get_specimen_names(self) -> List[str]:
        """Get list of specimen directory names."""
        specimens = list(self.path.glob('specimen_*'))
        return sorted([d.name for d in specimens if d.is_dir()])

    def __iter__(self) -> Iterator[Specimen]:
        """Iterate over specimens in the dataset."""
        specimen_dirs = list(self.path.glob('specimen_*'))

        for specimen_dir in sorted(specimen_dirs):
            if specimen_dir.is_dir():
                yield Specimen(specimen_dir, self)

    def __repr__(self) -> str:
        num_specimens = len(self.get_specimen_names())
        return f"D4D(path='{self.path}', specimens={num_specimens})"