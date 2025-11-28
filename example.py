#!/usr/bin/env python3
"""
Simple example demonstrating the D4D dataset loader and visualization.
"""

from d4d.loader import D4D
from visualize import visualize_clip_state


def main():
    # Initialize dataset
    dataset_path = "/media/data_ssd/datacollection4d/preprocessed_restructured"
    d4d = D4D(dataset_path)

    # Iterate through hierarchy: specimen → session → clip
    for specimen in d4d:
        print(f"\n{specimen.name}:")
        for session in specimen:
            print(f"  {session.name}:")
            for clip in session:
                print(f"    {clip.name}: {len(clip.left_img_paths)} images, {clip.duration:.1f}s")

    # Visualize first clip start state
    specimen = next(iter(d4d))
    session = next(iter(specimen))
    clip = next(iter(session))

    # Show clip attributes
    print(f"\n{'='*80}")
    print(f"Clip attributes for {clip.name}:")
    print(f"{'='*80}")
    print(f"  name: {clip.name}")
    print(f"  path: {clip.path}")
    print(f"  duration: {clip.duration}")
    print(f"  start_timestamp: {clip.start_timestamp}")
    print(f"  end_timestamp: {clip.end_timestamp}")
    print(f"  left_img_paths: {len(clip.left_img_paths)} images")
    print(f"  right_img_paths: {len(clip.right_img_paths)} images")
    print(f"  stereo_depth_paths: {len(clip.stereo_depth_paths)} depth maps")
    print(f"  pointclouds: {list(clip.pointclouds.keys())}")
    print(f"  endoscope_params: {clip.endoscope_params}")
    print(f"  zivid_params: {clip.zivid_params}")
    print(f"  poses: {list(clip.poses.keys())}")
    print(f"{'='*80}\n")

    if clip.pointclouds.get('start') and clip.left_img_paths and clip.stereo_depth_paths:
        visualize_clip_state(
            clip.pointclouds['start'],
            clip.left_img_paths[0],
            clip.stereo_depth_paths[0],
            clip.endoscope_params,
            clip.poses['start']
        )


if __name__ == "__main__":
    main()