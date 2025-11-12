"""
Animation Creator

This script creates animated files (WebP/MP4/MKV) from PNG frame sequences.
Supports WebP animations with loop control and video formats (MP4/MKV) with custom frame rates.
Now includes image resizing functionality for zoom in/out effects.

Usage:
    python utility/animation_creator.py --input_dir path/to/frames/ --output_path output.webp
    python utility/animation_creator.py --input_dir path/to/frames/ --output_path output.mkv --duration 50
    python utility/animation_creator.py --input_dir path/to/frames/ --output_path output.mkv --duration 200 --loop 5
    python utility/animation_creator.py --input_dir path/to/frames/ --output_path output.webp --resize 0.5  # Zoom out (50%)
    python utility/animation_creator.py --input_dir path/to/frames/ --output_path output.webp --resize 2.0   # Zoom in (200%)
"""
import os
from PIL import Image
from pathlib import Path
import argparse
import cv2
import numpy as np

def create_animation(input_dir, output_path, duration=100, loop=0, resize=1.0):
    """
    Create an animated file (WebP/MP4/MKV) from PNG frames
    
    Args:
        input_dir (str): Directory containing the PNG frames
        output_path (str): Path where the output file will be saved
        duration (int): Duration for each frame in milliseconds
        loop (int): Number of times to loop animation (0 = infinite, only for WebP)
        resize (float): Resize ratio for input images (1.0 = original size, 0.5 = half size, 2.0 = double size)
    """
    # Get list of frames and sort them
    frames = []
    frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frame_files:
        raise ValueError(f"No frame_*.png files found in {input_dir}")
    
    print(f"Found {len(frame_files)} frames")
    if resize != 1.0:
        print(f"Applying resize ratio: {resize}")
    
    # Get output format
    output_format = output_path.suffix.lower()
    
    if output_format == '.webp':
        _create_webp(input_dir, frame_files, output_path, duration, loop, resize)
    elif output_format in ['.mp4', '.mkv']:
        _create_video(input_dir, frame_files, output_path, duration, resize)
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: .webp, .mp4, .mkv")


def _resize_image(img, resize_ratio):
    """
    Resize an image using PIL with high-quality resampling
    
    Args:
        img: PIL Image object
        resize_ratio (float): Resize ratio
    
    Returns:
        PIL Image object: Resized image
    """
    if resize_ratio == 1.0:
        return img
    
    original_size = img.size
    new_size = (int(original_size[0] * resize_ratio), int(original_size[1] * resize_ratio))
    
    # Use LANCZOS for high-quality resampling
    if resize_ratio > 1.0:
        # Zooming in - use LANCZOS for upsampling
        return img.resize(new_size, Image.Resampling.LANCZOS)
    else:
        # Zooming out - use LANCZOS for downsampling with antialiasing
        return img.resize(new_size, Image.Resampling.LANCZOS)


def _resize_cv2_image(img, resize_ratio):
    """
    Resize an OpenCV image with high-quality resampling
    
    Args:
        img: OpenCV image array
        resize_ratio (float): Resize ratio
    
    Returns:
        OpenCV image array: Resized image
    """
    if resize_ratio == 1.0:
        return img
    
    height, width = img.shape[:2]
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    
    # Ensure dimensions are even (required by most video codecs)
    new_width = new_width if new_width % 2 == 0 else new_width + 1
    new_height = new_height if new_height % 2 == 0 else new_height + 1
    
    # Use INTER_LANCZOS4 for high-quality resampling
    if resize_ratio > 1.0:
        # Zooming in
        interpolation = cv2.INTER_LANCZOS4
    else:
        # Zooming out - use INTER_AREA for better downsampling
        interpolation = cv2.INTER_AREA
    
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation)


def _create_webp(input_dir, frame_files, output_path, duration, loop, resize):
    """Create WebP animation"""
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        try:
            with Image.open(frame_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply resize if needed
                img = _resize_image(img, resize)
                frames.append(img.copy())
            print(f"Processed {frame_file}")
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")
            continue
    
    if not frames:
        raise ValueError("No frames were successfully loaded")
    
    try:
        frames[0].save(
            output_path,
            format='WebP',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=loop,
            optimize=True,
            quality=90
        )
        print(f"Successfully created animated WebP: {output_path}")
        if resize != 1.0:
            print(f"Final image size: {frames[0].size} (resize ratio: {resize})")
    except Exception as e:
        print(f"Error saving WebP: {str(e)}")


def _create_video(input_dir, frame_files, output_path, duration, resize):
    """Create MP4/MKV video"""
    if not frame_files:
        return
    
    # Read first frame to get dimensions and apply resize
    first_frame = cv2.imread(os.path.join(input_dir, frame_files[0]))
    first_frame = _resize_cv2_image(first_frame, resize)
    height, width = first_frame.shape[:2]
    
    # Calculate FPS based on duration (converting from milliseconds to seconds)
    fps = 1000 / duration
    
    # Initialize video writer with VP9 codec
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for frame_file in frame_files:
            frame_path = os.path.join(input_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Apply resize to frame
                frame = _resize_cv2_image(frame, resize)
                out.write(frame)
                print(f"Processed {frame_file}")
            else:
                print(f"Error reading {frame_file}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        out.release()
        
    print(f"Successfully created video: {output_path}")
    if resize != 1.0:
        print(f"Final video resolution: {width}x{height} (resize ratio: {resize})")


def main():
    parser = argparse.ArgumentParser(description='Convert PNG frames to animated WebP/MP4/MKV')
    parser.add_argument('--input_dir', required=True, help='Directory containing PNG frames')
    parser.add_argument('--output_path', required=True, help='Path for output file (.webp/.mp4/.mkv)')
    parser.add_argument('--duration', type=int, default=100, help='Duration per frame in milliseconds')
    parser.add_argument('--loop', type=int, default=0, help='Number of animation loops (0 = infinite, WebP only)')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize ratio for input images (1.0=original, 0.5=half, 2.0=double)')
    
    args = parser.parse_args()
    
    # Validate resize parameter
    if args.resize <= 0:
        raise ValueError("Resize ratio must be greater than 0")
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_animation(
        str(input_dir),
        output_path,
        duration=args.duration,
        loop=args.loop,
        resize=args.resize
    )


if __name__ == "__main__":
    main()