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
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def create_animation(input_dir, output_path, duration=100, loop=0, resize=1.0, pad_color='#000000', fps=30):
    """
    Create an animated file (WebP/MP4/MKV) from PNG frames
    
    Args:
        input_dir (Path): Directory containing the PNG frames
        output_path (Path): Path where the output file will be saved
        duration (int): Duration for each frame in milliseconds (used for WebP)
        loop (int): Number of times to loop animation (0 = infinite, only for WebP)
        resize (float): Resize ratio for input images (1.0 = original size, 0.5 = half size, 2.0 = double size)
        pad_color (str): Hex color for padding (e.g., '#000000' for black)
        fps (int): Frames per second for video output (default 30)
    """
    # Get list of frames and sort them
    frame_files = sorted([f for f in input_dir.iterdir() if f.name.startswith('frame_') and f.suffix == '.png'])
    
    if not frame_files:
        raise ValueError(f"No frame_*.png files found in {input_dir}")
    
    print(f"Found {len(frame_files)} frames")
    if resize != 1.0:
        print(f"Applying resize ratio: {resize}")
    
    # Find the largest dimensions among all frames
    max_width, max_height = _find_max_dimensions(frame_files, resize)
    print(f"Standard dimensions (after resize): {max_width}x{max_height}")
    
    # Get output format
    output_format = output_path.suffix.lower()
    
    if output_format == '.webp':
        _create_webp(frame_files, output_path, duration, loop, resize, max_width, max_height, pad_color)
    elif output_format in ['.mp4', '.mkv']:
        _create_video(frame_files, output_path, duration, resize, max_width, max_height, pad_color, fps)
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: .webp, .mp4, .mkv")


def _find_max_dimensions(frame_files, resize_ratio):
    """Find the largest width and height among all frames after applying resize"""
    max_width = 0
    max_height = 0
    
    for frame_file in frame_files:
        try:
            with Image.open(frame_file) as img:
                width, height = img.size
                if resize_ratio != 1.0:
                    width = int(width * resize_ratio)
                    height = int(height * resize_ratio)
                max_width = max(max_width, width)
                max_height = max(max_height, height)
        except Exception as e:
            print(f"Warning: Could not read dimensions from {frame_file.name}: {str(e)}")
            continue
    
    if max_width == 0 or max_height == 0:
        raise ValueError("Could not determine frame dimensions")
    
    return max_width, max_height


def _hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _resize_image(img, resize_ratio):
    """Resize an image using PIL with high-quality resampling"""
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


def _pad_image(img, target_width, target_height, pad_color):
    """Pad an image to target dimensions with specified color (centered)"""
    width, height = img.size
    
    if width == target_width and height == target_height:
        return img
    
    # Create new image with padding color
    rgb_color = _hex_to_rgb(pad_color)
    padded = Image.new('RGB', (target_width, target_height), rgb_color)
    
    # Calculate position to center the original image
    x_offset = (target_width - width) // 2
    y_offset = (target_height - height) // 2
    
    # Paste original image onto padded canvas
    padded.paste(img, (x_offset, y_offset))
    
    return padded


def _resize_cv2_image(img, resize_ratio):
    """Resize an OpenCV image with high-quality resampling"""
    if resize_ratio == 1.0:
        return img
    
    height, width = img.shape[:2]
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)
    
    # Use INTER_LANCZOS4 for high-quality resampling
    if resize_ratio > 1.0:
        # Zooming in
        interpolation = cv2.INTER_LANCZOS4
    else:
        # Zooming out - use INTER_AREA for better downsampling
        interpolation = cv2.INTER_AREA
    
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation)


def _pad_cv2_image(img, target_width, target_height, pad_color):
    """Pad an OpenCV image to target dimensions with specified color (centered)"""
    height, width = img.shape[:2]
    
    if width == target_width and height == target_height:
        return img
    
    # Convert hex color to BGR for OpenCV
    rgb_color = _hex_to_rgb(pad_color)
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # RGB to BGR
    
    # Create padded image
    padded = np.full((target_height, target_width, 3), bgr_color, dtype=np.uint8)
    
    # Calculate position to center the original image
    x_offset = (target_width - width) // 2
    y_offset = (target_height - height) // 2
    
    # Place original image in center
    padded[y_offset:y_offset+height, x_offset:x_offset+width] = img
    
    return padded


def _create_webp(frame_files, output_path, duration, loop, resize, max_width, max_height, pad_color):
    """Create WebP animation"""
    frames = []
    for frame_file in frame_files:
        try:
            with Image.open(frame_file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply resize if needed
                img = _resize_image(img, resize)
                
                # Apply padding if needed
                img = _pad_image(img, max_width, max_height, pad_color)
                
                frames.append(img.copy())
            print(f"Processed {frame_file.name}")
        except Exception as e:
            print(f"Error processing {frame_file.name}: {str(e)}")
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


def _create_video(frame_files, output_path, duration, resize, max_width, max_height, pad_color, fps):
    """Create MP4/MKV video"""
    # Ensure dimensions are even (required by most video codecs)
    width = max_width if max_width % 2 == 0 else max_width + 1
    height = max_height if max_height % 2 == 0 else max_height + 1
    
    # Calculate how many times to repeat each frame to achieve the desired duration
    # duration is in milliseconds, so convert to seconds
    duration_seconds = duration / 1000.0
    frames_to_repeat = max(1, int(round(fps * duration_seconds)))
    
    print(f"Video settings: {fps} fps, each image repeated {frames_to_repeat} times ({duration}ms duration)")
    
    # Initialize video writer with VP9 codec
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                # Apply resize to frame
                frame = _resize_cv2_image(frame, resize)
                
                # Apply padding if needed
                frame = _pad_cv2_image(frame, width, height, pad_color)
                
                # Write the frame multiple times to achieve the desired duration
                for _ in range(frames_to_repeat):
                    out.write(frame)
                
                print(f"Processed {frame_file.name}")
            else:
                print(f"Error reading {frame_file.name}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        out.release()
        
    print(f"Successfully created video: {output_path}")
    print(f"Final video resolution: {width}x{height}")
    print(f"Total frames written: {len(frame_files) * frames_to_repeat}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert PNG frames to animated WebP/MP4/MKV')
    parser.add_argument('--input_dir', required=True, help='Directory containing PNG frames')
    parser.add_argument('--output_path', required=True, help='Path for output file (.webp/.mp4/.mkv)')
    parser.add_argument('--duration', type=int, default=100, help='Duration per frame in milliseconds')
    parser.add_argument('--loop', type=int, default=0, help='Number of animation loops (0 = infinite, WebP only)')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize ratio for input images (1.0=original, 0.5=half, 2.0=double)')
    parser.add_argument('--pad_color', type=str, default='#000000', help='Hex color for padding (e.g., #000000 for black)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for video output (default: 30)')
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)  
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate parameter
    if args.resize <= 0:
        raise ValueError("Resize ratio must be greater than 0")
    
    if not args.pad_color.startswith('#') or len(args.pad_color) != 7:
        raise ValueError("pad_color must be in hex format (e.g., #000000)")
    
    if args.fps <= 0:
        raise ValueError("FPS must be greater than 0")
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    create_animation(
        input_dir,
        output_path,
        duration=args.duration,
        loop=args.loop,
        resize=args.resize,
        pad_color=args.pad_color,
        fps=args.fps
    )


if __name__ == "__main__":
    main()