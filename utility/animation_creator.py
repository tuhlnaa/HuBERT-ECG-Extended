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
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class AnimationConfig:
    """Configuration for animation creation"""
    duration: int = 100  # milliseconds per frame
    loop: int = 0  # 0 = infinite
    resize: float = 1.0
    pad_color: str = '#000000'
    fps: int = 30
    webp_quality: int = 90
    video_codec: str = 'VP90'


class ColorUtils:
    """Utility class for color conversions"""
    
    @staticmethod
    def validate_hex(hex_color: str) -> None:
        """Validate hex color format"""
        if not hex_color.startswith('#') or len(hex_color) != 7:
            raise ValueError("pad_color must be in hex format (e.g., #000000)")
        # Validate hex digits
        try:
            int(hex_color[1:], 16)
        except ValueError:
            raise ValueError(f"Invalid hex color: {hex_color}")
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple for OpenCV"""
        rgb = ColorUtils.hex_to_rgb(hex_color)
        return (rgb[2], rgb[1], rgb[0])


class ImageProcessor:
    """Handles image resizing and padding operations"""
    
    @staticmethod
    def resize_pil(img: Image.Image, resize_ratio: float) -> Image.Image:
        """Resize PIL image with high-quality resampling"""
        if resize_ratio == 1.0:
            return img
        
        new_size = (
            int(img.size[0] * resize_ratio),
            int(img.size[1] * resize_ratio)
        )
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def pad_pil(img: Image.Image, target_width: int, target_height: int, 
                pad_color: str) -> Image.Image:
        """Pad PIL image to target dimensions (centered)"""
        width, height = img.size
        
        if width == target_width and height == target_height:
            return img
        
        rgb_color = ColorUtils.hex_to_rgb(pad_color)
        padded = Image.new('RGB', (target_width, target_height), rgb_color)
        
        x_offset = (target_width - width) // 2
        y_offset = (target_height - height) // 2
        padded.paste(img, (x_offset, y_offset))
        
        return padded
    
    @staticmethod
    def resize_cv2(img: np.ndarray, resize_ratio: float) -> np.ndarray:
        """Resize OpenCV image with high-quality resampling"""
        if resize_ratio == 1.0:
            return img
        
        height, width = img.shape[:2]
        new_size = (int(width * resize_ratio), int(height * resize_ratio))
        
        # Use INTER_AREA for downsampling, LANCZOS4 for upsampling
        interpolation = cv2.INTER_AREA if resize_ratio < 1.0 else cv2.INTER_LANCZOS4
        return cv2.resize(img, new_size, interpolation=interpolation)
    
    @staticmethod
    def pad_cv2(img: np.ndarray, target_width: int, target_height: int,
                pad_color: str) -> np.ndarray:
        """Pad OpenCV image to target dimensions (centered)"""
        height, width = img.shape[:2]
        
        if width == target_width and height == target_height:
            return img
        
        bgr_color = ColorUtils.hex_to_bgr(pad_color)
        padded = np.full((target_height, target_width, 3), bgr_color, dtype=np.uint8)
        
        x_offset = (target_width - width) // 2
        y_offset = (target_height - height) // 2
        padded[y_offset:y_offset+height, x_offset:x_offset+width] = img
        
        return padded


class FrameLoader:
    """Handles loading and dimension detection for frames"""
    
    @staticmethod
    def get_frame_files(input_dir: Path) -> List[Path]:
        """Get sorted list of frame files"""
        frame_files = sorted([
            f for f in input_dir.iterdir() 
            if f.name.startswith('frame_') and f.suffix == '.png'
        ])
        
        if not frame_files:
            raise ValueError(f"No frame_*.png files found in {input_dir}")
        
        return frame_files
    
    @staticmethod
    def find_max_dimensions(frame_files: List[Path], resize_ratio: float) -> Tuple[int, int]:
        """Find maximum dimensions among all frames after resize"""
        max_width = 0
        max_height = 0
        
        for frame_file in frame_files:
            try:
                with Image.open(frame_file) as img:
                    width = int(img.size[0] * resize_ratio)
                    height = int(img.size[1] * resize_ratio)
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
            except Exception as e:
                print(f"Warning: Could not read {frame_file.name}: {e}")
                continue
        
        if max_width == 0 or max_height == 0:
            raise ValueError("Could not determine frame dimensions")
        
        return max_width, max_height


class WebPCreator:
    """Creates WebP animations"""
    
    @staticmethod
    def create(frame_files: List[Path], output_path: Path, config: AnimationConfig,
               max_width: int, max_height: int) -> None:
        """Create WebP animation from frames"""
        frames = []
        processor = ImageProcessor()
        
        for frame_file in frame_files:
            try:
                with Image.open(frame_file) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = processor.resize_pil(img, config.resize)
                    img = processor.pad_pil(img, max_width, max_height, config.pad_color)
                    frames.append(img.copy())
                
                print(f"Processed {frame_file.name}")
            except Exception as e:
                print(f"Error processing {frame_file.name}: {e}")
                continue
        
        if not frames:
            raise ValueError("No frames were successfully loaded")
        
        frames[0].save(
            output_path,
            format='WebP',
            append_images=frames[1:],
            save_all=True,
            duration=config.duration,
            loop=config.loop,
            optimize=True,
            quality=config.webp_quality
        )
        
        print(f"Successfully created animated WebP: {output_path}")
        if config.resize != 1.0:
            print(f"Final size: {frames[0].size} (resize: {config.resize}x)")


class VideoCreator:
    """Creates MP4/MKV videos"""
    
    @staticmethod
    def create(frame_files: List[Path], output_path: Path, config: AnimationConfig,
               max_width: int, max_height: int) -> None:
        """Create video from frames"""
        # Ensure even dimensions for video codecs
        width = max_width + (max_width % 2)
        height = max_height + (max_height % 2)
        
        # Calculate frame repetition for desired duration
        duration_seconds = config.duration / 1000.0
        frames_to_repeat = max(1, int(round(config.fps * duration_seconds)))
        
        print(f"Video: {config.fps} fps, {frames_to_repeat} repeats per image ({config.duration}ms)")
        
        fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
        out = cv2.VideoWriter(str(output_path), fourcc, config.fps, (width, height))
        
        processor = ImageProcessor()
        frame_count = 0
        
        try:
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    print(f"Error reading {frame_file.name}")
                    continue
                
                frame = processor.resize_cv2(frame, config.resize)
                frame = processor.pad_cv2(frame, width, height, config.pad_color)
                
                for _ in range(frames_to_repeat):
                    out.write(frame)
                    frame_count += 1
                
                print(f"Processed {frame_file.name}")
        finally:
            out.release()
        
        print(f"Successfully created video: {output_path}")
        print(f"Resolution: {width}x{height}, Total frames: {frame_count}")


def create_animation(input_dir: Path, output_path: Path, config: AnimationConfig) -> None:
    """
    Create animated file (WebP/MP4/MKV) from PNG frames
    
    Args:
        input_dir: Directory containing PNG frames (frame_*.png)
        output_path: Output file path
        config: Animation configuration
    """
    # Load frames
    frame_files = FrameLoader.get_frame_files(input_dir)
    print(f"Found {len(frame_files)} frames")
    
    if config.resize != 1.0:
        print(f"Resize ratio: {config.resize}x")
    
    # Find maximum dimensions
    max_width, max_height = FrameLoader.find_max_dimensions(frame_files, config.resize)
    print(f"Standard dimensions: {max_width}x{max_height}")
    
    # Create output based on format
    output_format = output_path.suffix.lower()
    
    if output_format == '.webp':
        WebPCreator.create(frame_files, output_path, config, max_width, max_height)
    elif output_format in ['.mp4', '.mkv']:
        VideoCreator.create(frame_files, output_path, config, max_width, max_height)
    else:
        raise ValueError(
            f"Unsupported format: {output_format}. Supported: .webp, .mp4, .mkv"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert PNG frames to animated WebP/MP4/MKV')
    parser.add_argument('--input_dir', required=True, help='Directory containing PNG frames')
    parser.add_argument('--output_path', required=True, help='Output file path (.webp/.mp4/.mkv)')
    parser.add_argument('--duration', type=int, default=100, help='Duration per frame in milliseconds (default: 100)')
    parser.add_argument('--loop', type=int, default=0, help='Animation loops (0=infinite, WebP only, default: 0)')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize ratio (1.0=original, 0.5=half, 2.0=double)')
    parser.add_argument('--pad_color', type=str, default='#000000', help='Padding hex color (e.g., #000000 for black)')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.resize <= 0:
        raise ValueError("Resize ratio must be > 0")
    
    if args.fps <= 0:
        raise ValueError("FPS must be > 0")
    
    ColorUtils.validate_hex(args.pad_color)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = AnimationConfig(
        duration=args.duration,
        loop=args.loop,
        resize=args.resize,
        pad_color=args.pad_color,
        fps=args.fps
    )
    
    create_animation(input_dir, output_path, config)


if __name__ == "__main__":
    main()