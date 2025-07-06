"""Extract audio and frames from video files."""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip
from PIL import Image
import pytesseract

from utils.config import config
from utils.helpers import logger, format_timestamp, ensure_directory_exists

logger = logging.getLogger(__name__)

class VideoExtractor:
    """Extract audio and frames from video files."""
    
    def __init__(self):
        self.temp_dir = config.TEMP_DIR
        ensure_directory_exists(self.temp_dir)
    
    def extract_audio(self, video_path: str, video_id: str) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            
        Returns:
            Path to extracted audio file
        """
        try:
            audio_path = os.path.join(self.temp_dir, f"{video_id}_audio.wav")
            
            logger.info(f"Extracting audio from: {video_path}")
            
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio is None:
                    raise ValueError("No audio track found in video")
                
                # Extract audio as WAV for better compatibility with Whisper
                audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def extract_frames(self, video_path: str, video_id: str, interval: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            interval: Frame extraction interval in seconds
            
        Returns:
            List of frame information dictionaries
        """
        try:
            if interval is None:
                interval = config.FRAME_EXTRACTION_INTERVAL
            
            frames_dir = os.path.join(self.temp_dir, f"{video_id}_frames")
            ensure_directory_exists(frames_dir)
            
            logger.info(f"Extracting frames from: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            frames_info = []
            frame_number = 0
            
            # Extract frames at specified interval
            for timestamp in range(0, int(duration), interval):
                # Set video position to timestamp
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame
                frame_filename = f"frame_{timestamp:04d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Analyze frame for content
                frame_info = self._analyze_frame(frame, frame_path, timestamp)
                frames_info.append(frame_info)
                
                frame_number += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames_info)} frames")
            return frames_info
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def _analyze_frame(self, frame: np.ndarray, frame_path: str, timestamp: int) -> Dict[str, Any]:
        """
        Analyze a single frame for content.
        
        Args:
            frame: OpenCV frame
            frame_path: Path to saved frame
            timestamp: Frame timestamp in seconds
            
        Returns:
            Frame analysis information
        """
        try:
            # Basic frame information
            height, width = frame.shape[:2]
            
            # Convert to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Detect text in frame using OCR
            text_content = ""
            try:
                text_content = pytesseract.image_to_string(pil_image)
                text_content = text_content.strip()
            except Exception as e:
                logger.warning(f"OCR failed for frame at {timestamp}s: {e}")
            
            # Detect if frame contains code or diagrams
            contains_code = self._detect_code_in_frame(text_content, frame)
            contains_diagram = self._detect_diagram_in_frame(frame)
            
            # Calculate frame brightness and contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            return {
                'timestamp': timestamp,
                'timestamp_formatted': format_timestamp(timestamp),
                'frame_path': frame_path,
                'width': width,
                'height': height,
                'text_content': text_content,
                'contains_code': contains_code,
                'contains_diagram': contains_diagram,
                'brightness': float(brightness),
                'contrast': float(contrast),
                'has_text': len(text_content) > 10,  # Significant text content
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame at {timestamp}s: {e}")
            return {
                'timestamp': timestamp,
                'timestamp_formatted': format_timestamp(timestamp),
                'frame_path': frame_path,
                'width': 0,
                'height': 0,
                'text_content': '',
                'contains_code': False,
                'contains_diagram': False,
                'brightness': 0.0,
                'contrast': 0.0,
                'has_text': False,
            }
    
    def _detect_code_in_frame(self, text: str, frame: np.ndarray) -> bool:
        """
        Detect if frame contains code snippets.
        
        Args:
            text: OCR extracted text
            frame: OpenCV frame
            
        Returns:
            True if code is detected
        """
        # Keywords that often appear in code
        code_keywords = [
            'def ', 'class ', 'if ', 'else', 'for ', 'while ', 'import ',
            'return ', 'print(', 'function', 'var ', 'let ', 'const ',
            'public ', 'private ', 'static ', '++', '--', '==', '!=',
            'null', 'true', 'false', 'void', 'int ', 'string ', 'bool ',
            '{', '}', '[', ']', '(', ')', ';', '//', '/*', '*/', '#'
        ]
        
        text_lower = text.lower()
        code_score = sum(1 for keyword in code_keywords if keyword in text_lower)
        
        # Also check for monospace font patterns (common in code)
        # This is a simplified approach - could be enhanced with font detection
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        
        return code_score >= 3 or (indented_lines > 0 and len(lines) > 3)
    
    def _detect_diagram_in_frame(self, frame: np.ndarray) -> bool:
        """
        Detect if frame contains diagrams or flowcharts.
        
        Args:
            frame: OpenCV frame
            
        Returns:
            True if diagram is detected
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count rectangular shapes (common in diagrams)
            rectangles = 0
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a rectangle
                if len(approx) == 4:
                    rectangles += 1
            
            # If we have multiple rectangles, it's likely a diagram
            return rectangles >= 3
            
        except Exception as e:
            logger.warning(f"Error detecting diagram: {e}")
            return False
    
    def extract_key_frames(self, video_path: str, video_id: str, scene_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Extract key frames based on scene changes.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            scene_threshold: Threshold for scene change detection
            
        Returns:
            List of key frame information
        """
        try:
            frames_dir = os.path.join(self.temp_dir, f"{video_id}_keyframes")
            ensure_directory_exists(frames_dir)
            
            logger.info(f"Extracting key frames from: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            key_frames = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check for scene change
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_score = np.mean(diff) / 255.0
                    
                    if diff_score > scene_threshold:
                        # Scene change detected, save as key frame
                        frame_filename = f"keyframe_{int(timestamp):04d}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        
                        # Analyze key frame
                        frame_info = self._analyze_frame(frame, frame_path, int(timestamp))
                        key_frames.append(frame_info)
                
                prev_frame = gray
                frame_count += 1
                
                # Skip frames for efficiency
                for _ in range(int(fps)):  # Skip 1 second worth of frames
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(key_frames)} key frames")
            return key_frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {e}")
            raise
    
    def cleanup_temp_files(self, video_id: str) -> None:
        """Clean up temporary files for a video."""
        try:
            temp_files = [
                os.path.join(self.temp_dir, f"{video_id}_audio.wav"),
                os.path.join(self.temp_dir, f"{video_id}_frames"),
                os.path.join(self.temp_dir, f"{video_id}_keyframes")
            ]
            
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                    elif os.path.isdir(temp_path):
                        import shutil
                        shutil.rmtree(temp_path)
            
            logger.info(f"Cleaned up temporary files for video: {video_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def get_video_thumbnail(self, video_path: str, video_id: str, timestamp: int = 10) -> str:
        """
        Extract a thumbnail from video at specified timestamp.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            timestamp: Timestamp in seconds for thumbnail
            
        Returns:
            Path to thumbnail image
        """
        try:
            thumbnail_path = os.path.join(self.temp_dir, f"{video_id}_thumbnail.jpg")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            # Set position to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            
            ret, frame = cap.read()
            if not ret:
                # If timestamp is beyond video duration, get first frame
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(thumbnail_path, frame)
            
            cap.release()
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error extracting thumbnail: {e}")
            raise
