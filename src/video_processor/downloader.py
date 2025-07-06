"""YouTube video downloader using yt-dlp."""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import yt_dlp
from utils.config import config
from utils.helpers import extract_youtube_id, clean_filename, format_file_size

logger = logging.getLogger(__name__)

class VideoDownloader:
    """Download videos from YouTube and other platforms."""
    
    def __init__(self):
        self.download_dir = config.VIDEOS_DIR
        
    def download_youtube_video(self, url: str) -> Dict[str, Any]:
        """
        Download YouTube video and return metadata.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dict containing video metadata and file paths
        """
        try:
            # Extract video ID for filename
            video_id = extract_youtube_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[height<=720]/best',  # Limit quality for processing efficiency
                'outtmpl': os.path.join(self.download_dir, f'{video_id}.%(ext)s'),
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'ignoreerrors': True,
                # Add user agent to avoid bot detection
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                # Configure extractor arguments for YouTube
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],
                        'formats': ['missing_pot']  # Allow formats without PO tokens
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                
                # Check video duration (limit to reasonable length)
                duration = info.get('duration', 0)
                if duration > config.MAX_PROCESSING_TIME_MINUTES * 60:
                    raise ValueError(f"Video too long: {duration/60:.1f} minutes (max: {config.MAX_PROCESSING_TIME_MINUTES} minutes)")
                
                # Download the video
                logger.info(f"Downloading video: {info.get('title', 'Unknown')}")
                ydl.download([url])
                
                # Find the downloaded file
                video_path = self._find_downloaded_file(video_id)
                
                if not video_path:
                    raise FileNotFoundError("Downloaded video file not found")
                
                # Get file size
                file_size = os.path.getsize(video_path)
                
                # Check file size
                if file_size > config.MAX_VIDEO_SIZE_MB * 1024 * 1024:
                    os.remove(video_path)
                    raise ValueError(f"Video file too large: {format_file_size(file_size)}")
                
                metadata = {
                    'video_id': video_id,
                    'title': info.get('title', 'Unknown'),
                    'description': info.get('description', ''),
                    'duration': duration,
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'video_path': video_path,
                    'file_size': file_size,
                    'url': url
                }
                
                logger.info(f"Successfully downloaded: {metadata['title']}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise
    
    def _find_downloaded_file(self, video_id: str) -> Optional[str]:
        """Find the downloaded video file."""
        possible_extensions = ['.mp4', '.webm', '.mkv', '.avi']
        
        for ext in possible_extensions:
            video_path = os.path.join(self.download_dir, f'{video_id}{ext}')
            if os.path.exists(video_path):
                return video_path
        
        return None
    
    def validate_uploaded_video(self, file_path: str) -> Dict[str, Any]:
        """
        Validate uploaded video file and return metadata.
        
        Args:
            file_path: Path to uploaded video file
            
        Returns:
            Dict containing video metadata
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError("Video file not found")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > config.MAX_VIDEO_SIZE_MB * 1024 * 1024:
                raise ValueError(f"Video file too large: {format_file_size(file_size)}")
            
            # Get video duration and basic info
            from moviepy.editor import VideoFileClip
            with VideoFileClip(file_path) as clip:
                duration = clip.duration
                
                if duration > config.MAX_PROCESSING_TIME_MINUTES * 60:
                    raise ValueError(f"Video too long: {duration/60:.1f} minutes (max: {config.MAX_PROCESSING_TIME_MINUTES} minutes)")
            
            # Generate video ID from filename
            video_id = Path(file_path).stem
            
            metadata = {
                'video_id': video_id,
                'title': Path(file_path).stem,
                'description': '',
                'duration': duration,
                'uploader': 'Unknown',
                'upload_date': '',
                'view_count': 0,
                'like_count': 0,
                'video_path': file_path,
                'file_size': file_size,
                'url': file_path
            }
            
            logger.info(f"Validated uploaded video: {metadata['title']}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error validating uploaded video: {e}")
            raise
    
    def cleanup_video(self, video_path: str) -> None:
        """Clean up downloaded video file."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up video file: {video_path}")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading."""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'video_id': extract_youtube_id(url),
                    'title': info.get('title', 'Unknown'),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'thumbnail': info.get('thumbnail', ''),
                    'view_count': info.get('view_count', 0),
                }
                
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
