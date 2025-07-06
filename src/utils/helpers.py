"""Helper utilities for the DSA Video Summarizer."""

import os
import re
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_file_hash(file_path: str) -> str:
    """Generate MD5 hash for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def parse_timestamp(timestamp: str) -> float:
    """Parse timestamp string to seconds."""
    parts = timestamp.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        return int(parts[0])

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'youtube\.com/watch\?v=([^&]+)',
        r'youtu\.be/([^?]+)',
        r'youtube\.com/embed/([^?]+)',
        r'youtube\.com/v/([^?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters."""
    # Remove invalid characters for filenames
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def detect_programming_language(code: str) -> str:
    """Detect programming language from code snippet."""
    # Simple heuristic-based detection
    if 'def ' in code and 'import ' in code:
        return 'python'
    elif 'class ' in code and 'public ' in code:
        return 'java'
    elif '#include' in code and 'int main' in code:
        return 'cpp'
    elif 'function' in code and 'var ' in code:
        return 'javascript'
    elif 'func ' in code and 'package ' in code:
        return 'go'
    else:
        return 'text'

def extract_dsa_topics(text: str, dsa_keywords: List[str]) -> List[str]:
    """Extract DSA topics mentioned in text."""
    text_lower = text.lower()
    found_topics = []
    
    for topic in dsa_keywords:
        if topic.lower() in text_lower:
            found_topics.append(topic)
    
    return list(set(found_topics))  # Remove duplicates

def chunk_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at word boundary
        while end > start and text[end] != ' ':
            end -= 1
        
        if end == start:  # No space found, force break
            end = start + max_length
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def sanitize_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def create_timestamp_link(video_id: str, timestamp: float) -> str:
    """Create a YouTube timestamp link."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"

def get_video_duration_from_path(video_path: str) -> Optional[float]:
    """Get video duration from file path using moviepy."""
    try:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(video_path) as clip:
            return clip.duration
    except Exception as e:
        logging.error(f"Error getting video duration: {e}")
        return None

def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_filename_from_url(url: str) -> str:
    """Generate safe filename from URL."""
    video_id = extract_youtube_id(url)
    if video_id:
        return f"youtube_{video_id}"
    else:
        # For uploaded files, use hash of URL
        return f"video_{hashlib.md5(url.encode()).hexdigest()[:8]}"

logger = setup_logging()
