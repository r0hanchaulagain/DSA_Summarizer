"""Configuration management for DSA Video Summarizer."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Video Processing
    MAX_VIDEO_SIZE_MB: int = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
    FRAME_EXTRACTION_INTERVAL: int = int(os.getenv("FRAME_EXTRACTION_INTERVAL", "30"))
    MAX_PROCESSING_TIME_MINUTES: int = int(os.getenv("MAX_PROCESSING_TIME_MINUTES", "30"))
    
    # Database
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/embeddings")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/app.db")
    
    # Paths
    VIDEOS_DIR: str = os.getenv("VIDEOS_DIR", "./data/videos")
    SUMMARIES_DIR: str = os.getenv("SUMMARIES_DIR", "./data/summaries")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./data/temp")
    
    # DSA Topics and Keywords
    DSA_TOPICS = [
        "array", "linked list", "stack", "queue", "tree", "binary tree",
        "binary search tree", "heap", "hash table", "graph", "sorting",
        "searching", "dynamic programming", "recursion", "backtracking",
        "greedy", "divide and conquer", "two pointers", "sliding window",
        "breadth first search", "depth first search", "dijkstra", "bellman ford",
        "floyd warshall", "minimum spanning tree", "union find", "trie",
        "segment tree", "binary indexed tree", "avl tree", "red black tree",
        "b tree", "fibonacci heap", "bloom filter", "lru cache"
    ]
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        directories = [
            cls.VIDEOS_DIR,
            cls.SUMMARIES_DIR,
            cls.TEMP_DIR,
            cls.CHROMA_DB_PATH,
            os.path.dirname(cls.SQLITE_DB_PATH)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> bool:
        """Validate that required API keys are present."""
        if not cls.GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not found in environment variables")
            return False
        return True

# Initialize configuration
config = Config()
config.ensure_directories()
