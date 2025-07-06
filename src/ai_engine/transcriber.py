"""Audio transcription using OpenAI Whisper."""

import os
import logging
import whisper
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from utils.config import config
from utils.helpers import logger, format_timestamp, ensure_directory_exists

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """Transcribe audio files using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize transcriber with specified model size.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, video_id: str) -> Dict[str, Any]:
        """
        Transcribe audio file and return structured results.
        
        Args:
            audio_path: Path to audio file
            video_id: Unique identifier for the video
            
        Returns:
            Dict containing transcription results with timestamps
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                task="transcribe",
                language="en",
                word_timestamps=True,
                verbose=False
            )
            
            # Process the transcription results
            processed_result = self._process_transcription(result, video_id)
            
            # Save transcription to file
            self._save_transcription(processed_result, video_id)
            
            logger.info(f"Transcription completed for video: {video_id}")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def _process_transcription(self, result: Dict, video_id: str) -> Dict[str, Any]:
        """
        Process raw Whisper transcription results.
        
        Args:
            result: Raw Whisper transcription result
            video_id: Video identifier
            
        Returns:
            Processed transcription data
        """
        # Extract segments with timestamps
        segments = []
        
        for segment in result["segments"]:
            segment_data = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "start_formatted": format_timestamp(segment["start"]),
                "end_formatted": format_timestamp(segment["end"]),
                "text": segment["text"].strip(),
                "confidence": segment.get("avg_logprob", 0.0)
            }
            
            # Extract word-level timestamps if available
            if "words" in segment:
                words = []
                for word in segment["words"]:
                    word_data = {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 0.0)
                    }
                    words.append(word_data)
                segment_data["words"] = words
            
            segments.append(segment_data)
        
        # Create full text
        full_text = " ".join([segment["text"] for segment in segments])
        
        # Calculate statistics
        total_duration = segments[-1]["end"] if segments else 0
        word_count = len(full_text.split())
        
        processed_result = {
            "video_id": video_id,
            "language": result.get("language", "en"),
            "full_text": full_text,
            "segments": segments,
            "total_duration": total_duration,
            "word_count": word_count,
            "transcribed_at": datetime.now().isoformat(),
            "model_used": self.model_size
        }
        
        return processed_result
    
    def _save_transcription(self, transcription_data: Dict, video_id: str) -> str:
        """
        Save transcription data to JSON file.
        
        Args:
            transcription_data: Processed transcription data
            video_id: Video identifier
            
        Returns:
            Path to saved transcription file
        """
        try:
            transcription_dir = os.path.join(config.SUMMARIES_DIR, "transcriptions")
            os.makedirs(transcription_dir, exist_ok=True)
            
            transcription_file = os.path.join(transcription_dir, f"{video_id}_transcription.json")
            
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Transcription saved to: {transcription_file}")
            return transcription_file
            
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")
            raise
    
    def load_transcription(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Load existing transcription from file.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Transcription data if exists, None otherwise
        """
        try:
            transcription_dir = os.path.join(config.SUMMARIES_DIR, "transcriptions")
            transcription_file = os.path.join(transcription_dir, f"{video_id}_transcription.json")
            
            if os.path.exists(transcription_file):
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading transcription: {e}")
            return None
    
    def search_in_transcription(self, transcription_data: Dict, query: str) -> List[Dict[str, Any]]:
        """
        Search for specific text in transcription.
        
        Args:
            transcription_data: Transcription data
            query: Search query
            
        Returns:
            List of matching segments with timestamps
        """
        try:
            query_lower = query.lower()
            matches = []
            
            for segment in transcription_data["segments"]:
                if query_lower in segment["text"].lower():
                    matches.append({
                        "segment_id": segment["id"],
                        "text": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "start_formatted": segment["start_formatted"],
                        "end_formatted": segment["end_formatted"],
                        "confidence": segment.get("confidence", 0.0)
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching transcription: {e}")
            return []
    
    def get_transcription_summary(self, transcription_data: Dict) -> Dict[str, Any]:
        """
        Get summary statistics for transcription.
        
        Args:
            transcription_data: Transcription data
            
        Returns:
            Summary statistics
        """
        try:
            segments = transcription_data["segments"]
            
            # Calculate average segment length
            segment_lengths = [seg["end"] - seg["start"] for seg in segments]
            avg_segment_length = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0
            
            # Calculate speaking rate (words per minute)
            total_words = transcription_data["word_count"]
            total_duration_minutes = transcription_data["total_duration"] / 60
            words_per_minute = total_words / total_duration_minutes if total_duration_minutes > 0 else 0
            
            # Find longest pause between segments
            longest_pause = 0
            for i in range(1, len(segments)):
                pause = segments[i]["start"] - segments[i-1]["end"]
                longest_pause = max(longest_pause, pause)
            
            return {
                "total_segments": len(segments),
                "total_duration": transcription_data["total_duration"],
                "total_words": total_words,
                "average_segment_length": avg_segment_length,
                "words_per_minute": words_per_minute,
                "longest_pause": longest_pause,
                "language": transcription_data["language"],
                "model_used": transcription_data["model_used"]
            }
            
        except Exception as e:
            logger.error(f"Error getting transcription summary: {e}")
            return {}
    
    def extract_timestamps_for_topics(self, transcription_data: Dict, topics: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract timestamps where specific topics are mentioned.
        
        Args:
            transcription_data: Transcription data
            topics: List of topics to search for
            
        Returns:
            Dict mapping topics to their timestamp occurrences
        """
        try:
            topic_timestamps = {topic: [] for topic in topics}
            
            for segment in transcription_data["segments"]:
                text_lower = segment["text"].lower()
                
                for topic in topics:
                    if topic.lower() in text_lower:
                        topic_timestamps[topic].append({
                            "segment_id": segment["id"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "start_formatted": segment["start_formatted"],
                            "end_formatted": segment["end_formatted"],
                            "text": segment["text"],
                            "confidence": segment.get("confidence", 0.0)
                        })
            
            # Remove topics with no occurrences
            topic_timestamps = {k: v for k, v in topic_timestamps.items() if v}
            
            return topic_timestamps
            
        except Exception as e:
            logger.error(f"Error extracting topic timestamps: {e}")
            return {}
    
    def get_segment_by_timestamp(self, transcription_data: Dict, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Get transcription segment at specific timestamp.
        
        Args:
            transcription_data: Transcription data
            timestamp: Timestamp in seconds
            
        Returns:
            Segment data if found, None otherwise
        """
        try:
            for segment in transcription_data["segments"]:
                if segment["start"] <= timestamp <= segment["end"]:
                    return segment
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting segment by timestamp: {e}")
            return None
