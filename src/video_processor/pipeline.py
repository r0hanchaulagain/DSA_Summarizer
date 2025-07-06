"""Main video processing pipeline that orchestrates all components."""

import os
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from video_processor.downloader import VideoDownloader
from video_processor.extractor import VideoExtractor
from ai_engine.transcriber import AudioTranscriber
from ai_engine.content_analyzer import ContentAnalyzer
from ai_engine.summarizer import VideoSummarizer
from chatbot.vector_store import VideoVectorStore
from utils.config import config
from utils.helpers import logger, format_timestamp, format_file_size

class VideoProcessingPipeline:
    """Main pipeline for processing DSA videos end-to-end."""
    
    def __init__(self):
        self.downloader = VideoDownloader()
        self.extractor = VideoExtractor()
        self.transcriber = AudioTranscriber()
        self.analyzer = ContentAnalyzer()
        self.summarizer = VideoSummarizer()
        self.vector_store = VideoVectorStore()
        
    def process_video(self, video_input: str, input_type: str = "url") -> Dict[str, Any]:
        """
        Process a video from URL or file path through the complete pipeline.
        
        Args:
            video_input: YouTube URL or local file path
            input_type: "url" for YouTube URL, "file" for local file
            
        Returns:
            Complete processing results
        """
        start_time = datetime.now()
        video_metadata = None
        audio_path = None
        
        try:
            logger.info(f"Starting video processing pipeline for: {video_input}")
            
            # Step 1: Download or validate video
            if input_type == "url":
                logger.info("Step 1: Downloading video from YouTube...")
                video_metadata = self.downloader.download_youtube_video(video_input)
            else:
                logger.info("Step 1: Validating uploaded video...")
                video_metadata = self.downloader.validate_uploaded_video(video_input)
            
            video_id = video_metadata['video_id']
            video_path = video_metadata['video_path']
            
            logger.info(f"Video metadata obtained for: {video_metadata['title']}")
            
            # Step 2: Extract audio from video
            logger.info("Step 2: Extracting audio...")
            audio_path = self.extractor.extract_audio(video_path, video_id)
            
            # Step 3: Extract frames
            logger.info("Step 3: Extracting frames...")
            frames_data = self.extractor.extract_frames(video_path, video_id)
            
            # Step 4: Transcribe audio
            logger.info("Step 4: Transcribing audio...")
            transcription_data = self.transcriber.transcribe_audio(audio_path, video_id)
            
            # Step 5: Analyze content
            logger.info("Step 5: Analyzing content...")
            content_analysis = self.analyzer.analyze_transcription(transcription_data)
            
            # Step 6: Analyze frames
            logger.info("Step 6: Analyzing frames...")
            frames_analysis = self.analyzer.analyze_frame_content(frames_data)
            
            # Step 7: Generate summary
            logger.info("Step 7: Generating comprehensive summary...")
            summary_result = self.summarizer.generate_comprehensive_summary(
                video_metadata, transcription_data, content_analysis, frames_analysis
            )
            
            # Step 8: Store in vector database
            logger.info("Step 8: Storing content in vector database...")
            vector_store_success = self.vector_store.store_video_content(
                video_id, transcription_data, content_analysis, summary_result['summary_data']
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Compile final results
            results = {
                'video_id': video_id,
                'processing_status': 'completed',
                'processing_time_seconds': processing_time,
                'video_metadata': video_metadata,
                'transcription_data': transcription_data,
                'content_analysis': content_analysis,
                'frames_analysis': frames_analysis,
                'summary_result': summary_result,
                'vector_store_success': vector_store_success,
                'processed_at': datetime.now().isoformat(),
                'statistics': {
                    'total_segments': len(transcription_data['segments']),
                    'total_topics': len(content_analysis.get('topics_mentioned', {})),
                    'total_algorithms': len(content_analysis.get('algorithms_mentioned', [])),
                    'total_frames': len(frames_data),
                    'code_frames': frames_analysis.get('total_code_frames', 0),
                    'diagram_frames': frames_analysis.get('total_diagram_frames', 0)
                }
            }
            
            logger.info(f"Video processing completed successfully in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in video processing pipeline: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'video_id': video_metadata['video_id'] if video_metadata else 'unknown',
                'processing_status': 'failed',
                'error': str(e),
                'processing_time_seconds': processing_time,
                'processed_at': datetime.now().isoformat()
            }
            
        finally:
            # Cleanup temporary files
            if video_metadata:
                self._cleanup_temp_files(video_metadata['video_id'], audio_path)
    
    def _cleanup_temp_files(self, video_id: str, audio_path: Optional[str] = None):
        """Clean up temporary files after processing."""
        try:
            logger.info(f"Cleaning up temporary files for video: {video_id}")
            
            # Clean up extractor temp files
            self.extractor.cleanup_temp_files(video_id)
            
            # Remove audio file
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed temporary audio file: {audio_path}")
                
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")
    
    def get_processing_status(self, video_id: str) -> Dict[str, Any]:
        """Get processing status and results for a video."""
        try:
            # Check if transcription exists
            transcription = self.transcriber.load_transcription(video_id)
            
            # Check if content analysis exists
            content_analysis = self.analyzer.load_analysis(video_id)
            
            # Check if summary exists
            summary = self.summarizer.load_summary(video_id)
            
            # Check vector store
            vector_summary = self.vector_store.get_video_content_summary(video_id)
            
            status = {
                'video_id': video_id,
                'transcription_exists': transcription is not None,
                'content_analysis_exists': content_analysis is not None,
                'summary_exists': summary is not None,
                'vector_store_documents': vector_summary.get('total_documents', 0),
                'processing_complete': all([
                    transcription is not None,
                    content_analysis is not None,
                    summary is not None,
                    vector_summary.get('total_documents', 0) > 0
                ])
            }
            
            if status['processing_complete']:
                status['statistics'] = {
                    'total_segments': len(transcription['segments']) if transcription else 0,
                    'total_topics': len(content_analysis.get('topics_mentioned', {})) if content_analysis else 0,
                    'word_count': transcription.get('word_count', 0) if transcription else 0,
                    'duration': transcription.get('total_duration', 0) if transcription else 0
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {
                'video_id': video_id,
                'error': str(e),
                'processing_complete': False
            }
    
    def reprocess_video(self, video_id: str, steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Reprocess specific steps for an existing video.
        
        Args:
            video_id: Video identifier
            steps: List of steps to reprocess (optional, defaults to all)
            
        Returns:
            Reprocessing results
        """
        try:
            if steps is None:
                steps = ['content_analysis', 'summary', 'vector_store']
            
            results = {'video_id': video_id, 'reprocessed_steps': []}
            
            # Load existing data
            transcription = self.transcriber.load_transcription(video_id)
            if not transcription:
                raise ValueError(f"No transcription found for video: {video_id}")
            
            # Reprocess content analysis
            if 'content_analysis' in steps:
                logger.info(f"Reprocessing content analysis for video: {video_id}")
                content_analysis = self.analyzer.analyze_transcription(transcription)
                results['content_analysis'] = content_analysis
                results['reprocessed_steps'].append('content_analysis')
            else:
                content_analysis = self.analyzer.load_analysis(video_id)
            
            # Reprocess summary
            if 'summary' in steps and content_analysis:
                logger.info(f"Reprocessing summary for video: {video_id}")
                # Need video metadata for summary generation
                summary_result = self.summarizer.generate_comprehensive_summary(
                    {'video_id': video_id, 'title': 'Reprocessed Video'}, 
                    transcription, 
                    content_analysis
                )
                results['summary_result'] = summary_result
                results['reprocessed_steps'].append('summary')
            
            # Reprocess vector store
            if 'vector_store' in steps and content_analysis:
                logger.info(f"Reprocessing vector store for video: {video_id}")
                # Delete existing content first
                self.vector_store.delete_video_content(video_id)
                
                # Re-store content
                summary_data = results.get('summary_result', {}).get('summary_data')
                vector_success = self.vector_store.store_video_content(
                    video_id, transcription, content_analysis, summary_data
                )
                results['vector_store_success'] = vector_success
                results['reprocessed_steps'].append('vector_store')
            
            results['status'] = 'completed'
            results['processed_at'] = datetime.now().isoformat()
            
            logger.info(f"Reprocessing completed for video: {video_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error reprocessing video {video_id}: {e}")
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def get_video_info_preview(self, video_url: str) -> Dict[str, Any]:
        """Get video information without downloading."""
        try:
            return self.downloader.get_video_info(video_url)
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'error': str(e)}
    
    def validate_video_requirements(self, video_input: str, input_type: str = "url") -> Dict[str, Any]:
        """Validate if video meets processing requirements."""
        try:
            if input_type == "url":
                info = self.downloader.get_video_info(video_input)
                duration = info.get('duration', 0)
                
                validation = {
                    'valid': True,
                    'duration': duration,
                    'duration_formatted': format_timestamp(duration),
                    'title': info.get('title', 'Unknown'),
                    'warnings': [],
                    'errors': []
                }
                
                # Check duration limits
                max_duration = config.MAX_PROCESSING_TIME_MINUTES * 60
                if duration > max_duration:
                    validation['valid'] = False
                    validation['errors'].append(
                        f"Video too long: {duration/60:.1f} minutes (max: {config.MAX_PROCESSING_TIME_MINUTES} minutes)"
                    )
                
                # Check for warnings
                if duration > 1800:  # 30 minutes
                    validation['warnings'].append("Long video - processing may take significant time")
                
                return validation
                
            else:
                # For uploaded files, check size and format
                if not os.path.exists(video_input):
                    return {'valid': False, 'errors': ['File not found']}
                
                file_size = os.path.getsize(video_input)
                max_size = config.MAX_VIDEO_SIZE_MB * 1024 * 1024
                
                validation = {
                    'valid': file_size <= max_size,
                    'file_size': file_size,
                    'file_size_formatted': format_file_size(file_size),
                    'warnings': [],
                    'errors': []
                }
                
                if file_size > max_size:
                    validation['errors'].append(
                        f"File too large: {format_file_size(file_size)} (max: {config.MAX_VIDEO_SIZE_MB}MB)"
                    )
                
                return validation
                
        except Exception as e:
            logger.error(f"Error validating video requirements: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
