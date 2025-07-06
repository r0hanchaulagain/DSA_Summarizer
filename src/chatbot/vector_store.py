"""Vector store for video content using ChromaDB."""

import logging
import chromadb
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

from utils.config import config
from utils.helpers import logger, ensure_directory_exists

logger = logging.getLogger(__name__)

class VideoVectorStore:
    """Manage vector embeddings for video content."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection_name = "dsa_videos"
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing ChromaDB collection."""
        try:
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "DSA video content embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def store_video_content(
        self, 
        video_id: str,
        transcription_data: Dict,
        content_analysis: Dict,
        summary_data: Optional[Dict] = None
    ) -> bool:
        """
        Store video content in vector database.
        
        Args:
            video_id: Unique video identifier
            transcription_data: Transcription data with timestamps
            content_analysis: Content analysis results
            summary_data: Summary data (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Storing content for video: {video_id}")
            
            documents = []
            metadatas = []
            ids = []
            
            # Store transcription segments
            for segment in transcription_data['segments']:
                # Create document text
                doc_text = segment['text']
                
                # Create metadata
                metadata = {
                    'video_id': video_id,
                    'content_type': 'transcription',
                    'segment_id': segment['id'],
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'start_formatted': segment['start_formatted'],
                    'end_formatted': segment['end_formatted'],
                    'confidence': segment.get('confidence', 0.0)
                }
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(f"{video_id}_segment_{segment['id']}")
            
            # Store topic information
            topics_mentioned = content_analysis.get('topics_mentioned', {})
            for topic, mentions in topics_mentioned.items():
                for mention in mentions:
                    doc_text = f"Topic: {topic}. Context: {mention['text']}"
                    
                    metadata = {
                        'video_id': video_id,
                        'content_type': 'topic',
                        'topic': topic,
                        'segment_id': mention['segment_id'],
                        'start_time': mention['timestamp'],
                        'start_formatted': mention['timestamp_formatted'],
                        'confidence': mention.get('confidence', 0.0)
                    }
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(f"{video_id}_topic_{topic}_{mention['segment_id']}")
            
            # Store algorithm information
            algorithms = content_analysis.get('algorithms_mentioned', [])
            for i, algorithm in enumerate(algorithms):
                doc_text = f"Algorithm: {algorithm['algorithm']}. Context: {algorithm['text']}"
                
                metadata = {
                    'video_id': video_id,
                    'content_type': 'algorithm',
                    'algorithm': algorithm['algorithm'],
                    'segment_id': algorithm['segment_id'],
                    'start_time': algorithm['timestamp'],
                    'start_formatted': algorithm['timestamp_formatted']
                }
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(f"{video_id}_algorithm_{i}")
            
            # Store code snippets
            code_snippets = content_analysis.get('code_snippets', [])
            for i, snippet in enumerate(code_snippets):
                doc_text = f"Code example: {snippet['text']}. Language: {snippet.get('detected_language', 'unknown')}"
                
                metadata = {
                    'video_id': video_id,
                    'content_type': 'code',
                    'language': snippet.get('detected_language', 'unknown'),
                    'segment_id': snippet['segment_id'],
                    'start_time': snippet['timestamp'],
                    'start_formatted': snippet['timestamp_formatted']
                }
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(f"{video_id}_code_{i}")
            
            # Store complexity analysis
            complexity_analysis = content_analysis.get('complexity_analysis', {})
            for complexity_type, mentions in complexity_analysis.items():
                for i, mention in enumerate(mentions):
                    doc_text = f"Complexity discussion ({complexity_type}): {mention['text']}"
                    
                    metadata = {
                        'video_id': video_id,
                        'content_type': 'complexity',
                        'complexity_type': complexity_type,
                        'pattern': mention['pattern_matched'],
                        'segment_id': mention['segment_id'],
                        'start_time': mention['timestamp'],
                        'start_formatted': mention['timestamp_formatted']
                    }
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(f"{video_id}_complexity_{complexity_type}_{i}")
            
            # Store summary sections if available
            if summary_data:
                for section in summary_data.get('detailed_breakdown', []):
                    doc_text = f"Section {section['section_number']}: {section['summary']}"
                    
                    metadata = {
                        'video_id': video_id,
                        'content_type': 'summary',
                        'section_number': section['section_number'],
                        'start_time': section['start_time'],
                        'end_time': section['end_time'],
                        'start_formatted': section['start_formatted'],
                        'end_formatted': section['end_formatted'],
                        'topics_covered': ', '.join(section.get('topics_covered', []))
                    }
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(f"{video_id}_summary_section_{section['section_number']}")
            
            # Add all documents to collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Stored {len(documents)} documents for video: {video_id}")
                return True
            else:
                logger.warning(f"No documents to store for video: {video_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error storing video content: {e}")
            return False
    
    def search_content(
        self, 
        query: str, 
        video_id: Optional[str] = None,
        content_types: Optional[List[str]] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for content in the vector store.
        
        Args:
            query: Search query
            video_id: Specific video ID to search in (optional)
            content_types: List of content types to filter by (optional)
            n_results: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            
            if video_id:
                where_clause['video_id'] = video_id
            
            if content_types:
                where_clause['content_type'] = {"$in": content_types}
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def get_video_content_summary(self, video_id: str) -> Dict[str, Any]:
        """Get summary of stored content for a video."""
        try:
            # Get all content for the video
            results = self.collection.get(
                where={"video_id": video_id}
            )
            
            if not results['documents']:
                return {'video_id': video_id, 'total_documents': 0}
            
            # Count by content type
            content_type_counts = {}
            for metadata in results['metadatas']:
                content_type = metadata.get('content_type', 'unknown')
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            
            # Get time range
            start_times = [
                metadata.get('start_time', 0) 
                for metadata in results['metadatas'] 
                if metadata.get('start_time') is not None
            ]
            
            min_time = min(start_times) if start_times else 0
            max_time = max(start_times) if start_times else 0
            
            return {
                'video_id': video_id,
                'total_documents': len(results['documents']),
                'content_type_counts': content_type_counts,
                'time_range': {
                    'start': min_time,
                    'end': max_time
                },
                'stored_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting video content summary: {e}")
            return {'video_id': video_id, 'error': str(e)}
    
    def search_by_timestamp(
        self, 
        video_id: str, 
        timestamp: float, 
        window: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Search for content around a specific timestamp.
        
        Args:
            video_id: Video ID
            timestamp: Target timestamp in seconds
            window: Time window in seconds (±window around timestamp)
            
        Returns:
            List of content around the timestamp
        """
        try:
            start_time = timestamp - window
            end_time = timestamp + window
            
            results = self.collection.get(
                where={
                    "video_id": video_id,
                    "start_time": {"$gte": start_time, "$lte": end_time}
                }
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'id': results['ids'][i]
                    }
                    formatted_results.append(result)
            
            # Sort by timestamp
            formatted_results.sort(key=lambda x: x['metadata'].get('start_time', 0))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by timestamp: {e}")
            return []
    
    def search_topics(self, video_id: str, topic: str) -> List[Dict[str, Any]]:
        """Search for specific topic mentions in a video."""
        try:
            results = self.collection.get(
                where={
                    "video_id": video_id,
                    "content_type": "topic",
                    "topic": {"$contains": topic}
                }
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'id': results['ids'][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching topics: {e}")
            return []
    
    def search_algorithms(self, video_id: str, algorithm: str) -> List[Dict[str, Any]]:
        """Search for specific algorithm mentions in a video."""
        try:
            results = self.collection.get(
                where={
                    "video_id": video_id,
                    "content_type": "algorithm",
                    "algorithm": {"$contains": algorithm}
                }
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'id': results['ids'][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching algorithms: {e}")
            return []
    
    def delete_video_content(self, video_id: str) -> bool:
        """Delete all content for a specific video."""
        try:
            # Get all IDs for the video
            results = self.collection.get(
                where={"video_id": video_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents for video: {video_id}")
                return True
            else:
                logger.info(f"No content found to delete for video: {video_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error deleting video content: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_count = self.collection.count()
            
            # Get all metadatas to analyze
            all_data = self.collection.get()
            
            # Count unique videos
            video_ids = set()
            content_types = {}
            
            for metadata in all_data['metadatas']:
                video_ids.add(metadata.get('video_id', 'unknown'))
                content_type = metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                'total_documents': collection_count,
                'unique_videos': len(video_ids),
                'content_type_distribution': content_types,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
