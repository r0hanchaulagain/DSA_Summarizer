"""Analyze video content for DSA topics and concepts."""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from collections import Counter
from datetime import datetime

from utils.config import config
from utils.helpers import logger, ensure_directory_exists

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Analyze transcription and frame content for DSA concepts."""
    
    def __init__(self):
        self.dsa_topics = config.DSA_TOPICS
        self.programming_keywords = self._load_programming_keywords()
        self.complexity_patterns = self._load_complexity_patterns()
    
    def _load_programming_keywords(self) -> Dict[str, List[str]]:
        """Load programming language specific keywords."""
        return {
            'python': [
                'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while',
                'try', 'except', 'with', 'lambda', 'yield', 'return', 'print',
                'list', 'dict', 'set', 'tuple', 'range', 'len', 'append', 'extend'
            ],
            'java': [
                'class', 'public', 'private', 'static', 'void', 'int', 'String',
                'ArrayList', 'HashMap', 'LinkedList', 'TreeMap', 'HashSet',
                'System.out.println', 'new', 'extends', 'implements', 'interface'
            ],
            'cpp': [
                '#include', 'using namespace', 'int main', 'cout', 'cin', 'vector',
                'map', 'set', 'queue', 'stack', 'priority_queue', 'pair',
                'std::', 'class', 'struct', 'template', 'typename'
            ],
            'javascript': [
                'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while',
                'Array', 'Object', 'Map', 'Set', 'push', 'pop', 'slice',
                'console.log', 'return', 'class', 'extends'
            ]
        }
    
    def _load_complexity_patterns(self) -> Dict[str, List[str]]:
        """Load time/space complexity patterns."""
        return {
            'time_complexity': [
                r'O\(1\)', r'O\(log n\)', r'O\(n\)', r'O\(n log n\)',
                r'O\(n\^2\)', r'O\(n\^3\)', r'O\(2\^n\)', r'O\(n!\)',
                'constant time', 'logarithmic time', 'linear time',
                'linearithmic time', 'quadratic time', 'exponential time'
            ],
            'space_complexity': [
                'space complexity', 'memory usage', 'auxiliary space',
                'in-place', 'extra space', 'O\(1\) space', 'O\(n\) space'
            ]
        }
    
    def analyze_transcription(self, transcription_data: Dict) -> Dict[str, Any]:
        """
        Analyze transcription for DSA concepts and topics.
        
        Args:
            transcription_data: Transcription data from Whisper
            
        Returns:
            Analysis results including topics, concepts, and timestamps
        """
        try:
            logger.info(f"Analyzing transcription for video: {transcription_data['video_id']}")
            
            full_text = transcription_data['full_text']
            segments = transcription_data['segments']
            
            # Extract DSA topics mentioned
            mentioned_topics = self._extract_topics_with_timestamps(segments)
            
            # Detect programming languages used
            programming_languages = self._detect_programming_languages(full_text, segments)
            
            # Extract code snippets
            code_snippets = self._extract_code_snippets(segments)
            
            # Analyze complexity discussions
            complexity_analysis = self._analyze_complexity_mentions(segments)
            
            # Extract algorithm names and patterns
            algorithms = self._extract_algorithm_mentions(segments)
            
            # Identify problem-solving patterns
            problem_patterns = self._identify_problem_patterns(segments)
            
            # Generate topic summary
            topic_summary = self._generate_topic_summary(mentioned_topics, segments)
            
            analysis_result = {
                'video_id': transcription_data['video_id'],
                'analyzed_at': datetime.now().isoformat(),
                'topics_mentioned': mentioned_topics,
                'programming_languages': programming_languages,
                'code_snippets': code_snippets,
                'complexity_analysis': complexity_analysis,
                'algorithms_mentioned': algorithms,
                'problem_patterns': problem_patterns,
                'topic_summary': topic_summary,
                'total_topics_found': len(mentioned_topics),
                'analysis_confidence': self._calculate_analysis_confidence(mentioned_topics, algorithms)
            }
            
            # Save analysis results
            self._save_analysis(analysis_result)
            
            logger.info(f"Content analysis completed for video: {transcription_data['video_id']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing transcription: {e}")
            raise
    
    def _extract_topics_with_timestamps(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract DSA topics with their timestamps."""
        topic_mentions = {}
        
        for segment in segments:
            text_lower = segment['text'].lower()
            
            for topic in self.dsa_topics:
                if topic.lower() in text_lower:
                    if topic not in topic_mentions:
                        topic_mentions[topic] = []
                    
                    topic_mentions[topic].append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'confidence': segment.get('confidence', 0.0)
                    })
        
        return topic_mentions
    
    def _detect_programming_languages(self, full_text: str, segments: List[Dict]) -> Dict[str, Any]:
        """Detect programming languages mentioned or shown."""
        language_scores = {}
        language_segments = {}
        
        for lang, keywords in self.programming_keywords.items():
            score = 0
            lang_segments = []
            
            for segment in segments:
                text_lower = segment['text'].lower()
                segment_score = 0
                
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        segment_score += 1
                
                if segment_score > 0:
                    score += segment_score
                    lang_segments.append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'keyword_matches': segment_score
                    })
            
            if score > 0:
                language_scores[lang] = score
                language_segments[lang] = lang_segments
        
        # Determine primary language
        primary_language = max(language_scores.items(), key=lambda x: x[1])[0] if language_scores else None
        
        return {
            'primary_language': primary_language,
            'language_scores': language_scores,
            'language_segments': language_segments
        }
    
    def _extract_code_snippets(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Extract potential code snippets from transcription."""
        code_snippets = []
        
        # Patterns that often indicate code
        code_indicators = [
            'here\'s the code', 'the code is', 'let me code', 'write the function',
            'the implementation', 'here\'s how we implement', 'the algorithm looks like',
            'the solution is', 'we can write', 'function definition'
        ]
        
        for segment in segments:
            text_lower = segment['text'].lower()
            
            # Check for code indicators
            for indicator in code_indicators:
                if indicator in text_lower:
                    # Try to detect programming language
                    detected_lang = detect_programming_language(segment['text'])
                    
                    code_snippets.append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'detected_language': detected_lang,
                        'indicator_matched': indicator
                    })
                    break
        
        return code_snippets
    
    def _analyze_complexity_mentions(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analyze mentions of time and space complexity."""
        complexity_mentions = {
            'time_complexity': [],
            'space_complexity': []
        }
        
        for segment in segments:
            text = segment['text']
            
            # Check for time complexity patterns
            for pattern in self.complexity_patterns['time_complexity']:
                if re.search(pattern, text, re.IGNORECASE):
                    complexity_mentions['time_complexity'].append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'pattern_matched': pattern
                    })
            
            # Check for space complexity patterns
            for pattern in self.complexity_patterns['space_complexity']:
                if re.search(pattern, text, re.IGNORECASE):
                    complexity_mentions['space_complexity'].append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'pattern_matched': pattern
                    })
        
        return complexity_mentions
    
    def _extract_algorithm_mentions(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions of specific algorithms."""
        algorithms = [
            'bubble sort', 'merge sort', 'quick sort', 'heap sort', 'insertion sort',
            'binary search', 'linear search', 'depth first search', 'breadth first search',
            'dijkstra', 'bellman ford', 'floyd warshall', 'kruskal', 'prim',
            'kadane', 'fibonacci', 'dp', 'dynamic programming', 'greedy',
            'backtracking', 'two pointers', 'sliding window'
        ]
        
        algorithm_mentions = []
        
        for segment in segments:
            text_lower = segment['text'].lower()
            
            for algorithm in algorithms:
                if algorithm in text_lower:
                    algorithm_mentions.append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'algorithm': algorithm
                    })
        
        return algorithm_mentions
    
    def _identify_problem_patterns(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Identify common problem-solving patterns mentioned."""
        patterns = [
            'brute force', 'optimization', 'edge case', 'base case',
            'recursive case', 'iteration', 'memoization', 'tabulation',
            'bottom up', 'top down', 'divide and conquer'
        ]
        
        pattern_mentions = []
        
        for segment in segments:
            text_lower = segment['text'].lower()
            
            for pattern in patterns:
                if pattern in text_lower:
                    pattern_mentions.append({
                        'segment_id': segment['id'],
                        'timestamp': segment['start'],
                        'timestamp_formatted': segment['start_formatted'],
                        'text': segment['text'],
                        'pattern': pattern
                    })
        
        return pattern_mentions
    
    def _generate_topic_summary(self, topics_mentioned: Dict, segments: List[Dict]) -> Dict[str, Any]:
        """Generate a summary of topics covered."""
        if not topics_mentioned:
            return {'total_topics': 0, 'coverage_timeline': []}
        
        # Create timeline of topic coverage
        timeline = []
        for topic, mentions in topics_mentioned.items():
            for mention in mentions:
                timeline.append({
                    'topic': topic,
                    'timestamp': mention['timestamp'],
                    'timestamp_formatted': mention['timestamp_formatted']
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        # Calculate topic distribution
        topic_counts = Counter(item['topic'] for item in timeline)
        
        return {
            'total_topics': len(topics_mentioned),
            'most_discussed_topics': topic_counts.most_common(5),
            'coverage_timeline': timeline,
            'topic_distribution': dict(topic_counts)
        }
    
    def _calculate_analysis_confidence(self, topics_mentioned: Dict, algorithms: List[Dict]) -> float:
        """Calculate confidence score for the analysis."""
        # Base confidence on number of topics and algorithms found
        topic_score = min(len(topics_mentioned) * 0.1, 0.5)  # Max 0.5 for topics
        algorithm_score = min(len(algorithms) * 0.05, 0.3)   # Max 0.3 for algorithms
        
        # Add base confidence
        base_confidence = 0.2
        
        total_confidence = base_confidence + topic_score + algorithm_score
        return min(total_confidence, 1.0)  # Cap at 1.0
    
    def _save_analysis(self, analysis_data: Dict) -> str:
        """Save analysis results to file."""
        try:
            analysis_dir = os.path.join(config.SUMMARIES_DIR, "content_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_file = os.path.join(analysis_dir, f"{analysis_data['video_id']}_analysis.json")
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Content analysis saved to: {analysis_file}")
            return analysis_file
            
        except Exception as e:
            logger.error(f"Error saving content analysis: {e}")
            raise
    
    def load_analysis(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load existing content analysis from file."""
        try:
            analysis_dir = os.path.join(config.SUMMARIES_DIR, "content_analysis")
            analysis_file = os.path.join(analysis_dir, f"{video_id}_analysis.json")
            
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading content analysis: {e}")
            return None
    
    def analyze_frame_content(self, frames_data: List[Dict]) -> Dict[str, Any]:
        """Analyze frames for DSA-related visual content."""
        try:
            code_frames = []
            diagram_frames = []
            text_frames = []
            other_frames = []
            
            for frame in frames_data:
                if frame.get('contains_code'):
                    # Extract topics from OCR text
                    topics = extract_dsa_topics(frame.get('text_content', ''), self.dsa_topics)
                    code_frames.append({
                        'timestamp': frame['timestamp'],
                        'timestamp_formatted': frame['timestamp_formatted'],
                        'frame_path': frame['frame_path'],
                        'text_content': frame.get('text_content', ''),
                        'topics_detected': topics
                    })
                
                if frame.get('contains_diagram'):
                    diagram_frames.append({
                        'timestamp': frame['timestamp'],
                        'timestamp_formatted': frame['timestamp_formatted'],
                        'frame_path': frame['frame_path'],
                        'text_content': frame.get('text_content', '')
                    })
                
                if frame.get('has_text'):
                    topics = extract_dsa_topics(frame.get('text_content', ''), self.dsa_topics)
                    if topics:  # Only include frames with DSA topics
                        text_frames.append({
                            'timestamp': frame['timestamp'],
                            'timestamp_formatted': frame['timestamp_formatted'],
                            'frame_path': frame['frame_path'],
                            'text_content': frame.get('text_content', ''),
                            'topics_detected': topics
                        })
                
                # Collect neutral frames as fallback for visuals
                if not frame.get('contains_code') and not frame.get('contains_diagram') and not frame.get('has_text'):
                    other_frames.append({
                        'timestamp': frame['timestamp'],
                        'timestamp_formatted': frame['timestamp_formatted'],
                        'frame_path': frame['frame_path']
                    })
            
            return {
                'code_frames': code_frames,
                'diagram_frames': diagram_frames,
                'text_frames': text_frames,
                'other_frames': other_frames,
                'total_code_frames': len(code_frames),
                'total_diagram_frames': len(diagram_frames),
                'total_text_frames': len(text_frames),
                'total_other_frames': len(other_frames)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame content: {e}")
            return {}
    
    def get_topic_timeline(self, analysis_data: Dict) -> List[Dict[str, Any]]:
        """Get chronological timeline of all topics discussed."""
        try:
            timeline = []
            
            # Add topics from transcription
            for topic, mentions in analysis_data.get('topics_mentioned', {}).items():
                for mention in mentions:
                    timeline.append({
                        'timestamp': mention['timestamp'],
                        'timestamp_formatted': mention['timestamp_formatted'],
                        'type': 'topic',
                        'content': topic,
                        'text': mention['text'],
                        'source': 'transcription'
                    })
            
            # Add algorithms
            for algorithm in analysis_data.get('algorithms_mentioned', []):
                timeline.append({
                    'timestamp': algorithm['timestamp'],
                    'timestamp_formatted': algorithm['timestamp_formatted'],
                    'type': 'algorithm',
                    'content': algorithm['algorithm'],
                    'text': algorithm['text'],
                    'source': 'transcription'
                })
            
            # Add complexity mentions
            for complexity_type in ['time_complexity', 'space_complexity']:
                for mention in analysis_data.get('complexity_analysis', {}).get(complexity_type, []):
                    timeline.append({
                        'timestamp': mention['timestamp'],
                        'timestamp_formatted': mention['timestamp_formatted'],
                        'type': complexity_type,
                        'content': mention['pattern_matched'],
                        'text': mention['text'],
                        'source': 'transcription'
                    })
            
            # Sort by timestamp
            timeline.sort(key=lambda x: x['timestamp'])
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error generating topic timeline: {e}")
            return []

def extract_dsa_topics(text: str, dsa_topics: List[str]) -> List[str]:
    """Extract DSA topics from text content."""
    found_topics = []
    text_lower = text.lower()
    
    for topic in dsa_topics:
        if topic.lower() in text_lower:
            found_topics.append(topic)
    
    return found_topics

def detect_programming_language(text: str) -> str:
    """Detect programming language from text content."""
    text_lower = text.lower()
    
    # Simple language detection based on keywords
    if 'def ' in text_lower or 'import ' in text_lower or 'print(' in text_lower:
        return 'python'
    elif 'public class' in text_lower or 'System.out' in text_lower:
        return 'java'
    elif '#include' in text_lower or 'cout' in text_lower or 'std::' in text_lower:
        return 'cpp'
    elif 'function ' in text_lower or 'console.log' in text_lower or 'let ' in text_lower:
        return 'javascript'
    else:
        return 'unknown'
