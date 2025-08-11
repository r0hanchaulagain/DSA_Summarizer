"""Generate comprehensive video summaries using local AI."""

import os
import logging
from typing import Dict, List, Any, Optional
import json
import markdown2
from datetime import datetime

from utils.config import config
from utils.helpers import logger, format_timestamp, ensure_directory_exists
from ai_engine.local_llm import local_llm_manager

logger = logging.getLogger(__name__)

class VideoSummarizer:
    """Generate comprehensive summaries of DSA videos using local LLM."""
    
    def __init__(self):
        self.llm_manager = local_llm_manager
        self.gemini_available = self._check_gemini_availability()
    
    def _check_gemini_availability(self) -> bool:
        """Check if Gemini API is available for enhanced summaries."""
        try:
            if config.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=config.GEMINI_API_KEY)
                # Test if we can create a model instance
                model = genai.GenerativeModel('gemini-1.5-flash')
                return True
            return False
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")
            return False
    
    def generate_comprehensive_summary(
        self, 
        video_metadata: Dict,
        transcription_data: Dict,
        content_analysis: Dict,
        frames_analysis: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive video summary combining all analysis.
        
        Args:
            video_metadata: Video metadata
            transcription_data: Transcription results
            content_analysis: Content analysis results
            frames_analysis: Frame analysis results (optional)
            
        Returns:
            Comprehensive summary data
        """
        try:
            logger.info(f"Generating comprehensive summary for video: {video_metadata['video_id']}")
            
            # Generate different sections of the summary
            executive_summary = self._generate_executive_summary(video_metadata, content_analysis)
            
            detailed_breakdown = self._generate_detailed_breakdown(
                transcription_data, content_analysis, frames_analysis
            )
            
            code_examples = self._extract_and_explain_code(content_analysis, transcription_data)
            
            learning_objectives = self._identify_learning_objectives(content_analysis)
            
            next_steps = self._suggest_next_steps(content_analysis)
            
            summary_document = self._compile_summary_document(
                video_metadata,
                executive_summary,
                detailed_breakdown,
                code_examples,
                learning_objectives,
                next_steps,
                content_analysis
            )
            
            # Save summary
            summary_file = self._save_summary(summary_document, video_metadata['video_id'])
            
            # Generate markdown version
            markdown_file = self._generate_markdown_summary(summary_document, video_metadata['video_id'])
            
            result = {
                'video_id': video_metadata['video_id'],
                'generated_at': datetime.now().isoformat(),
                'summary_data': summary_document,
                'summary_file': summary_file,
                'markdown_file': markdown_file,
                'word_count': len(summary_document.get('full_summary', '').split()),
                'sections_count': len(detailed_breakdown)
            }
            
            logger.info(f"Summary generation completed for video: {video_metadata['video_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {e}")
            raise
    
    def enhance_summary_with_gemini(self, video_id: str) -> Dict[str, Any]:
        """
        Enhance an existing local LLM summary using Gemini API.
        
        Args:
            video_id: ID of the video to enhance
            
        Returns:
            Enhanced summary data
        """
        try:
            if not self.gemini_available:
                raise Exception("Gemini API not available. Please check your API key.")
            
            # Load existing summary
            existing_summary = self.load_summary(video_id)
            if not existing_summary:
                raise Exception(f"No existing summary found for video {video_id}")
            
            # Load transcription and content analysis
            from ai_engine.transcriber import AudioTranscriber
            from ai_engine.content_analyzer import ContentAnalyzer
            
            transcriber = AudioTranscriber()
            content_analyzer = ContentAnalyzer()
            
            transcription_data = transcriber.load_transcription(video_id)
            content_analysis = content_analyzer.load_analysis(video_id)
            
            if not transcription_data or not content_analysis:
                raise Exception("Missing transcription or content analysis data")
            
            # Generate enhanced summary using Gemini
            enhanced_summary = self._generate_gemini_enhanced_summary(
                existing_summary, transcription_data, content_analysis
            )
            
            # Save enhanced version
            enhanced_file = self._save_enhanced_summary(enhanced_summary, video_id)
            
            # Generate enhanced markdown
            enhanced_markdown = self._generate_enhanced_markdown_summary(enhanced_summary, video_id)
            
            result = {
                'video_id': video_id,
                'enhanced_at': datetime.now().isoformat(),
                'enhanced_summary': enhanced_summary,
                'enhanced_file': enhanced_file,
                'enhanced_markdown': enhanced_markdown,
                'original_summary': existing_summary,
                'improvement_notes': self._analyze_improvements(existing_summary, enhanced_summary)
            }
            
            logger.info(f"Gemini enhancement completed for video: {video_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing summary with Gemini: {e}")
            raise
    
    def _generate_gemini_enhanced_summary(
        self, 
        existing_summary: Dict, 
        transcription_data: Dict, 
        content_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate enhanced summary using Gemini API."""
        try:
            import google.generativeai as genai
            
            # Prepare context for Gemini
            gemini_context = f"""
            Original Summary: {existing_summary.get('summary_data', {}).get('executive_summary', '')}
            
            Video Content: {transcription_data.get('full_text', '')[:2000]}...
            
            DSA Topics Identified: {list(content_analysis.get('topics_mentioned', {}).keys())}
            Algorithms Mentioned: {[alg['algorithm'] for alg in content_analysis.get('algorithms_mentioned', [])]}
            
            Please enhance this summary with:
            1. More detailed algorithm explanations
            2. Better code examples and explanations
            3. Enhanced learning objectives
            4. More comprehensive topic breakdown
            5. Better next steps and practice problems
            6. Improved executive summary
            """
            
            prompt = """
            As an expert DSA instructor, enhance this video summary significantly.
            Focus on making it more educational, comprehensive, and actionable for students.
            Maintain the same structure but improve every section with deeper insights.
            """
            
            # Generate enhanced content using Gemini
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(f"{gemini_context}\n\n{prompt}")
            
            # Parse and structure the enhanced response
            enhanced_content = response.text.strip()
            
            # Create enhanced summary structure
            enhanced_summary = {
                'executive_summary': self._extract_enhanced_section(enhanced_content, 'executive'),
                'detailed_breakdown': self._extract_enhanced_section(enhanced_content, 'breakdown'),
                'learning_objectives': self._extract_enhanced_section(enhanced_content, 'objectives'),
                'next_steps': self._extract_enhanced_section(enhanced_content, 'next_steps'),
                'enhancement_notes': 'Enhanced using Gemini API for improved quality and depth',
                'original_summary_id': existing_summary.get('summary_data', {}).get('video_id', ''),
                'enhanced_at': datetime.now().isoformat()
            }
            
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Error generating Gemini enhanced summary: {e}")
            raise
    
    def _extract_enhanced_section(self, content: str, section_type: str) -> str:
        """Extract specific sections from Gemini response."""
        # Simple extraction - in practice, you might want more sophisticated parsing
        if section_type == 'executive':
            return content[:500] + "..." if len(content) > 500 else content
        elif section_type == 'breakdown':
            return content[500:1500] + "..." if len(content) > 1500 else content
        elif section_type == 'objectives':
            return content[1500:2000] + "..." if len(content) > 2000 else content
        else:
            return content
    
    def _save_enhanced_summary(self, enhanced_summary: Dict, video_id: str) -> str:
        """Save enhanced summary to file."""
        try:
            filename = f"enhanced_summary_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(config.SUMMARIES_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(enhanced_summary, f, indent=2)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving enhanced summary: {e}")
            raise
    
    def _generate_enhanced_markdown_summary(self, enhanced_summary: Dict, video_id: str) -> str:
        """Generate enhanced markdown summary."""
        try:
            filename = f"enhanced_summary_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            filepath = os.path.join(config.SUMMARIES_DIR, filename)
            
            markdown_content = self._create_enhanced_markdown_content(enhanced_summary)
            
            with open(filepath, 'w') as f:
                f.write(markdown_content)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating enhanced markdown: {e}")
            raise
    
    def _create_enhanced_markdown_content(self, enhanced_summary: Dict) -> str:
        """Create enhanced markdown content."""
        content = f"""# Enhanced Video Summary (Gemini Enhanced)

## 🚀 Executive Summary
{enhanced_summary.get('executive_summary', 'No executive summary available.')}

## 📚 Learning Objectives
{enhanced_summary.get('learning_objectives', 'No learning objectives available.')}

## 🔍 Detailed Breakdown
{enhanced_summary.get('detailed_breakdown', 'No detailed breakdown available.')}

## 🎯 Next Steps
{enhanced_summary.get('next_steps', 'No next steps available.')}

---

*This summary was enhanced using Gemini API for improved quality and depth.*
*Enhanced at: {enhanced_summary.get('enhanced_at', 'Unknown')}*
"""
        return content
    
    def _analyze_improvements(self, original: Dict, enhanced: Dict) -> Dict[str, Any]:
        """Analyze improvements made by Gemini enhancement."""
        try:
            original_text = str(original.get('summary_data', {}).get('full_summary', ''))
            enhanced_text = str(enhanced.get('enhanced_summary', ''))
            
            improvements = {
                'word_count_increase': len(enhanced_text.split()) - len(original_text.split()),
                'enhancement_ratio': len(enhanced_text) / len(original_text) if len(original_text) > 0 else 1.0,
                'sections_enhanced': len(enhanced.keys()),
                'quality_improvements': [
                    'More detailed explanations',
                    'Enhanced code examples',
                    'Better learning objectives',
                    'Comprehensive topic breakdown',
                    'Actionable next steps'
                ]
            }
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error analyzing improvements: {e}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, video_metadata: Dict, content_analysis: Dict) -> str:
        """Generate executive summary of the video."""
        try:
            topics = list(content_analysis.get('topics_mentioned', {}).keys())
            algorithms = [alg['algorithm'] for alg in content_analysis.get('algorithms_mentioned', [])]
            primary_lang = content_analysis.get('programming_languages', {}).get('primary_language', 'Unknown')
            
            prompt = f"""
            Generate a concise executive summary (2-3 sentences) for a DSA educational video with the following information:
            
            Video Title: {video_metadata.get('title', 'Unknown')}
            Duration: {format_timestamp(video_metadata.get('duration', 0))}
            Topics Covered: {', '.join(topics[:5]) if topics else 'General DSA concepts'}
            Algorithms Discussed: {', '.join(algorithms[:3]) if algorithms else 'Various algorithms'}
            Primary Language: {primary_lang}
            
            Focus on what students will learn and the main concepts covered.
            """
            
            return self.llm_manager.generate_response(prompt, "")
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"This video covers DSA concepts including {', '.join(topics[:3]) if topics else 'various topics'}."
    
    def _generate_detailed_breakdown(
        self, 
        transcription_data: Dict, 
        content_analysis: Dict,
        frames_analysis: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Generate detailed breakdown of video sections."""
        try:
            segments = transcription_data['segments']
            timeline = content_analysis.get('topic_summary', {}).get('coverage_timeline', [])
            
            # Group segments into logical sections (every 5-10 minutes)
            section_duration = 300  # 5 minutes
            video_duration = transcription_data.get('total_duration', 0)
            sections = []
            
            current_section_start = 0
            section_number = 1
            
            while current_section_start < video_duration:
                section_end = min(current_section_start + section_duration, video_duration)
                
                # Get segments for this section
                section_segments = [
                    seg for seg in segments 
                    if current_section_start <= seg['start'] < section_end
                ]
                
                if section_segments:
                    # Get topics covered in this section
                    section_topics = [
                        item for item in timeline
                        if current_section_start <= item['timestamp'] < section_end
                    ]
                    
                    section_text = ' '.join([seg['text'] for seg in section_segments])
                    
                    # Generate summary for this section
                    section_summary = self._summarize_section(
                        section_text, section_topics, section_number
                    )
                    
                    sections.append({
                        'section_number': section_number,
                        'start_time': current_section_start,
                        'end_time': section_end,
                        'start_formatted': format_timestamp(current_section_start),
                        'end_formatted': format_timestamp(section_end),
                        'topics_covered': [topic.get('topic', topic.get('content', 'Unknown')) for topic in section_topics],
                        'summary': section_summary,
                        'segment_count': len(section_segments)
                    })
                
                current_section_start = section_end
                section_number += 1
            
            return sections
            
        except Exception as e:
            logger.error(f"Error generating detailed breakdown: {e}")
            return []
    
    def _summarize_section(self, section_text: str, section_topics: List[Dict], section_number: int) -> str:
        """Summarize a specific section of the video."""
        try:
            if not section_text.strip():
                return "No significant content in this section."
            
            topics_str = ', '.join(set([topic.get('topic', topic.get('content', 'Unknown')) for topic in section_topics]))
            
            prompt = f"""
            Summarize this section of a DSA educational video in 2-3 sentences:
            
            Section {section_number} Topics: {topics_str if topics_str else 'General discussion'}
            
            Transcript: {section_text[:1000]}...
            
            Focus on the key concepts explained and any specific algorithms or data structures discussed.
            """
            
            return self.llm_manager.generate_response(prompt, "")
            
        except Exception as e:
            logger.error(f"Error summarizing section: {e}")
            return "Summary generation failed for this section."
    
    def _extract_and_explain_code(self, content_analysis: Dict, transcription_data: Dict) -> List[Dict[str, Any]]:
        """Extract and explain code examples from the video."""
        try:
            code_snippets = content_analysis.get('code_snippets', [])
            enhanced_snippets = []
            
            for snippet in code_snippets:
                # Get the surrounding context
                segment_id = snippet['segment_id']
                
                # Find surrounding segments for context
                all_segments = transcription_data['segments']
                context_segments = []
                
                for i, seg in enumerate(all_segments):
                    if seg['id'] == segment_id:
                        # Get 1 segment before and 2 segments after for context
                        start_idx = max(0, i - 1)
                        end_idx = min(len(all_segments), i + 3)
                        context_segments = all_segments[start_idx:end_idx]
                        break
                
                context_text = ' '.join([seg['text'] for seg in context_segments])
                
                # Generate explanation
                explanation = self._explain_code_snippet(
                    snippet['text'], 
                    context_text, 
                    snippet.get('detected_language', 'unknown')
                )
                
                enhanced_snippets.append({
                    'timestamp': snippet['timestamp'],
                    'timestamp_formatted': snippet['timestamp_formatted'],
                    'original_text': snippet['text'],
                    'detected_language': snippet.get('detected_language', 'unknown'),
                    'explanation': explanation,
                    'context': context_text[:500] + '...' if len(context_text) > 500 else context_text
                })
            
            return enhanced_snippets
            
        except Exception as e:
            logger.error(f"Error extracting and explaining code: {e}")
            return []
    
    def _explain_code_snippet(self, code_text: str, context: str, language: str) -> str:
        """Generate explanation for a code snippet."""
        try:
            prompt = f"""
            Explain this code snippet from a DSA educational video:
            
            Language: {language}
            Context: {context[:300]}...
            Code/Algorithm Description: {code_text}
            
            Provide a clear explanation of:
            1. What this code/algorithm does
            2. Key concepts being demonstrated
            3. Time/space complexity if mentioned
            
            Keep the explanation educational and suitable for DSA students.
            """
            
            return self.llm_manager.generate_response(prompt, "")
            
        except Exception as e:
            logger.error(f"Error explaining code snippet: {e}")
            return "Code explanation not available."
    
    def _identify_learning_objectives(self, content_analysis: Dict) -> List[str]:
        """Identify learning objectives based on content analysis."""
        try:
            topics = list(content_analysis.get('topics_mentioned', {}).keys())
            algorithms = [alg['algorithm'] for alg in content_analysis.get('algorithms_mentioned', [])]
            patterns = [p['pattern'] for p in content_analysis.get('problem_patterns', [])]
            
            objectives = []
            
            # Generate objectives based on topics
            for topic in topics[:5]:  # Limit to top 5 topics
                objectives.append(f"Understand and implement {topic} data structure/algorithm")
            
            # Add algorithm-specific objectives
            for algorithm in algorithms[:3]:  # Limit to top 3 algorithms
                objectives.append(f"Learn the {algorithm} algorithm and its applications")
            
            # Add pattern-based objectives
            unique_patterns = list(set(patterns))[:3]
            for pattern in unique_patterns:
                objectives.append(f"Master {pattern} problem-solving technique")
            
            # If no specific objectives, add general ones
            if not objectives:
                objectives = [
                    "Understand fundamental data structures and algorithms",
                    "Learn problem-solving techniques in DSA",
                    "Improve algorithmic thinking skills"
                ]
            
            return objectives[:6]  # Limit to 6 objectives
            
        except Exception as e:
            logger.error(f"Error identifying learning objectives: {e}")
            return ["Learn fundamental DSA concepts"]
    
    def _suggest_next_steps(self, content_analysis: Dict) -> Dict[str, List[str]]:
        """Suggest next steps for learning."""
        try:
            topics = list(content_analysis.get('topics_mentioned', {}).keys())
            algorithms = [alg['algorithm'] for alg in content_analysis.get('algorithms_mentioned', [])]
            
            next_steps = {
                'practice_problems': [],
                'related_topics': [],
                'advanced_concepts': []
            }
            
            # Suggest practice problems based on topics
            topic_problems = {
                'array': ['Two Sum', 'Maximum Subarray', 'Rotate Array'],
                'linked list': ['Reverse Linked List', 'Merge Two Lists', 'Detect Cycle'],
                'tree': ['Binary Tree Traversal', 'Maximum Depth', 'Path Sum'],
                'graph': ['DFS/BFS Traversal', 'Number of Islands', 'Course Schedule'],
                'dynamic programming': ['Fibonacci', 'Climbing Stairs', 'Coin Change'],
                'sorting': ['Implement Merge Sort', 'Quick Sort Variations', 'Custom Comparators']
            }
            
            for topic in topics[:3]:
                if topic in topic_problems:
                    next_steps['practice_problems'].extend(topic_problems[topic][:2])
            
            # Suggest related topics
            topic_relationships = {
                'array': ['hash table', 'two pointers', 'sliding window'],
                'linked list': ['stack', 'queue', 'tree'],
                'tree': ['graph', 'heap', 'trie'],
                'graph': ['tree', 'dynamic programming', 'greedy'],
                'sorting': ['searching', 'heap', 'divide and conquer']
            }
            
            for topic in topics[:2]:
                if topic in topic_relationships:
                    next_steps['related_topics'].extend(topic_relationships[topic][:2])
            
            # Suggest advanced concepts
            advanced_concepts = [
                'Advanced graph algorithms (Dijkstra, Floyd-Warshall)',
                'Segment trees and Fenwick trees',
                'Advanced dynamic programming patterns',
                'System design with data structures',
                'Competitive programming techniques'
            ]
            
            next_steps['advanced_concepts'] = advanced_concepts[:3]
            
            # Remove duplicates
            for key in next_steps:
                next_steps[key] = list(set(next_steps[key]))
            
            return next_steps
            
        except Exception as e:
            logger.error(f"Error suggesting next steps: {e}")
            return {
                'practice_problems': ['Solve related LeetCode problems'],
                'related_topics': ['Explore connected DSA topics'],
                'advanced_concepts': ['Study advanced algorithms']
            }
    
    def _compile_summary_document(
        self,
        video_metadata: Dict,
        executive_summary: str,
        detailed_breakdown: List[Dict],
        code_examples: List[Dict],
        learning_objectives: List[str],
        next_steps: Dict[str, List[str]],
        content_analysis: Dict
    ) -> Dict[str, Any]:
        """Compile all summary components into a final document."""
        
        # Create timeline
        timeline = content_analysis.get('topic_summary', {}).get('coverage_timeline', [])
        
        # Generate full summary text
        full_summary = self._generate_full_summary_text(
            executive_summary, detailed_breakdown, code_examples
        )
        
        return {
            'video_id': video_metadata['video_id'],
            'title': video_metadata.get('title', 'Unknown'),
            'duration': video_metadata.get('duration', 0),
            'duration_formatted': format_timestamp(video_metadata.get('duration', 0)),
            'uploader': video_metadata.get('uploader', 'Unknown'),
            'generated_at': datetime.now().isoformat(),
            
            'executive_summary': executive_summary,
            'learning_objectives': learning_objectives,
            'detailed_breakdown': detailed_breakdown,
            'code_examples': code_examples,
            'topic_timeline': timeline,
            'next_steps': next_steps,
            'full_summary': full_summary,
            
            'statistics': {
                'total_topics': len(content_analysis.get('topics_mentioned', {})),
                'total_algorithms': len(content_analysis.get('algorithms_mentioned', [])),
                'total_code_snippets': len(code_examples),
                'sections_count': len(detailed_breakdown),
                'primary_language': content_analysis.get('programming_languages', {}).get('primary_language', 'Unknown')
            }
        }
    
    def _generate_full_summary_text(
        self, 
        executive_summary: str, 
        detailed_breakdown: List[Dict], 
        code_examples: List[Dict]
    ) -> str:
        """Generate full summary text for easy reading."""
        
        summary_parts = [executive_summary]
        
        # Add section summaries
        for section in detailed_breakdown:
            section_text = f"\\n\\nSection {section['section_number']} ({section['start_formatted']} - {section['end_formatted']}): {section['summary']}"
            summary_parts.append(section_text)
        
        # Add code examples summary
        if code_examples:
            summary_parts.append("\\n\\nCode Examples:")
            for i, example in enumerate(code_examples[:3], 1):
                summary_parts.append(f"\\n{i}. At {example['timestamp_formatted']}: {example['explanation'][:100]}...")
        
        return ' '.join(summary_parts)
    
    def _save_summary(self, summary_data: Dict, video_id: str) -> str:
        """Save summary data to JSON file."""
        try:
            summaries_dir = config.SUMMARIES_DIR
            os.makedirs(summaries_dir, exist_ok=True)
            
            summary_file = os.path.join(summaries_dir, f"{video_id}_summary.json")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Summary saved to: {summary_file}")
            return summary_file
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            raise
    
    def _generate_markdown_summary(self, summary_data: Dict, video_id: str) -> str:
        """Generate markdown version of the summary."""
        try:
            markdown_content = self._create_markdown_content(summary_data)
            
            summaries_dir = config.SUMMARIES_DIR
            markdown_file = os.path.join(summaries_dir, f"{video_id}_summary.md")
            
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown summary saved to: {markdown_file}")
            return markdown_file
            
        except Exception as e:
            logger.error(f"Error generating markdown summary: {e}")
            raise
    
    def _create_markdown_content(self, summary_data: Dict) -> str:
        """Create markdown content from summary data."""
        
        markdown_lines = [
            f"# {summary_data['title']}",
            "",
            f"**Duration:** {summary_data['duration_formatted']}  ",
            f"**Uploader:** {summary_data['uploader']}  ",
            f"**Generated:** {summary_data['generated_at'][:19]}  ",
            "",
            "## Executive Summary",
            "",
            summary_data['executive_summary'],
            "",
            "## Learning Objectives",
            ""
        ]
        
        # Add learning objectives
        for objective in summary_data['learning_objectives']:
            markdown_lines.append(f"- {objective}")
        
        markdown_lines.extend(["", "## Video Breakdown", ""])
        
        # Add detailed breakdown
        for section in summary_data['detailed_breakdown']:
            markdown_lines.extend([
                f"### Section {section['section_number']} ({section['start_formatted']} - {section['end_formatted']})",
                "",
                section['summary'],
                ""
            ])
            
            if section['topics_covered']:
                markdown_lines.append(f"**Topics:** {', '.join(section['topics_covered'])}")
                markdown_lines.append("")
        
        # Add code examples
        if summary_data['code_examples']:
            markdown_lines.extend(["## Code Examples", ""])
            
            for i, example in enumerate(summary_data['code_examples'], 1):
                markdown_lines.extend([
                    f"### Example {i} - {example['timestamp_formatted']}",
                    "",
                    f"**Language:** {example['detected_language']}",
                    "",
                    example['explanation'],
                    "",
                    "```",
                    example['original_text'],
                    "```",
                    ""
                ])
        
        # Add timeline
        if summary_data['topic_timeline']:
            markdown_lines.extend(["## Topic Timeline", ""])
            
            for item in summary_data['topic_timeline'][:10]:  # Limit to first 10 items
                topic = item.get('topic', item.get('content', 'Unknown topic'))
                item_type = item.get('type', 'topic')
                markdown_lines.append(
                    f"- **{item['timestamp_formatted']}**: {topic} ({item_type})"
                )
            
            markdown_lines.append("")
        
        # Add next steps
        markdown_lines.extend(["## Next Steps", ""])
        
        for category, items in summary_data['next_steps'].items():
            if items:
                category_title = category.replace('_', ' ').title()
                markdown_lines.extend([f"### {category_title}", ""])
                for item in items:
                    markdown_lines.append(f"- {item}")
                markdown_lines.append("")
        
        # Add statistics
        stats = summary_data['statistics']
        markdown_lines.extend([
            "## Summary Statistics",
            "",
            f"- **Topics Covered:** {stats['total_topics']}",
            f"- **Algorithms Mentioned:** {stats['total_algorithms']}",
            f"- **Code Examples:** {stats['total_code_snippets']}",
            f"- **Primary Language:** {stats['primary_language']}",
            ""
        ])
        
        return "\\n".join(markdown_lines)
    
    def load_summary(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load existing summary from file."""
        try:
            summary_file = os.path.join(config.SUMMARIES_DIR, f"{video_id}_summary.json")
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading summary: {e}")
            return None
