"""Query processor for the DSA video chatbot."""

import logging
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import re

from chatbot.vector_store import VideoVectorStore
from utils.config import config
from utils.helpers import logger

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Process user queries and generate responses about video content."""
    
    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.vector_store = VideoVectorStore()
        
        # Query intent patterns
        self.intent_patterns = {
            'timestamp': [
                r'when.*(?:mentioned|discussed|talked about|explained)',
                r'at what time',
                r'timestamp.*(?:for|of)',
                r'what time.*(?:does|is)',
                r'time.*(?:stamp|mark)'
            ],
            'topic_overview': [
                r'what.*topics.*covered',
                r'what.*discussed',
                r'main.*concepts?',
                r'overview.*video',
                r'summary.*topics?'
            ],
            'algorithm_search': [
                r'what.*algorithms?',
                r'which.*algorithms?',
                r'algorithms?.*(?:mentioned|explained|discussed)',
                r'sorting.*algorithms?',
                r'search.*algorithms?'
            ],
            'code_examples': [
                r'code.*examples?',
                r'show.*code',
                r'implementation',
                r'programming.*examples?',
                r'source.*code'
            ],
            'complexity': [
                r'time.*complexity',
                r'space.*complexity',
                r'big.*o',
                r'complexity.*analysis',
                r'efficiency'
            ],
            'explanation': [
                r'explain.*(?:how|what|why)',
                r'how.*(?:does|works?)',
                r'what.*(?:is|does)',
                r'why.*(?:is|does)',
                r'can you explain'
            ]
        }
    
    def process_query(self, query: str, video_id: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User's question
            video_id: ID of the video being queried
            
        Returns:
            Response with answer and supporting information
        """
        try:
            logger.info(f"Processing query for video {video_id}: {query[:50]}...")
            
            # Determine query intent
            intent = self._classify_intent(query)
            
            # Search for relevant content
            relevant_content = self._search_relevant_content(query, video_id, intent)
            
            # Generate response based on intent and content
            response = self._generate_response(query, relevant_content, intent, video_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try asking again.",
                'error': str(e),
                'related_content': [],
                'timestamps': []
            }
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the user's query."""
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default to explanation if no specific intent found
        return 'explanation'
    
    def _search_relevant_content(
        self, 
        query: str, 
        video_id: str, 
        intent: str
    ) -> List[Dict[str, Any]]:
        """Search for relevant content based on query and intent."""
        
        # Adjust search based on intent
        if intent == 'topic_overview':
            # Search for topics and summary content
            results = self.vector_store.search_content(
                query, 
                video_id=video_id,
                content_types=['topic', 'summary', 'transcription'],
                n_results=8
            )
        
        elif intent == 'algorithm_search':
            # Search specifically for algorithms
            results = self.vector_store.search_content(
                query,
                video_id=video_id,
                content_types=['algorithm', 'transcription'],
                n_results=5
            )
        
        elif intent == 'code_examples':
            # Search for code snippets
            results = self.vector_store.search_content(
                query,
                video_id=video_id,
                content_types=['code', 'transcription'],
                n_results=5
            )
        
        elif intent == 'complexity':
            # Search for complexity discussions
            results = self.vector_store.search_content(
                query,
                video_id=video_id,
                content_types=['complexity', 'transcription'],
                n_results=5
            )
        
        elif intent == 'timestamp':
            # Extract potential topics from query and search
            topic_keywords = self._extract_topic_keywords(query)
            if topic_keywords:
                results = []
                for keyword in topic_keywords:
                    topic_results = self.vector_store.search_topics(video_id, keyword)
                    results.extend(topic_results)
                
                # If no topic-specific results, do general search
                if not results:
                    results = self.vector_store.search_content(
                        query,
                        video_id=video_id,
                        n_results=5
                    )
            else:
                results = self.vector_store.search_content(
                    query,
                    video_id=video_id,
                    n_results=5
                )
        
        else:
            # General search for explanation
            results = self.vector_store.search_content(
                query,
                video_id=video_id,
                n_results=6
            )
        
        return results
    
    def _extract_topic_keywords(self, query: str) -> List[str]:
        """Extract potential DSA topic keywords from query."""
        dsa_topics = config.DSA_TOPICS
        found_topics = []
        
        query_lower = query.lower()
        
        for topic in dsa_topics:
            if topic.lower() in query_lower:
                found_topics.append(topic)
        
        return found_topics
    
    def _generate_response(
        self,
        query: str,
        relevant_content: List[Dict[str, Any]],
        intent: str,
        video_id: str
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API with relevant content."""
        
        try:
            # Prepare context from relevant content
            context_parts = []
            timestamps_info = []
            
            for content in relevant_content[:5]:  # Use top 5 results
                metadata = content['metadata']
                
                context_text = f"Content: {content['document']}"
                
                if 'start_formatted' in metadata:
                    context_text += f" (Timestamp: {metadata['start_formatted']})"
                    timestamps_info.append({
                        'timestamp': metadata.get('start_time', 0),
                        'timestamp_formatted': metadata['start_formatted'],
                        'text': content['document']
                    })
                
                context_text += f" (Type: {metadata.get('content_type', 'unknown')})"
                context_parts.append(context_text)
            
            context = "\\n\\n".join(context_parts)
            
            # Create prompt based on intent
            if intent == 'timestamp':
                prompt = f"""
                You are a helpful AI assistant that answers questions about DSA educational videos.
                
                User Question: {query}
                
                Video Content Context:
                {context}
                
                The user is asking about when something was mentioned or discussed. Based on the provided context, 
                answer their question and provide specific timestamps where relevant information can be found.
                
                If you find relevant timestamps, mention them clearly in your response.
                Keep your answer focused on the specific timing and context of what was discussed.
                """
            
            elif intent == 'topic_overview':
                prompt = f"""
                You are a helpful AI assistant that answers questions about DSA educational videos.
                
                User Question: {query}
                
                Video Content Context:
                {context}
                
                The user is asking for an overview of topics covered in the video. Based on the provided context,
                give a comprehensive summary of the main DSA topics, concepts, and algorithms discussed.
                
                Organize your response clearly and mention any specific areas of focus.
                """
            
            elif intent == 'algorithm_search':
                prompt = f"""
                You are a helpful AI assistant that answers questions about DSA educational videos.
                
                User Question: {query}
                
                Video Content Context:
                {context}
                
                The user is asking about algorithms mentioned in the video. Based on the provided context,
                list and briefly explain the algorithms discussed, including any implementation details or
                complexity analysis mentioned.
                """
            
            elif intent == 'code_examples':
                prompt = f"""
                You are a helpful AI assistant that answers questions about DSA educational videos.
                
                User Question: {query}
                
                Video Content Context:
                {context}
                
                The user is asking about code examples or implementations. Based on the provided context,
                describe the code examples shown in the video, including the programming language used,
                what the code demonstrates, and any important implementation details.
                """
            
            elif intent == 'complexity':
                prompt = f"""
                You are a helpful AI assistant that answers questions about DSA educational videos.
                
                User Question: {query}
                
                Video Content Context:
                {context}
                
                The user is asking about complexity analysis. Based on the provided context,
                explain the time and space complexity discussions in the video, including any
                Big O notation mentioned and efficiency considerations.
                """
            
            else:  # explanation
                prompt = f"""
                You are a helpful AI assistant that answers questions about DSA educational videos.
                
                User Question: {query}
                
                Video Content Context:
                {context}
                
                Based on the provided context from the video, answer the user's question clearly and comprehensively.
                If the context doesn't contain enough information to fully answer the question, 
                say so and provide what information you can from the available content.
                
                Focus on being educational and helpful for someone learning DSA concepts.
                """
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3
                )
            )
            
            answer = response.text.strip()
            
            # Format final response
            return {
                'answer': answer,
                'intent': intent,
                'related_content': relevant_content[:3],  # Return top 3 for display
                'timestamps': timestamps_info[:5],  # Return top 5 timestamps
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'answer': f"I found some relevant content about your question, but I'm having trouble generating a complete response. Here's what I found: {relevant_content[0]['document'][:200] if relevant_content else 'No specific content found.'}...",
                'error': str(e),
                'related_content': relevant_content[:3],
                'timestamps': timestamps_info[:5] if 'timestamps_info' in locals() else []
            }
    
    def get_video_topics_overview(self, video_id: str) -> Dict[str, Any]:
        """Get an overview of all topics covered in a video."""
        try:
            # Search for topic content
            topic_results = self.vector_store.search_content(
                "topics concepts algorithms",
                video_id=video_id,
                content_types=['topic', 'algorithm', 'summary'],
                n_results=15
            )
            
            topics = set()
            algorithms = set()
            
            for result in topic_results:
                metadata = result['metadata']
                
                if metadata.get('content_type') == 'topic':
                    topics.add(metadata.get('topic', ''))
                elif metadata.get('content_type') == 'algorithm':
                    algorithms.add(metadata.get('algorithm', ''))
            
            return {
                'topics': list(topics),
                'algorithms': list(algorithms),
                'total_content_pieces': len(topic_results)
            }
            
        except Exception as e:
            logger.error(f"Error getting topics overview: {e}")
            return {'topics': [], 'algorithms': [], 'error': str(e)}
    
    def search_by_timestamp_range(
        self, 
        video_id: str, 
        start_time: float, 
        end_time: float
    ) -> List[Dict[str, Any]]:
        """Search for content within a specific timestamp range."""
        try:
            results = []
            
            # Search around the midpoint of the range
            mid_time = (start_time + end_time) / 2
            window = (end_time - start_time) / 2 + 30  # Add 30 second buffer
            
            content = self.vector_store.search_by_timestamp(
                video_id, mid_time, window
            )
            
            # Filter to exact range
            for item in content:
                item_start = item['metadata'].get('start_time', 0)
                if start_time <= item_start <= end_time:
                    results.append(item)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by timestamp range: {e}")
            return []
    
    def get_suggested_questions(self, video_id: str) -> List[str]:
        """Generate suggested questions based on video content."""
        try:
            # Get overview of topics
            overview = self.get_video_topics_overview(video_id)
            
            suggestions = [
                "What are the main topics covered in this video?",
                "Can you summarize the key concepts explained?"
            ]
            
            # Add topic-specific questions
            for topic in overview['topics'][:3]:
                if topic:
                    suggestions.append(f"When is {topic} discussed in the video?")
                    suggestions.append(f"Explain the {topic} concept from the video")
            
            # Add algorithm-specific questions
            for algorithm in overview['algorithms'][:2]:
                if algorithm:
                    suggestions.append(f"How does the {algorithm} algorithm work?")
                    suggestions.append(f"What is the complexity of {algorithm}?")
            
            # Add general questions
            suggestions.extend([
                "Show me the code examples from the video",
                "What algorithms are mentioned?",
                "Are there any complexity discussions?"
            ])
            
            return suggestions[:8]  # Return max 8 suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggested questions: {e}")
            return [
                "What topics are covered in this video?",
                "What algorithms are explained?",
                "Show me the code examples",
                "What is discussed in the video?"
            ]
