"""Streamlit frontend for DSA Video Summarizer."""

import streamlit as st
import os
import sys
import traceback
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from video_processor.pipeline import VideoProcessingPipeline
from chatbot.query_processor import QueryProcessor
from utils.config import config
from utils.helpers import format_timestamp, format_file_size

# Configure Streamlit page
st.set_page_config(
    page_title="DSA Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize components
@st.cache_resource
def initialize_pipeline():
    """Initialize the video processing pipeline."""
    return VideoProcessingPipeline()

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot query processor."""
    return QueryProcessor()

def main():
    """Main Streamlit application."""
    
    st.title("🎥 DSA Video Summarizer & Chatbot")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox(
            "Select Page",
            ["🏠 Home", "🎬 Process Video", "📄 View Summary", "💬 Chat with Video", "📊 Statistics"]
        )
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # API Key validation - now optional
        if not config.GEMINI_API_KEY:
            st.info("ℹ️ Using local AI models (no API key required)")
            st.success("✅ Local LLM system active")
            st.warning("💡 Add Gemini API key for enhanced summaries")
        else:
            st.success("✅ Gemini API configured for enhanced features")
            st.info("🚀 You can enhance local summaries with Gemini!")
        
        # Current video info
        if st.session_state.current_video_id:
            st.markdown("---")
            st.header("📹 Current Video")
            st.info(f"ID: {st.session_state.current_video_id}")
            
            if st.button("🗑️ Clear Current Video"):
                st.session_state.current_video_id = None
                st.session_state.processing_results = None
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎬 Process Video":
        show_process_video_page()
    elif page == "📄 View Summary":
        show_summary_page()
    elif page == "💬 Chat with Video":
        show_chat_page()
    elif page == "📊 Statistics":
        show_statistics_page()

def show_home_page():
    """Display the home page."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to DSA Video Summarizer!")
        
        st.markdown("""
        This AI-powered tool helps you learn Data Structures and Algorithms by:
        
        ### Key Features
        - **Video Processing**: Automatically analyze YouTube videos or uploaded files
        - **Smart Transcription**: Extract and timestamp all spoken content
        - **Content Analysis**: Identify DSA topics, algorithms, and code snippets
        - **Comprehensive Summaries**: Generate detailed study materials
        - **Interactive Chat**: Ask questions about the video content
        - **Timestamp Search**: Find specific topics with exact timing
        
        ### How It Works
        1. **Input**: Provide a YouTube URL or upload a video file
        2. **Processing**: AI analyzes audio, video frames, and content
        3. **Analysis**: Extracts DSA topics, algorithms, and code examples
        4. **Summary**: Generates comprehensive study materials
        5. **Chat**: Ask questions and get instant answers about the content
        
        ### Perfect For
        - Students learning DSA concepts
        - Interview preparation
        - Review and quick reference
        - Understanding complex algorithms
        - Code implementation examples
        """)
        
    with col2:
        st.header("Quick Start")
        
        # Quick stats
        try:
            pipeline = initialize_pipeline()
            vector_store = pipeline.vector_store
            stats = vector_store.get_collection_stats()
            
            st.metric("Videos Processed", stats.get('unique_videos', 0))
            st.metric("Total Documents", stats.get('total_documents', 0))
            
        except Exception as e:
            st.warning("Could not load statistics")
        
        st.markdown("---")
        
        # Sample video for testing
        st.subheader("Try with Sample Video")
        sample_url = "https://www.youtube.com/watch?v=8hly31xKli0"  # Sample DSA video
        if st.button("Process Sample Video"):
            st.session_state.sample_url = sample_url
            st.info(f"Sample URL loaded: {sample_url}")
        
        st.markdown("---")
        
        # Recent activity
        st.subheader("System Status")
        if config.validate_api_keys():
            st.success("All systems operational")
        else:
            st.error("Configuration issues detected")

def show_process_video_page():
    """Display the video processing page."""
    
    st.header("Process DSA Video")
    
    pipeline = initialize_pipeline()
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["🌐 YouTube URL", "📁 Upload Video File"]
    )
    
    video_input = None
    input_type = None
    
    if input_method == "🌐 YouTube URL":
        # Check for sample URL from home page
        default_url = st.session_state.get('sample_url', '')
        video_input = st.text_input(
            "Enter YouTube URL:",
            value=default_url,
            placeholder="https://www.youtube.com/watch?v=..."
        )
        input_type = "url"
        
        # Preview video info
        if video_input and st.button("Preview Video Info"):
            with st.spinner("Getting video information..."):
                info = pipeline.get_video_info_preview(video_input)
                
                if 'error' in info:
                    st.error(f"Error: {info['error']}")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Title:** {info.get('title', 'Unknown')}")
                        st.info(f"**Duration:** {format_timestamp(info.get('duration', 0))}")
                    with col2:
                        st.info(f"**Uploader:** {info.get('uploader', 'Unknown')}")
                        st.info(f"**Views:** {info.get('view_count', 'Unknown'):,}")
    
    else:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a DSA educational video (max 500MB)"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = os.path.join(config.TEMP_DIR, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            video_input = temp_path
            input_type = "file"
            
            file_size = len(uploaded_file.getvalue())
            st.info(f"File uploaded: {uploaded_file.name} ({format_file_size(file_size)})")
    
    # Validation and processing
    if video_input:
        # Validate requirements
        validation = pipeline.validate_video_requirements(video_input, input_type)
        
        if not validation['valid']:
            st.error("Video does not meet requirements:")
            for error in validation['errors']:
                st.error(f"• {error}")
        else:
            if validation.get('warnings'):
                for warning in validation['warnings']:
                    st.warning(f"{warning}")
            
            st.success("Video meets requirements")
            
            # Processing controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("Ready to process! This may take several minutes depending on video length.")
                
                # Display processing summary here if available
                if st.session_state.processing_results:
                    st.markdown("---")
                    display_processing_summary(st.session_state.processing_results)
            
            with col2:
                if st.button("Start Processing", type="primary"):
                    process_video(pipeline, video_input, input_type)

def process_video(pipeline, video_input, input_type):
    """Process the video through the pipeline."""
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Start processing
            status_text.text("Initializing video processing...")
            progress_bar.progress(10)
            
            # Update progress periodically
            def update_progress(step, message):
                progress_values = {
                    1: 20, 2: 30, 3: 40, 4: 60, 5: 70, 6: 80, 7: 90, 8: 95
                }
                progress_bar.progress(progress_values.get(step, 10))
                status_text.text(message)
            
            # Process video
            update_progress(1, "Downloading/validating video...")
            results = pipeline.process_video(video_input, input_type)
            
            progress_bar.progress(100)
            status_text.text("Processing completed!")
            
            if results['processing_status'] == 'completed':
                st.success(f"Video processed successfully in {results['processing_time_seconds']:.1f} seconds!")
                
                # Store results in session state
                st.session_state.current_video_id = results['video_id']
                st.session_state.processing_results = results
                
                # Summary will be displayed in the main page layout
                st.rerun()
                
            else:
                st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")
            st.error(f"Error during processing: {str(e)}")
            st.error(traceback.format_exc())

def display_processing_summary(results):
    """Display a summary of processing results."""
    
    # Create a card-like container for the processing summary
    with st.container():
        st.markdown("---")
        
        # Header with icon
        st.subheader("🔄 Processing Summary")
        
        # Basic metrics in a compact horizontal layout
        col1, col2, col3, col4 = st.columns(4)
        
        stats = results['statistics']
        
        with col1:
            st.metric("Segments", stats['total_segments'])
        with col2:
            st.metric("Topics", stats['total_topics'])
        with col3:
            st.metric("Algorithms", stats['total_algorithms'])
        with col4:
            st.metric("Frames", stats['total_frames'])
        
        st.markdown("---")
        
        # Content analysis in a compact layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Topics Found:**")
            topics = list(results['content_analysis']['topics_mentioned'].keys())
            if topics:
                unique_topics = list(set(topics))
                for i, topic in enumerate(unique_topics[:5], 1):
                    st.write(f"{i}. {topic}")
                if len(unique_topics) > 5:
                    st.caption(f"... and {len(unique_topics) - 5} more")
            else:
                st.caption("No specific DSA topics detected")
        
        with col2:
            st.write("**Algorithms Mentioned:**")
            algorithms = [alg['algorithm'] for alg in results['content_analysis']['algorithms_mentioned']]
            if algorithms:
                unique_algorithms = list(set(algorithms))
                for i, algorithm in enumerate(unique_algorithms[:5], 1):
                    st.write(f"{i}. {algorithm}")
                if len(unique_algorithms) > 5:
                    st.caption(f"... and {len(unique_algorithms) - 5} more")
            else:
                st.caption("No specific algorithms detected")
        
        st.markdown("---")
        
        # Next steps section
        st.write("**Next Steps:**")
        
        # Action buttons in a compact row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 View Full Summary", use_container_width=True):
                st.session_state.page = "📄 View Summary"
        
        with col2:
            if st.button("💬 Start Chatting", use_container_width=True):
                st.session_state.page = "💬 Chat with Video"
        
        with col3:
            if st.button("📥 Download Results", use_container_width=True):
                download_results(results)

def show_summary_page():
    """Display the video summary page."""
    
    st.header("📄 Video Summary")
    
    if not st.session_state.current_video_id:
        st.warning("⚠️ No video currently loaded. Please process a video first.")
        return
    
    pipeline = initialize_pipeline()
    
    try:
        # Load summary data
        summary_data = pipeline.summarizer.load_summary(st.session_state.current_video_id)
        
        if not summary_data:
            st.error("❌ No summary found for current video")
            return
        
        # Display video info
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"🎬 {summary_data['title']}")
        
        with col2:
            st.metric("⏱️ Duration", summary_data['duration_formatted'])
        
        # Executive summary - removed heading but kept content
        st.markdown("---")
        st.write(summary_data['executive_summary'])
        
        # Learning objectives
        st.markdown("---")
        st.header("Learning Objectives")
        for i, objective in enumerate(summary_data['learning_objectives'], 1):
            st.write(f"{i}. {objective}")
        
        # Video breakdown
        st.markdown("---")
        st.header("Video Breakdown")
        
        for section in summary_data['detailed_breakdown']:
            with st.expander(
                f"Section {section['section_number']} ({section['start_formatted']} - {section['end_formatted']})"
            ):
                st.write(section['summary'])
                if section['topics_covered']:
                    # Remove duplicates from topics
                    unique_topics = list(set(section['topics_covered']))
                    st.write(f"**Topics:** {', '.join(unique_topics)}")
        
        # Code examples
        if summary_data['code_examples']:
            st.markdown("---")
            st.header("Code Examples")
            
            for i, example in enumerate(summary_data['code_examples'], 1):
                with st.expander(f"Example {i} - {example['timestamp_formatted']}"):
                    st.write(f"**Language:** {example['detected_language']}")
                    st.write(example['explanation'])
                    if example['original_text']:
                        st.code(example['original_text'], language=example['detected_language'])
        
        # Gemini Enhancement Button
        if config.GEMINI_API_KEY:
            st.markdown("---")
            st.header("🚀 Enhance Summary with Gemini")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 Click below to enhance this summary using Gemini API for improved quality and depth.")
            
            with col2:
                if st.button("✨ Enhance with Gemini", type="primary", use_container_width=True):
                    with st.spinner("Enhancing summary with Gemini..."):
                        try:
                            enhanced_results = pipeline.summarizer.enhance_summary_with_gemini(
                                st.session_state.current_video_id
                            )
                            
                            # Store enhanced results
                            st.session_state.enhanced_summary = enhanced_results
                            
                            st.success("✅ Summary enhanced successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error enhancing summary: {e}")
        
        # Display enhanced summary if available
        if hasattr(st.session_state, 'enhanced_summary') and st.session_state.enhanced_summary:
            st.markdown("---")
            st.header("✨ Enhanced Summary (Gemini)")
            
            enhanced = st.session_state.enhanced_summary
            
            # Show improvement metrics
            if 'improvement_notes' in enhanced:
                improvements = enhanced['improvement_notes']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Word Increase", improvements.get('word_count_increase', 0))
                with col2:
                    st.metric("📊 Enhancement Ratio", f"{improvements.get('enhancement_ratio', 1.0):.1f}x")
                with col3:
                    st.metric("🔧 Sections Enhanced", improvements.get('sections_enhanced', 0))
            
            # Display enhanced content
            if 'enhanced_summary' in enhanced:
                enhanced_data = enhanced['enhanced_summary']
                
                # Enhanced executive summary
                st.subheader("🚀 Enhanced Executive Summary")
                st.write(enhanced_data.get('executive_summary', 'No enhanced summary available.'))
                
                # Enhanced learning objectives
                if enhanced_data.get('learning_objectives'):
                    st.subheader("📚 Enhanced Learning Objectives")
                    st.write(enhanced_data.get('learning_objectives', ''))
                
                # Enhanced detailed breakdown
                if enhanced_data.get('detailed_breakdown'):
                    st.subheader("🔍 Enhanced Detailed Breakdown")
                    st.write(enhanced_data.get('detailed_breakdown', ''))
                
                # Enhanced next steps
                if enhanced_data.get('next_steps'):
                    st.subheader("🎯 Enhanced Next Steps")
                    st.write(enhanced_data.get('next_steps', ''))
                
                # Download enhanced summary
                if st.button("📥 Download Enhanced Summary", use_container_width=True):
                    download_enhanced_summary(enhanced)
        
        # Topic timeline - grouped by topic instead of duplicating
        if summary_data['topic_timeline']:
            st.markdown("---")
            st.header("Topic Timeline")
            
            # Group topics by name and collect timestamps
            topic_groups = {}
            for item in summary_data['topic_timeline']:
                content = item.get('topic', item.get('content', 'Unknown topic'))
                if content not in topic_groups:
                    topic_groups[content] = []
                topic_groups[content].append(item['timestamp_formatted'])
            
            # Display grouped topics with merged consecutive timestamps
            for topic, timestamps in sorted(topic_groups.items()):
                if len(timestamps) == 1:
                    st.write(f"**{timestamps[0]}**: {topic}")
                else:
                    # Sort timestamps chronologically
                    timestamps.sort()
                    
                    # Merge consecutive timestamps within 2 minutes (120 seconds)
                    merged_ranges = []
                    current_start = timestamps[0]
                    current_end = timestamps[0]
                    
                    for i in range(1, len(timestamps)):
                        # Check if current timestamp is within 2 minutes of the previous
                        prev_time = timestamps[i-1]
                        curr_time = timestamps[i]
                        
                        # Convert timestamps to seconds for comparison
                        def timestamp_to_seconds(ts):
                            parts = ts.split(':')
                            if len(parts) == 3:  # HH:MM:SS
                                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            elif len(parts) == 2:  # MM:SS
                                return int(parts[0]) * 60 + int(parts[1])
                            else:
                                return int(parts[0])
                        
                        prev_seconds = timestamp_to_seconds(prev_time)
                        curr_seconds = timestamp_to_seconds(curr_time)
                        
                        if curr_seconds - prev_seconds <= 120:  # Within 2 minutes
                            current_end = curr_time
                        else:
                            # End current range and start new one
                            if current_start == current_end:
                                merged_ranges.append(current_start)
                            else:
                                merged_ranges.append(f"{current_start} - {current_end}")
                            current_start = curr_time
                            current_end = curr_time
                    
                    # Add the last range
                    if current_start == current_end:
                        merged_ranges.append(current_start)
                    else:
                        merged_ranges.append(f"{current_start} - {current_end}")
                    
                    # Display merged topic timeline
                    if len(merged_ranges) == 1:
                        st.write(f"**{merged_ranges[0]}**: {topic}")
                    else:
                        st.write(f"**{topic}**:")
                        for time_range in merged_ranges:
                            st.write(f"  • {time_range}")
        
        # Next steps
        st.markdown("---")
        st.header("Next Steps")
        
        next_steps = summary_data['next_steps']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Practice Problems")
            for problem in next_steps.get('practice_problems', []):
                st.write(f"• {problem}")
        
        with col2:
            st.subheader("Related Topics")
            for topic in next_steps.get('related_topics', []):
                st.write(f"• {topic}")
        
        with col3:
            st.subheader("Advanced Concepts")
            for concept in next_steps.get('advanced_concepts', []):
                st.write(f"• {concept}")
        
        # Download options
        st.markdown("---")
        st.header("📥 Download Summary")
        
        # Check if markdown file exists
        markdown_file = summary_data.get('markdown_file')
        if markdown_file and os.path.exists(markdown_file):
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            st.download_button(
                label="📄 Download Summary (Markdown)",
                data=markdown_content,
                file_name=f"{st.session_state.current_video_id}_summary.md",
                mime="text/markdown",
                use_container_width=True,
                help="Download a nicely formatted markdown summary of the video"
            )
        else:
            # Generate markdown content on the fly if file doesn't exist
            markdown_content = generate_summary_markdown(summary_data)
            
            st.download_button(
                label="📄 Download Summary (Markdown)",
                data=markdown_content,
                file_name=f"{st.session_state.current_video_id}_summary.md",
                mime="text/markdown",
                use_container_width=True,
                help="Download a nicely formatted markdown summary of the video"
            )
        
        # Optional: Keep JSON download in a collapsible section for developers
        with st.expander("🔧 Developer Options"):
            st.info("Advanced users can download raw data in JSON format")
            st.download_button(
                label="📊 Download Raw Data (JSON)",
                data=str(summary_data),
                file_name=f"{st.session_state.current_video_id}_raw_data.json",
                mime="application/json",
                use_container_width=True,
                help="Raw processing data for technical analysis"
            )
    
    except Exception as e:
        st.error(f"Error loading summary: {str(e)}")

def show_chat_page():
    """Display the chat interface."""
    
    st.header("Chat with Video")
    
    if not st.session_state.current_video_id:
        st.warning("No video currently loaded. Please process a video first.")
        return
    
    try:
        chatbot = initialize_chatbot()
        
        # Chat interface
        st.subheader(f"Chatting about: {st.session_state.current_video_id}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
                
                # Show timestamps if available
                if 'timestamps' in message:
                    with st.expander("Related Timestamps"):
                        for ts in message['timestamps']:
                            st.write(f"• {ts['timestamp_formatted']}: {ts['text'][:100]}...")
        
        # Chat input
        user_input = st.chat_input("Ask a question about the video...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Process query and get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.process_query(
                        user_input,
                        st.session_state.current_video_id
                    )
                
                st.write(response['answer'])
                
                # Show related content if available
                if response.get('related_content'):
                    with st.expander("Related Content"):
                        for content in response['related_content'][:3]:
                            st.write(f"**{content['metadata']['content_type']}**: {content['document'][:200]}...")
                            if 'start_formatted' in content['metadata']:
                                st.write(f"{content['metadata']['start_formatted']}")
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamps': response.get('timestamps', []),
                    'related_content': response.get('related_content', [])
                })
        
        # Suggested questions
        st.markdown("---")
        st.subheader("💡 Suggested Questions")
        
        suggestions = [
            "What topics are covered in this video?",
            "What algorithms are explained?",
            "Show me the code examples",
            "What is the time complexity discussed?",
            "When is [specific topic] mentioned?",
            "Explain the main concept of this video"
        ]
        
        col1, col2 = st.columns(2)
        
        for i, suggestion in enumerate(suggestions):
            with col1 if i % 2 == 0 else col2:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    # Trigger the suggestion as if user typed it
                    st.session_state.suggested_query = suggestion
    
    except Exception as e:
        st.error(f"Error in chat interface: {str(e)}")

def show_statistics_page():
    """Display system statistics."""
    
    st.header("📊 System Statistics")
    
    try:
        pipeline = initialize_pipeline()
        vector_store = pipeline.vector_store
        
        # Collection stats
        stats = vector_store.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📼 Videos Processed", stats.get('unique_videos', 0))
        
        with col2:
            st.metric("📄 Total Documents", stats.get('total_documents', 0))
        
        with col3:
            st.metric("🗄️ Collection Name", stats.get('collection_name', 'N/A'))
        
        # Content type distribution
        if 'content_type_distribution' in stats:
            st.markdown("---")
            st.subheader("📊 Content Type Distribution")
            
            import pandas as pd
            df = pd.DataFrame(
                list(stats['content_type_distribution'].items()),
                columns=['Content Type', 'Count']
            )
            st.bar_chart(df.set_index('Content Type'))
        
        # System status
        st.markdown("---")
        st.subheader("⚙️ System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if config.validate_api_keys():
                st.success("✅ API Keys Configured")
            else:
                st.error("❌ API Keys Missing")
        
        with col2:
            try:
                # Test vector store connection
                vector_store.get_collection_stats()
                st.success("✅ Vector Store Connected")
            except:
                st.error("❌ Vector Store Error")
        
        # Configuration info
        st.markdown("---")
        st.subheader("🔧 Configuration")
        
        config_info = {
            "Max Video Size": f"{config.MAX_VIDEO_SIZE_MB} MB",
            "Max Processing Time": f"{config.MAX_PROCESSING_TIME_MINUTES} minutes",
            "Frame Extraction Interval": f"{config.FRAME_EXTRACTION_INTERVAL} seconds",
            "Debug Mode": config.DEBUG
        }
        
        for key, value in config_info.items():
            st.write(f"**{key}**: {value}")
    
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def download_results(results):
    """Provide download options for processing results."""
    
    st.markdown("---")
    st.subheader("📥 Download Options")
    
    # Generate markdown summary
    markdown_content = generate_markdown_summary(results)
    
    # Markdown download
    st.download_button(
        label="📄 Download Summary (Markdown)",
        data=markdown_content,
        file_name=f"{results['video_id']}_summary.md",
        mime="text/markdown"
    )
    
    # JSON results (keep as backup option)
    import json
    results_json = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="🔧 Download Raw Results (JSON)",
        data=results_json,
        file_name=f"{results['video_id']}_raw_results.json",
        mime="application/json",
        help="Raw processing data for technical use"
    )

def download_enhanced_summary(enhanced_results):
    """Download enhanced summary results."""
    try:
        # Convert enhanced results to JSON string
        import json
        json_str = json.dumps(enhanced_results, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download Enhanced Summary (JSON)",
            data=json_str,
            file_name=f"enhanced_summary_{enhanced_results.get('video_id', 'unknown')}.json",
            mime="application/json"
        )
        
        # Also offer markdown download if available
        if 'enhanced_markdown' in enhanced_results:
            with open(enhanced_results['enhanced_markdown'], 'r') as f:
                markdown_content = f.read()
            
            st.download_button(
                label="📥 Download Enhanced Summary (Markdown)",
                data=markdown_content,
                file_name=f"enhanced_summary_{enhanced_results.get('video_id', 'unknown')}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"Error creating enhanced summary download: {e}")

def generate_summary_markdown(summary_data):
    """Generate a nicely formatted markdown summary from summary data."""
    
    markdown = f"""# DSA Video Summary

## 🎬 Video Information
- **Title**: {summary_data.get('title', 'Unknown')}
- **Duration**: {summary_data.get('duration_formatted', 'Unknown')}
- **Generated**: {summary_data.get('generated_at', 'Unknown')}

## 🚀 Executive Summary
{summary_data.get('executive_summary', 'No executive summary available.')}

## 📚 Learning Objectives
"""
    
    # Add learning objectives
    objectives = summary_data.get('learning_objectives', [])
    if objectives:
        for i, objective in enumerate(objectives, 1):
            markdown += f"{i}. {objective}\n"
    else:
        markdown += "No specific learning objectives identified.\n"
    
    # Add video breakdown
    markdown += "\n## 🔍 Video Breakdown\n\n"
    breakdown = summary_data.get('detailed_breakdown', [])
    if breakdown:
        for section in breakdown:
            markdown += f"### Section {section.get('section_number', 'Unknown')} "
            markdown += f"({section.get('start_formatted', 'Unknown')} - {section.get('end_formatted', 'Unknown')})\n\n"
            markdown += f"{section.get('summary', 'No summary available.')}\n\n"
            
            # Add topics covered
            topics = section.get('topics_covered', [])
            if topics:
                markdown += f"**Topics Covered**: {', '.join(topics)}\n\n"
    else:
        markdown += "No detailed breakdown available.\n\n"
    
    # Add code examples
    code_examples = summary_data.get('code_examples', [])
    if code_examples:
        markdown += "## 💻 Code Examples\n\n"
        for i, example in enumerate(code_examples, 1):
            markdown += f"### Example {i} - {example.get('timestamp_formatted', 'Unknown')}\n\n"
            markdown += f"**Language**: {example.get('detected_language', 'Unknown')}\n\n"
            markdown += f"**Explanation**: {example.get('explanation', 'No explanation available.')}\n\n"
            
            if example.get('original_text'):
                markdown += f"**Code**:\n```{example.get('detected_language', 'text')}\n{example.get('original_text')}\n```\n\n"
    
    # Add topic timeline
    timeline = summary_data.get('topic_timeline', [])
    if timeline:
        markdown += "## 📅 Topic Timeline\n\n"
        for item in timeline:
            topic = item.get('topic', item.get('content', 'Unknown topic'))
            timestamp = item.get('timestamp_formatted', 'Unknown')
            markdown += f"- **{timestamp}**: {topic}\n"
    
    # Add next steps if available
    next_steps = summary_data.get('next_steps', {})
    if next_steps:
        markdown += "\n## 🎯 Next Steps\n\n"
        
        if next_steps.get('practice_problems'):
            markdown += "### Practice Problems\n"
            for problem in next_steps['practice_problems']:
                markdown += f"- {problem}\n"
            markdown += "\n"
        
        if next_steps.get('related_topics'):
            markdown += "### Related Topics\n"
            for topic in next_steps['related_topics']:
                markdown += f"- {topic}\n"
            markdown += "\n"
        
        if next_steps.get('advanced_concepts'):
            markdown += "### Advanced Concepts\n"
            for concept in next_steps['advanced_concepts']:
                markdown += f"- {concept}\n"
    
    markdown += f"""
---

*Generated by DSA Video Summarizer using local AI models*
*Video ID: {summary_data.get('video_id', 'Unknown')}*
"""
    
    return markdown

def generate_markdown_summary(results):
    """Generate a human-readable markdown summary of the processing results."""
    
    # Get basic info
    video_id = results.get('video_id', 'Unknown')
    processing_time = results.get('processing_time_seconds', 0)
    stats = results.get('statistics', {})
    content_analysis = results.get('content_analysis', {})
    
    markdown = f"""# DSA Video Processing Summary

## 📹 Video Information
- **Video ID**: {video_id}
- **Processing Time**: {processing_time:.1f} seconds
- **Status**: ✅ Processing completed successfully

## 📊 Processing Statistics
| Metric | Value |
|--------|-------|
| **Segments** | {stats.get('total_segments', 0)} |
| **Topics** | {stats.get('total_topics', 0)} |
| **Algorithms** | {stats.get('total_algorithms', 0)} |
| **Frames** | {stats.get('total_frames', 0)} |

## 🔍 Content Analysis

### Topics Found
"""
    
    # Add topics
    topics = list(content_analysis.get('topics_mentioned', {}).keys())
    if topics:
        unique_topics = list(set(topics))
        for i, topic in enumerate(unique_topics, 1):
            markdown += f"{i}. **{topic}**\n"
    else:
        markdown += "No specific DSA topics detected\n"
    
    markdown += "\n### Algorithms Mentioned\n"
    
    # Add algorithms
    algorithms = [alg.get('algorithm', '') for alg in content_analysis.get('algorithms_mentioned', [])]
    if algorithms:
        unique_algorithms = list(set(algorithms))
        for i, algorithm in enumerate(unique_algorithms, 1):
            markdown += f"{i}. **{algorithm}**\n"
    else:
        markdown += "No specific algorithms detected\n"
    
    # Add summary if available
    if 'summary' in results:
        markdown += f"\n## 📝 Video Summary\n\n{results['summary']}\n"
    
    # Add segments if available
    if 'segments' in results and results['segments']:
        markdown += f"\n## 🎬 Video Segments\n\n"
        for i, segment in enumerate(results['segments'][:10], 1):  # Show first 10 segments
            timestamp = segment.get('timestamp', 'Unknown')
            content = segment.get('content', 'No content')
            markdown += f"### Segment {i} (at {timestamp})\n{content[:200]}...\n\n"
        
        if len(results['segments']) > 10:
            markdown += f"... and {len(results['segments']) - 10} more segments\n"
    
    # Add processing details
    markdown += f"""
## ⚙️ Processing Details
- **Processing Date**: {results.get('processing_date', 'Unknown')}
- **Model Used**: {results.get('model_used', 'Unknown')}
- **Processing Pipeline**: DSA Video Summarizer v1.0

---
*Generated by DSA Video Summarizer & Chatbot*
"""
    
    return markdown

if __name__ == "__main__":
    main()
