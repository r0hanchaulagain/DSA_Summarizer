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
        
        # API Key validation
        if not config.GEMINI_API_KEY:
            st.error("⚠️ Gemini API key not configured!")
            st.info("Please set your Gemini API key in the .env file")
        else:
            st.success("✅ Gemini API key configured")
        
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
        st.header("Welcome to DSA Video Summarizer! 🚀")
        
        st.markdown("""
        This AI-powered tool helps you learn Data Structures and Algorithms by:
        
        ### 🎯 Key Features
        - **📹 Video Processing**: Automatically analyze YouTube videos or uploaded files
        - **📝 Smart Transcription**: Extract and timestamp all spoken content
        - **🧠 Content Analysis**: Identify DSA topics, algorithms, and code snippets
        - **📊 Comprehensive Summaries**: Generate detailed study materials
        - **💬 Interactive Chat**: Ask questions about the video content
        - **🔍 Timestamp Search**: Find specific topics with exact timing
        
        ### 🛠️ How It Works
        1. **Input**: Provide a YouTube URL or upload a video file
        2. **Processing**: AI analyzes audio, video frames, and content
        3. **Analysis**: Extracts DSA topics, algorithms, and code examples
        4. **Summary**: Generates comprehensive study materials
        5. **Chat**: Ask questions and get instant answers about the content
        
        ### 📚 Perfect For
        - Students learning DSA concepts
        - Interview preparation
        - Review and quick reference
        - Understanding complex algorithms
        - Code implementation examples
        """)
        
    with col2:
        st.header("🚀 Quick Start")
        
        # Quick stats
        try:
            pipeline = initialize_pipeline()
            vector_store = pipeline.vector_store
            stats = vector_store.get_collection_stats()
            
            st.metric("📼 Videos Processed", stats.get('unique_videos', 0))
            st.metric("📄 Total Documents", stats.get('total_documents', 0))
            
        except Exception as e:
            st.warning("Could not load statistics")
        
        st.markdown("---")
        
        # Sample video for testing
        st.subheader("🎬 Try with Sample Video")
        sample_url = "https://www.youtube.com/watch?v=8hly31xKli0"  # Sample DSA video
        if st.button("🎯 Process Sample Video"):
            st.session_state.sample_url = sample_url
            st.info(f"Sample URL loaded: {sample_url}")
        
        st.markdown("---")
        
        # Recent activity
        st.subheader("📊 System Status")
        if config.validate_api_keys():
            st.success("✅ All systems operational")
        else:
            st.error("❌ Configuration issues detected")

def show_process_video_page():
    """Display the video processing page."""
    
    st.header("🎬 Process DSA Video")
    
    pipeline = initialize_pipeline()
    
    # Input method selection
    input_method = st.radio(
        "📥 Choose input method:",
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
        if video_input and st.button("🔍 Preview Video Info"):
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
            st.info(f"📁 File uploaded: {uploaded_file.name} ({format_file_size(file_size)})")
    
    # Validation and processing
    if video_input:
        # Validate requirements
        validation = pipeline.validate_video_requirements(video_input, input_type)
        
        if not validation['valid']:
            st.error("❌ Video does not meet requirements:")
            for error in validation['errors']:
                st.error(f"• {error}")
        else:
            if validation.get('warnings'):
                for warning in validation['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            st.success("✅ Video meets requirements")
            
            # Processing controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("🎯 Ready to process! This may take several minutes depending on video length.")
            
            with col2:
                if st.button("🚀 Start Processing", type="primary"):
                    process_video(pipeline, video_input, input_type)

def process_video(pipeline, video_input, input_type):
    """Process the video through the pipeline."""
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Start processing
            status_text.text("🎬 Initializing video processing...")
            progress_bar.progress(10)
            
            # Update progress periodically
            def update_progress(step, message):
                progress_values = {
                    1: 20, 2: 30, 3: 40, 4: 60, 5: 70, 6: 80, 7: 90, 8: 95
                }
                progress_bar.progress(progress_values.get(step, 10))
                status_text.text(message)
            
            # Process video
            update_progress(1, "📥 Downloading/validating video...")
            results = pipeline.process_video(video_input, input_type)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing completed!")
            
            if results['processing_status'] == 'completed':
                st.success(f"🎉 Video processed successfully in {results['processing_time_seconds']:.1f} seconds!")
                
                # Store results in session state
                st.session_state.current_video_id = results['video_id']
                st.session_state.processing_results = results
                
                # Display summary
                display_processing_summary(results)
                
            else:
                st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")
            st.error(f"Error during processing: {str(e)}")
            st.error(traceback.format_exc())

def display_processing_summary(results):
    """Display a summary of processing results."""
    
    st.markdown("---")
    st.header("📊 Processing Summary")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = results['statistics']
    
    with col1:
        st.metric("📝 Segments", stats['total_segments'])
    with col2:
        st.metric("🎯 Topics", stats['total_topics'])
    with col3:
        st.metric("🔢 Algorithms", stats['total_algorithms'])
    with col4:
        st.metric("🖼️ Frames", stats['total_frames'])
    
    # Content preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Topics Found")
        topics = list(results['content_analysis']['topics_mentioned'].keys())
        if topics:
            for topic in topics[:5]:  # Show first 5 topics
                st.write(f"• {topic}")
            if len(topics) > 5:
                st.write(f"... and {len(topics) - 5} more")
        else:
            st.info("No specific DSA topics detected")
    
    with col2:
        st.subheader("🧮 Algorithms Mentioned")
        algorithms = [alg['algorithm'] for alg in results['content_analysis']['algorithms_mentioned']]
        if algorithms:
            for algorithm in algorithms[:5]:  # Show first 5 algorithms
                st.write(f"• {algorithm}")
            if len(algorithms) > 5:
                st.write(f"... and {len(algorithms) - 5} more")
        else:
            st.info("No specific algorithms detected")
    
    # Quick actions
    st.markdown("---")
    st.header("🎯 Next Steps")
    
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
        
        # Executive summary
        st.markdown("---")
        st.header("📋 Executive Summary")
        st.write(summary_data['executive_summary'])
        
        # Learning objectives
        st.markdown("---")
        st.header("🎯 Learning Objectives")
        for i, objective in enumerate(summary_data['learning_objectives'], 1):
            st.write(f"{i}. {objective}")
        
        # Video breakdown
        st.markdown("---")
        st.header("📚 Video Breakdown")
        
        for section in summary_data['detailed_breakdown']:
            with st.expander(
                f"Section {section['section_number']} ({section['start_formatted']} - {section['end_formatted']})"
            ):
                st.write(section['summary'])
                if section['topics_covered']:
                    st.write(f"**Topics:** {', '.join(section['topics_covered'])}")
        
        # Code examples
        if summary_data['code_examples']:
            st.markdown("---")
            st.header("💻 Code Examples")
            
            for i, example in enumerate(summary_data['code_examples'], 1):
                with st.expander(f"Example {i} - {example['timestamp_formatted']}"):
                    st.write(f"**Language:** {example['detected_language']}")
                    st.write(example['explanation'])
                    if example['original_text']:
                        st.code(example['original_text'], language=example['detected_language'])
        
        # Topic timeline
        if summary_data['topic_timeline']:
            st.markdown("---")
            st.header("📈 Topic Timeline")
            
            for item in summary_data['topic_timeline'][:10]:  # Show first 10 items
                st.write(f"**{item['timestamp_formatted']}**: {item['content']} ({item['type']})")
        
        # Next steps
        st.markdown("---")
        st.header("🚀 Next Steps")
        
        next_steps = summary_data['next_steps']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Practice Problems")
            for problem in next_steps.get('practice_problems', []):
                st.write(f"• {problem}")
        
        with col2:
            st.subheader("🔗 Related Topics")
            for topic in next_steps.get('related_topics', []):
                st.write(f"• {topic}")
        
        with col3:
            st.subheader("🎓 Advanced Concepts")
            for concept in next_steps.get('advanced_concepts', []):
                st.write(f"• {concept}")
        
        # Download options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download JSON Summary", use_container_width=True):
                st.download_button(
                    label="💾 Download",
                    data=str(summary_data),
                    file_name=f"{st.session_state.current_video_id}_summary.json",
                    mime="application/json"
                )
        
        with col2:
            # Check if markdown file exists
            markdown_file = summary_data.get('markdown_file')
            if markdown_file and os.path.exists(markdown_file):
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                st.download_button(
                    label="📄 Download Markdown",
                    data=markdown_content,
                    file_name=f"{st.session_state.current_video_id}_summary.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error loading summary: {str(e)}")

def show_chat_page():
    """Display the chat interface."""
    
    st.header("💬 Chat with Video")
    
    if not st.session_state.current_video_id:
        st.warning("⚠️ No video currently loaded. Please process a video first.")
        return
    
    try:
        chatbot = initialize_chatbot()
        
        # Chat interface
        st.subheader(f"🎬 Chatting about: {st.session_state.current_video_id}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
                
                # Show timestamps if available
                if 'timestamps' in message:
                    with st.expander("📍 Related Timestamps"):
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
                    with st.expander("📚 Related Content"):
                        for content in response['related_content'][:3]:
                            st.write(f"**{content['metadata']['content_type']}**: {content['document'][:200]}...")
                            if 'start_formatted' in content['metadata']:
                                st.write(f"⏰ {content['metadata']['start_formatted']}")
                
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
    
    # JSON results
    import json
    results_json = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="📄 Download Full Results (JSON)",
        data=results_json,
        file_name=f"{results['video_id']}_results.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
