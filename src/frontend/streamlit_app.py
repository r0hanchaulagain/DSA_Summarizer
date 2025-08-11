"""Streamlit frontend for DSA Video Summarizer."""

import streamlit as st
import os
import sys
import json
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

def safe_display_image(img_path: str, caption: str = ""):
    """Safely display an image with error handling."""
    try:
        # Check if file exists and is readable
        if not os.path.exists(img_path):
            st.warning(f"Image file not found: {img_path}")
            return
        
        # Display image
        st.image(img_path)
        
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        st.write(f"Image path: {img_path}")

def clear_all_history():
    """Clear all stored data including summaries, videos, and temporary files."""
    try:
        import shutil
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Clearing summaries directory...")
        progress_bar.progress(20)
        
        # Clear summaries directory
        if os.path.exists(config.SUMMARIES_DIR):
            shutil.rmtree(config.SUMMARIES_DIR)
            os.makedirs(config.SUMMARIES_DIR, exist_ok=True)
            st.success("✅ Summaries directory cleared")
        else:
            st.info("ℹ️ Summaries directory was already empty")
        
        status_text.text("Clearing videos directory...")
        progress_bar.progress(40)
        
        # Clear videos directory
        if os.path.exists(config.VIDEOS_DIR):
            shutil.rmtree(config.VIDEOS_DIR)
            os.makedirs(config.VIDEOS_DIR, exist_ok=True)
            st.success("✅ Videos directory cleared")
        else:
            st.info("ℹ️ Videos directory was already empty")
        
        status_text.text("Clearing temporary files...")
        progress_bar.progress(60)
        
        # Clear temp directory
        if os.path.exists(config.TEMP_DIR):
            shutil.rmtree(config.TEMP_DIR)
            os.makedirs(config.TEMP_DIR, exist_ok=True)
            st.success("✅ Temporary files cleared")
        else:
            st.info("ℹ️ Temporary directory was already empty")
        
        status_text.text("Clearing vector database...")
        progress_bar.progress(80)
        
        # Clear vector store database
        if os.path.exists(config.CHROMA_DB_PATH):
            shutil.rmtree(config.CHROMA_DB_PATH)
            os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
            st.success("✅ Vector database cleared")
        else:
            st.info("ℹ️ Vector database was already empty")
        
        status_text.text("Clearing SQLite database...")
        progress_bar.progress(90)
        
        # Clear SQLite database
        if os.path.exists(config.SQLITE_DB_PATH):
            os.remove(config.SQLITE_DB_PATH)
            st.success("✅ SQLite database cleared")
        else:
            st.info("ℹ️ SQLite database was already empty")
        
        status_text.text("Clearing session state...")
        progress_bar.progress(100)
        
        # Clear session state
        if 'current_video_id' in st.session_state:
            del st.session_state.current_video_id
        if 'processing_results' in st.session_state:
            del st.session_state.processing_results
        if 'chat_history' in st.session_state:
            del st.session_state.chat_history
        if 'enhanced_summary' in st.session_state:
            del st.session_state.enhanced_summary
        if 'history_selected_video_id' in st.session_state:
            del st.session_state.history_selected_video_id
        
        status_text.text("✅ Complete!")
        st.success("🎉 All history has been cleared successfully!")
        st.info("The application will now start fresh. You can process new videos to create summaries.")
        
    except Exception as e:
        st.error(f"❌ Error clearing history: {e}")
        st.error("Some files may not have been cleared. Please check the data folder manually.")

def clear_temp_files_only():
    """Clear only temporary files while keeping summaries and videos."""
    try:
        import shutil
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Clearing temporary files...")
        progress_bar.progress(50)
        
        # Clear temp directory only
        if os.path.exists(config.TEMP_DIR):
            shutil.rmtree(config.TEMP_DIR)
            os.makedirs(config.TEMP_DIR, exist_ok=True)
            st.success("✅ Temporary files cleared")
        else:
            st.info("ℹ️ Temporary directory was already empty")
        
        status_text.text("Clearing vector database...")
        progress_bar.progress(100)
        
        # Clear vector store database (this is also temporary data)
        if os.path.exists(config.CHROMA_DB_PATH):
            shutil.rmtree(config.CHROMA_DB_PATH)
            os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
            st.success("✅ Vector database cleared")
        else:
            st.info("ℹ️ Vector database was already empty")
        
        status_text.text("✅ Complete!")
        st.success("🧹 Temporary files cleared successfully!")
        st.info("Your summaries and videos are still intact.")
        
    except Exception as e:
        st.error(f"❌ Error clearing temporary files: {e}")

def test_ollama_connection():
    """Test the Ollama connection and provide detailed feedback."""
    try:
        pipeline = initialize_pipeline()
        if hasattr(pipeline, 'llm_manager') and hasattr(pipeline.llm_manager, 'ollama'):
            ollama = pipeline.llm_manager.ollama
            
            # Get health status
            health_status = ollama.get_health_status()
            
            st.write("## 🧪 Ollama Connection Test Results")
            
            # Display status
            if health_status['status'] == 'healthy':
                st.success("✅ **Connection Test: PASSED**")
                st.info(f"**Model**: {health_status['required_model']} is loaded and ready")
                st.info(f"**URL**: {health_status['base_url']}")
                st.info(f"**Available Models**: {', '.join(health_status['available_models'])}")
                
                # Test actual generation
                st.write("**Testing Response Generation...**")
                try:
                    test_response = ollama.generate_response("Hello! Please respond with 'Ollama is working correctly.'", "")
                    if "ollama is working correctly" in test_response.lower():
                        st.success("✅ **Generation Test: PASSED**")
                        st.info(f"**Response**: {test_response}")
                    else:
                        st.warning("⚠️ **Generation Test: PARTIAL**")
                        st.info(f"**Response**: {test_response}")
                except Exception as e:
                    st.error(f"❌ **Generation Test: FAILED**")
                    st.error(f"**Error**: {e}")
                    
            elif health_status['status'] == 'model_not_loaded':
                st.warning("⚠️ **Connection Test: PARTIAL**")
                st.info(f"**URL**: {health_status['base_url']} - ✅ Service is running")
                st.error(f"**Model**: {health_status['required_model']} - ❌ Not loaded")
                st.info(f"**Available**: {', '.join(health_status['available_models'])}")
                st.error("**Solution**: Run `ollama pull codellama:7b`")
                
            elif health_status['status'] == 'connection_error':
                st.error("❌ **Connection Test: FAILED**")
                st.error(f"**URL**: {health_status['base_url']} - Cannot connect")
                st.error("**Solution**: Run `ollama serve` to start the service")
                
            else:
                st.error("❌ **Connection Test: FAILED**")
                st.error(f"**Error**: {health_status['error']}")
                if 'suggestion' in health_status:
                    st.info(f"**Suggestion**: {health_status['suggestion']}")
                    
        else:
            st.error("❌ **Test Failed**: Local LLM manager not available")
            
    except Exception as e:
        st.error(f"❌ **Test Error**: {e}")
        st.error("Could not initialize pipeline for testing")

def main():
    """Main Streamlit application."""
    
    st.title("🎥 DSA Video Summarizer & Chatbot")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        # Keep selected page in session state
        if 'nav_page' not in st.session_state:
            st.session_state.nav_page = "🏠 Home"

        nav_options = [
            "🏠 Home",
            "🎬 Process Video",
            "📄 View Summary",
            "🗂️ Summary History",
            "💬 Chat with Video",
            "📊 Statistics"
        ]

        for i, label in enumerate(nav_options):
            if st.button(
                label,
                key=f"nav_{i}",
                use_container_width=True
            ):
                st.session_state.nav_page = label
                st.rerun()

        page = st.session_state.nav_page
        
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
        
        # AI Model Status
        st.markdown("---")
        st.header("🤖 AI Model Status")
        
        try:
            pipeline = initialize_pipeline()
            if hasattr(pipeline, 'llm_manager'):
                llm_manager = pipeline.llm_manager
                model_info = llm_manager.get_model_info()
                
                # Show primary model
                if model_info['primary_model'] == 'Gemini':
                    st.success(f"🚀 Primary: Gemini API")
                    st.info(f"Fallback: {', '.join(model_info['fallback_models'])}")
                else:
                    st.info(f"🤖 Primary: {model_info['primary_model']}")
                    st.info(f"Fallback: {', '.join(model_info['fallback_models'])}")
                
                # Show current model
                st.write(f"**Current Model**: {model_info['current_model']}")
                
                # Check Ollama specifically for detailed status
                if hasattr(llm_manager, 'ollama'):
                    health_status = llm_manager.ollama.get_health_status()
                    
                    if health_status['status'] == 'healthy':
                        st.success(f"✅ Ollama: {health_status['required_model']} is ready")
                        st.info(f"Available models: {', '.join(health_status['available_models'][:3])}")
                    elif health_status['status'] == 'model_not_loaded':
                        st.warning(f"⚠️ Ollama: Model {health_status['required_model']} not loaded")
                        st.info(f"Available: {', '.join(health_status['available_models'][:3])}")
                        st.error("💡 Run: ollama pull codellama:7b")
                    elif health_status['status'] == 'connection_error':
                        st.error("❌ Ollama: Cannot connect to service")
                        st.error("💡 Run: ollama serve")
                    else:
                        st.error(f"❌ Ollama: {health_status['error']}")
                        if 'suggestion' in health_status:
                            st.info(f"💡 {health_status['suggestion']}")
            else:
                st.info("ℹ️ AI model status unavailable")
        except Exception as e:
            st.error(f"❌ Error checking AI model status: {e}")
        
        # Troubleshooting help
        with st.expander("🔧 Troubleshooting AI Models"):
            st.write("**Dual-Model System:**")
            st.write("• **Primary**: Gemini API (when configured)")
            st.write("• **Fallback**: Ollama local model")
            st.write("• **Final Fallback**: Rule-based system")
            
            st.write("**Common Ollama Issues:**")
            st.write("1. **Service not running**: Run `ollama serve`")
            st.write("2. **Model not loaded**: Run `ollama pull codellama:7b`")
            st.write("3. **Timeout errors**: Check if model is responding")
            st.write("4. **Port conflicts**: Ensure port 11434 is free")
            st.write("5. **Memory issues**: Close other applications using GPU")
            
            st.write("**Quick Fix Commands:**")
            st.code("ollama serve\nollama pull codellama:7b\nollama list", language="bash")
            
            # Test Ollama connection
            if st.button("🧪 Test Ollama Connection", use_container_width=True):
                test_ollama_connection()
        
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
    elif page == "🗂️ Summary History":
        show_summary_history_page()
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

            # Check for existing summary to enable reuse
            existing_video_id = None
            if input_type == "url" and video_input:
                try:
                    info = pipeline.get_video_info_preview(video_input)
                    existing_video_id = info.get('video_id')
                except Exception:
                    existing_video_id = None
            elif input_type == "file":
                # For files, the video_id is filename stem
                try:
                    existing_video_id = os.path.splitext(os.path.basename(video_input))[0]
                except Exception:
                    existing_video_id = None

            summary_exists = False
            if existing_video_id:
                # Handle cached pipeline that may not have the new has_summary method
                try:
                    has_summary_fn = getattr(pipeline.summarizer, 'has_summary', None)
                    if callable(has_summary_fn):
                        summary_exists = has_summary_fn(existing_video_id)
                    else:
                        summary_exists = os.path.exists(
                            os.path.join(config.SUMMARIES_DIR, f"{existing_video_id}_summary.json")
                        )
                except Exception:
                    summary_exists = os.path.exists(
                        os.path.join(config.SUMMARIES_DIR, f"{existing_video_id}_summary.json")
                    )
            
            # Processing controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("Ready to process! This may take several minutes depending on video length.")
                
                # Display processing summary here if available
                if st.session_state.processing_results:
                    st.markdown("---")
                    display_processing_summary(st.session_state.processing_results)
            
            with col2:
                if summary_exists:
                    st.success("Previously generated summary found.")
                    if st.button("Use Existing Summary", type="primary", use_container_width=True):
                        # Load and display existing summary immediately
                        st.session_state.current_video_id = existing_video_id
                        # Build a lightweight results dict to drive the UI
                        loaded_summary = pipeline.summarizer.load_summary(existing_video_id)
                        if loaded_summary:
                            st.session_state.processing_results = {
                                'video_id': existing_video_id,
                                'processing_status': 'completed',
                                'processing_time_seconds': 0.0,
                                'video_metadata': {'video_id': existing_video_id, 'title': loaded_summary.get('title', 'Unknown')},
                                'transcription_data': {},
                                'content_analysis': {},
                                'frames_analysis': {},
                                'summary_result': {'summary_data': loaded_summary},
                                'vector_store_success': False,
                                'processed_at': loaded_summary.get('generated_at', ''),
                                'statistics': loaded_summary.get('statistics', {})
                            }
                            st.rerun()
                    if st.button("Regenerate Summary", use_container_width=True):
                        process_video(pipeline, video_input, input_type)
                else:
                    if st.button("Start Processing", type="primary", use_container_width=True):
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

def display_processing_summary(results):
    """Display a summary of processing results."""
    
    # Create a card-like container for the processing summary
    with st.container():
        st.markdown("---")
        
        # Header with icon
        st.subheader("🔄 Processing Summary")
        
        # Basic metrics in a compact horizontal layout
        col1, col2, col3, col4 = st.columns(4)
        
        stats = results.get('statistics', {})
        
        with col1:
            st.metric("Segments", stats.get('total_segments', 0))
        with col2:
            st.metric("Topics", stats.get('total_topics', 0))
        with col3:
            st.metric("Algorithms", stats.get('total_algorithms', 0))
        with col4:
            st.metric("Frames", stats.get('total_frames', 0))
        
        st.markdown("---")
        
        # Content analysis in a compact layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Topics Found:**")
            topics = list(results.get('content_analysis', {}).get('topics_mentioned', {}).keys())
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
            algorithms = [alg.get('algorithm','') for alg in results.get('content_analysis', {}).get('algorithms_mentioned', [])]
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
                st.session_state.nav_page = "📄 View Summary"
                st.rerun()
        
        with col2:
            if st.button("💬 Start Chatting", use_container_width=True):
                st.session_state.nav_page = "💬 Chat with Video"
                st.rerun()
        
        with col3:
            if st.button("📥 Download Results", use_container_width=True):
                download_results(results)

def show_summary_page():
    """Display the video summary page."""
    
    st.header("📄 Video Summary")
    
    if not st.session_state.current_video_id:
        st.warning("⚠️ No video currently loaded. Please process a video first.")
        st.write("💡 **To fix this:** Go to 'Process Video' page and process a video, then return here.")
        return
    
    pipeline = initialize_pipeline()
    
    try:
        # Try to load summary
        try:
            summary_data = pipeline.summarizer.load_summary(st.session_state.current_video_id)
        except Exception as load_error:
            st.error(f"❌ Error loading summary: {load_error}")
            st.write("💡 **Troubleshooting:**")
            st.write("1. Check if the summary file exists in the data folder")
            st.write("2. Try processing the video again")
            st.write("3. Check the console for detailed error messages")
            return
        
        if not summary_data:
            st.error("❌ No summary found for current video")
            st.write("💡 **Possible causes:**")
            st.write("1. Video processing may have failed")
            st.write("2. Summary file may be corrupted")
            st.write("3. Try processing the video again")
            
            # Add a button to go back to process video
            if st.button("🎬 Go to Process Video", use_container_width=True):
                st.session_state.nav_page = "🎬 Process Video"
                st.rerun()
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
        
        # Showcase key frames extracted from the video (if any)
        try:
            if summary_data.get('key_frames'):
                st.markdown("---")
                st.header("Key Frames From The Video")

                frames = summary_data['key_frames']
                
                # Display in rows of 3
                for i in range(0, len(frames), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(frames):
                            frame = frames[idx]
                            with col:
                                st.caption(f"{frame.get('timestamp_formatted', '')} • {frame.get('type','')}")
                                img_path = frame.get('frame_path')
                                if img_path and os.path.exists(img_path):
                                    safe_display_image(img_path, frame.get('caption', ''))
                                if frame.get('caption'):
                                    st.caption(frame['caption'])
        except Exception as e:
            st.error(f"❌ Error displaying key frames: {e}")

        # Video breakdown
        st.markdown("---")
        st.header("Video Breakdown")
        
        try:
            for section in summary_data['detailed_breakdown']:
                with st.expander(
                    f"Section {section['section_number']} ({section['start_formatted']} - {section['end_formatted']})"
                ):
                    st.write(section['summary'])
                    if section['topics_covered']:
                        # Remove duplicates from topics
                        unique_topics = list(set(section['topics_covered']))
                        st.write(f"**Topics:** {', '.join(unique_topics)}")

                    # Show representative frames for this section if available
                    frames = section.get('frames', [])
                    if frames:
                        cols = st.columns(3)
                        for i, frame in enumerate(frames[:6]):
                            with cols[i % 3]:
                                st.caption(f"{frame.get('timestamp_formatted','')} • {frame.get('type','')}")
                                img_path = frame.get('frame_path')
                                if img_path and os.path.exists(img_path):
                                    safe_display_image(img_path, frame.get('caption', ''))
                                if frame.get('caption'):
                                    st.caption(frame['caption'])
        except Exception as e:
            st.error(f"❌ Error displaying video breakdown: {e}")
        
        # Code examples
        try:
            if summary_data['code_examples']:
                st.markdown("---")
                st.header("Code Examples")
                
                for i, example in enumerate(summary_data['code_examples'], 1):
                    with st.expander(f"Example {i} - {example['timestamp_formatted']}"):
                        st.write(f"**Language:** {example['detected_language']}")
                        st.write(example['explanation'])
                        if example['original_text']:
                            st.code(example['original_text'], language=example['detected_language'])
        except Exception as e:
            st.error(f"❌ Error displaying code examples: {e}")
        
        # Gemini Enhancement Button
        try:
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
        except Exception as e:
            st.error(f"❌ Error setting up Gemini enhancement: {e}")
        
        # Display enhanced summary if available
        try:
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
        except Exception as e:
            st.error(f"❌ Error displaying enhanced summary: {e}")
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

def show_summary_history_page():
    """Display list of all stored summaries and allow viewing directly."""
    st.header("🗂️ Summary History")

    pipeline = initialize_pipeline()
    
    # Clear history button and storage info
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Calculate storage usage
    total_size = 0
    file_count = 0
    
    for directory in [config.SUMMARIES_DIR, config.VIDEOS_DIR, config.TEMP_DIR]:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except:
                        pass
    
    with col1:
        st.metric("📁 Files", file_count)
    
    with col2:
        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            if size_mb > 1024:
                size_gb = size_mb / 1024
                st.metric("💾 Storage", f"{size_gb:.1f} GB")
            else:
                st.metric("💾 Storage", f"{size_mb:.1f} MB")
        else:
            st.metric("💾 Storage", "0 MB")
        
        # Clear buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
                st.session_state.show_clear_confirmation = True
            st.caption("⚠️ Deletes everything")
        with col_b:
            if st.button("🧹 Clear Temp Files", type="secondary", use_container_width=True):
                clear_temp_files_only()
            st.caption("💾 Keeps summaries & videos")
    
    with col3:
        summary_count = 0
        if os.path.exists(config.SUMMARIES_DIR):
            summary_count = len([f for f in os.listdir(config.SUMMARIES_DIR) if f.endswith('_summary.json')])
        st.metric("🗂️ Summaries", summary_count)
    
    # Show message if no data to clear
    if file_count == 0:
        st.info("ℹ️ No stored data found. The data folder is empty.")
    
    # Confirmation dialog
    if st.session_state.get('show_clear_confirmation', False):
        st.warning("⚠️ This will permanently delete ALL stored data!")
        
        # Show what will be deleted
        with st.expander("📋 What will be deleted:", expanded=True):
            st.write("**The following data will be permanently removed:**")
            
            # Count files in each directory
            summary_count = 0
            video_count = 0
            temp_count = 0
            
            if os.path.exists(config.SUMMARIES_DIR):
                summary_count = len([f for f in os.listdir(config.SUMMARIES_DIR) if f.endswith('_summary.json')])
            if os.path.exists(config.VIDEOS_DIR):
                video_count = len([f for f in os.listdir(config.VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
            if os.path.exists(config.TEMP_DIR):
                temp_count = len(os.listdir(config.TEMP_DIR))
            
            st.write(f"• 📄 **Summaries**: {summary_count} summary files")
            st.write(f"• 🎬 **Videos**: {video_count} video files")
            st.write(f"• 🗂️ **Temporary files**: {temp_count} temp files")
            st.write(f"• 🗄️ **Vector database**: All embeddings and search data")
            st.write(f"• 💾 **SQLite database**: All application data")
            st.write(f"• 🔄 **Session data**: Current video, chat history, etc.")
        
        st.error("**This action cannot be undone!**")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_clear_confirmation = False
                st.rerun()
        with col3:
            if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
                clear_all_history()
                st.session_state.show_clear_confirmation = False
                st.rerun()

    try:
        # Handle cached instances that may not have list_summaries
        list_fn = getattr(pipeline.summarizer, 'list_summaries', None)
        if callable(list_fn):
            summaries = list_fn()
        else:
            summaries = []
            try:
                if os.path.exists(config.SUMMARIES_DIR):
                    for name in os.listdir(config.SUMMARIES_DIR):
                        if name.endswith('_summary.json'):
                            path = os.path.join(config.SUMMARIES_DIR, name)
                            try:
                                with open(path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    summaries.append({
                                        'video_id': data.get('video_id', os.path.splitext(name)[0].replace('_summary', '')),
                                        'title': data.get('title', 'Unknown'),
                                        'generated_at': data.get('generated_at', ''),
                                        'duration_formatted': data.get('duration_formatted', ''),
                                        'file': path
                                    })
                            except Exception:
                                continue
                summaries.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
            except Exception:
                summaries = []

        if not summaries:
            st.info("No summaries found yet. Process a video to create your first summary.")
            return

        # Optional filter
        query = st.text_input("Filter by title or video ID:", placeholder="Type to filter...")
        if query:
            q = query.lower()
            summaries = [s for s in summaries if q in str(s.get('title','')).lower() or q in str(s.get('video_id','')).lower()]

        # List summaries with action buttons
        for item in summaries:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.write(f"**{item.get('title','Unknown')}**")
                    st.caption(f"ID: {item.get('video_id','')} ")
                with col2:
                    st.write(item.get('duration_formatted', ''))
                with col3:
                    st.write(item.get('generated_at','')[:19])
                with col4:
                    if st.button("View", key=f"history_view_{item['video_id']}"):
                        st.session_state.history_selected_video_id = item['video_id']
                        st.rerun()

        # If a summary was selected, render it below
        selected_id = st.session_state.get('history_selected_video_id')
        if selected_id:
            st.markdown("---")
            st.subheader("Selected Summary")
            data = pipeline.summarizer.load_summary(selected_id)
            if not data:
                st.error("Summary could not be loaded.")
                return

            # Quick actions
            colA, colB = st.columns(2)
            with colA:
                if st.button("Open in View Summary", key="open_in_view_summary"):
                    st.session_state.current_video_id = selected_id
                    st.rerun()
            with colB:
                st.caption("Use Process Video page to regenerate if needed.")

            # Basic info
            col1, col2 = st.columns([3,1])
            with col1:
                st.subheader(f"🎬 {data.get('title','Unknown')}")
            with col2:
                st.metric("⏱️ Duration", data.get('duration_formatted',''))

            # Executive summary
            st.markdown("---")
            st.write(data.get('executive_summary',''))

            # Key frames
            key_frames = data.get('key_frames', [])
            if key_frames:
                st.markdown("---")
                st.header("Key Frames From The Video")
                for i in range(0, len(key_frames), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(key_frames):
                            frame = key_frames[idx]
                            with col:
                                st.caption(f"{frame.get('timestamp_formatted','')} • {frame.get('type','')}")
                                img_path = frame.get('frame_path')
                                if img_path and os.path.exists(img_path):
                                    safe_display_image(img_path, frame.get('caption', ''))
                                if frame.get('caption'):
                                    st.caption(frame['caption'])
            # Learning objectives
            if data.get('learning_objectives'):
                st.markdown("---")
                st.header("Learning Objectives")
                for i, objective in enumerate(data['learning_objectives'], 1):
                    st.write(f"{i}. {objective}")

            # Breakdown (compact)
            if data.get('detailed_breakdown'):
                st.markdown("---")
                st.header("Video Breakdown")
                for section in data['detailed_breakdown'][:10]:
                    with st.expander(
                        f"Section {section['section_number']} ({section['start_formatted']} - {section['end_formatted']})"
                    ):
                        st.write(section.get('summary',''))
                        topics = list(set(section.get('topics_covered', [])))
                        if topics:
                            st.write(f"**Topics:** {', '.join(topics)}")

    except Exception as e:
        st.error(f"Error loading summary history: {e}")

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
        
        # Check for suggested query from button clicks
        if 'suggested_query' in st.session_state and st.session_state.suggested_query:
            user_input = st.session_state.suggested_query
            # Clear the suggested query to prevent it from being processed again
            del st.session_state.suggested_query
        else:
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
        
        # Get dynamic suggestions from the chatbot
        try:
            suggestions = chatbot.get_suggested_questions(st.session_state.current_video_id)
        except Exception as e:
            # Fallback to static suggestions if dynamic ones fail
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
                    # Set the suggested query to be processed in the next iteration
                    st.session_state.suggested_query = suggestion
                    st.rerun()
    
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
    
    # Add key frames (with image links if paths are available)
    key_frames = summary_data.get('key_frames', [])
    if key_frames:
        markdown += "\n## 🖼️ Key Frames\n\n"
        for frame in key_frames:
            ts = frame.get('timestamp_formatted', 'Unknown')
            ftype = frame.get('type', 'frame')
            fpath = frame.get('frame_path', '')
            caption = frame.get('caption', '')
            markdown += f"- {ts} • {ftype.capitalize()}\n"
            if fpath:
                markdown += f"  \n  ![]({fpath})\n"
            if caption:
                markdown += f"  \n  _{caption}_\n"
        markdown += "\n"

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
