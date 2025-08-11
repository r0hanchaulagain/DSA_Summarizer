# 🎥 DSA Video Summarizer & Chatbot

An AI-powered tool that automatically analyzes Data Structures and Algorithms (DSA) educational videos and creates comprehensive summaries with an intelligent chatbot interface.

## 🎯 Features

### 📹 Video Processing
- **YouTube Integration**: Download and process videos directly from YouTube URLs
- **File Upload Support**: Process local video files (MP4, AVI, MOV, MKV)
- **Smart Audio Extraction**: High-quality audio extraction for accurate transcription
- **Frame Analysis**: Extract and analyze video frames for code snippets and diagrams

### 🧠 AI-Powered Analysis
- **Speech-to-Text**: Accurate transcription using OpenAI Whisper with timestamps
- **Content Analysis**: Automatic identification of DSA topics, algorithms, and concepts
- **Code Detection**: Extract and analyze code snippets with language detection
- **Complexity Analysis**: Identify time and space complexity discussions
- **Local AI Processing**: Offline summarization and chatbot using local LLM models
- **Dual Summary System**: Local LLM summaries + optional Gemini enhancement

### 📊 Comprehensive Summaries
- **Executive Summary**: Concise overview of video content
- **Learning Objectives**: Clear educational goals based on content
- **Detailed Breakdown**: Section-by-section analysis with timestamps
- **Code Examples**: Extracted code with explanations
- **Topic Timeline**: Chronological overview of topics discussed
- **Next Steps**: Suggested practice problems and related topics

### 💬 Intelligent Chatbot
- **Video-Specific Q&A**: Ask questions about the specific video content
- **Timestamp Search**: Find when specific topics are discussed
- **Context-Aware Responses**: Intelligent answers based on video content
- **Multiple Query Types**: Support for topic overview, algorithm search, code examples, and more
- **🚀 Dual AI System**: Gemini API (primary) + Ollama local LLM (fallback) + Rule-based system (final fallback)

### 📄 Export Options
- **Markdown Summaries**: Well-formatted study materials
- **JSON Data**: Complete processing results for further analysis
- **Timestamped References**: Direct links to specific video moments

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │    │   Processing    │    │    Output       │
│                 │    │    Pipeline     │    │                 │
│ • YouTube URLs  │───▶│                 │───▶│ • Summaries     │
│ • Local Files   │    │ • Transcription │    │ • Chatbot       │
└─────────────────┘    │ • Analysis      │    │ • Exports       │
                       │ • Summarization │    └─────────────────┘
                       └─────────────────┘
```

### Core Components

1. **Video Processor**: Download and extract content from videos
2. **AI Engine**: Transcribe, analyze, and summarize content
3. **Vector Store**: Store and search video content using ChromaDB
4. **Chatbot System**: Process queries and generate responses
5. **Frontend**: Streamlit-based user interface

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg (for video processing)
- Tesseract OCR (for text extraction from frames)
- **Optional**: Gemini API key for enhanced features
- **Recommended**: Local LLM setup (Ollama) for offline operation
- **CPU Optimized**: CodeLlama:7b runs efficiently without GPU

### Local LLM Setup (Recommended)

For offline operation without external APIs:

1. **Install Ollama** (see [LOCAL_LLM_SETUP.md](LOCAL_LLM_SETUP.md) for detailed instructions):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. **Start Ollama service**:
```bash
ollama serve
```

3. **Download a model**:
```bash
ollama pull codellama:7b  # Excellent for DSA & code (CPU optimized)
# or
ollama pull llama2:7b  # Good for general DSA concepts
```

4. **Test the setup**:
```bash
ollama run codellama:7b "Explain binary search"
```

5. **Run the connection test**:
```bash
python test_ollama.py
```

6. **Test the dual-model system**:
```bash
python test_dual_models.py
```

### 🚀 Dual AI Model System

The application now features a sophisticated dual-model architecture:

#### **Primary Model: Gemini API**
- **When Available**: Automatically used for all AI operations
- **Benefits**: Higher quality responses, better understanding of complex queries
- **Requirements**: Valid Gemini API key in environment variables

#### **Fallback Model: Ollama Local LLM**
- **When Used**: When Gemini is unavailable or fails
- **Benefits**: Offline operation, no API costs, privacy-focused
- **Models**: CodeLlama:7b (recommended), Llama2:7b, or any Ollama model

#### **Final Fallback: Rule-based System**
- **When Used**: When both AI models fail
- **Benefits**: Always available, basic but reliable responses
- **Features**: Template-based summaries and explanations

#### **Automatic Fallback Logic**
1. **Try Gemini first** (if API key configured)
2. **Fall back to Ollama** (if Gemini fails or unavailable)
3. **Use rule-based system** (if all AI models fail)
4. **Provide user feedback** about which model was used

This ensures maximum reliability while maintaining high-quality AI responses.

## 🔧 Troubleshooting

### Common Ollama Issues

#### 1. **Connection Timeout Errors**
```
Error: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=60)
```

**Solutions:**
- **Check if Ollama is running**: `ps aux | grep ollama`
- **Start Ollama service**: `ollama serve`
- **Verify port availability**: `netstat -tlnp | grep 11434`
- **Test connection**: `python test_ollama.py`

#### 2. **Model Not Found Errors**
```
Error: Model 'codellama:7b' not found
```

**Solutions:**
- **List available models**: `ollama list`
- **Pull required model**: `ollama pull codellama:7b`
- **Check model status**: `ollama show codellama:7b`

#### 3. **Service Not Responding**
```
Error: Cannot connect to Ollama service
```

**Solutions:**
- **Restart Ollama**: `pkill ollama && ollama serve`
- **Check system resources**: Ensure sufficient RAM/CPU
- **Verify firewall settings**: Port 11434 should be accessible
- **Check logs**: `ollama serve --verbose`

#### 4. **Performance Issues**
- **Reduce model size**: Use `codellama:7b` instead of larger models
- **Close other applications**: Free up system resources
- **Adjust timeout settings**: Increase `OLLAMA_TIMEOUT` in environment
- **Use CPU optimization**: Ensure model is CPU-optimized

### Quick Diagnostic Commands

```bash
# Check Ollama status
ollama list

# Test basic connectivity
curl http://localhost:11434/api/tags

# Run comprehensive test
python test_ollama.py

# Check system resources
htop
free -h
```

### Environment Variables

You can customize Ollama behavior with these environment variables:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="codellama:7b"
export OLLAMA_TIMEOUT="120"
export OLLAMA_RETRY_ATTEMPTS="2"
```

The system now provides **two levels of summarization**:

#### **Level 1: Local LLM Summary (Always Available)**
- **Generated by**: CodeLlama:7b (local, no internet required)
- **Quality**: Good to excellent for DSA content
- **Speed**: Fast (3-8 seconds)
- **Cost**: Free (runs locally)

#### **Level 2: Gemini Enhanced Summary (Optional)**
- **Generated by**: Gemini API (requires internet + API key)
- **Quality**: Excellent to outstanding
- **Speed**: Fast (5-15 seconds)
- **Cost**: Small API usage fee
- **Enhancement**: Improves existing local summary by 2-3x quality

#### **How It Works:**
1. **First**: System generates local LLM summary automatically
2. **Then**: User can click "✨ Enhance with Gemini" button
3. **Result**: Gemini takes the local summary and significantly improves it
4. **Benefit**: Best of both worlds - local processing + enhanced quality

### System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg tesseract-ocr tesseract-ocr-eng
```

#### macOS:
```bash
brew install ffmpeg tesseract
```

#### Windows:
- Download FFmpeg from https://ffmpeg.org/download.html
- Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki

### Python Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd dsa-video-summarizer
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables (optional):**
```bash
cp .env.example .env
# Edit .env and optionally add your Gemini API key for enhanced features
# For offline operation, no API key is required
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Optional: Gemini API Configuration (for enhanced features)
# GEMINI_API_KEY=your_gemini_api_key_here

# Local LLM Configuration (recommended for offline use)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=codellama:7b

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO

# Video Processing Configuration
MAX_VIDEO_SIZE_MB=500
FRAME_EXTRACTION_INTERVAL=30
MAX_PROCESSING_TIME_MINUTES=30

# Database Configuration
CHROMA_DB_PATH=./data/embeddings
SQLITE_DB_PATH=./data/app.db

# Paths
VIDEOS_DIR=./data/videos
SUMMARIES_DIR=./data/summaries
TEMP_DIR=./data/temp
```

**Note**: For offline operation, only the basic configuration is required. The system will automatically use local AI models.

## 🎮 Usage

### Running the Application

1. **Start the Streamlit application:**
```bash
streamlit run src/frontend/streamlit_app.py
```

2. **Open your web browser** and navigate to `http://localhost:8501`

### Processing Videos

#### Method 1: YouTube URL
1. Navigate to the "Process Video" page
2. Select "YouTube URL" as input method
3. Enter a YouTube video URL
4. Click "Preview Video Info" to validate
5. Click "Start Processing" to begin analysis

#### Method 2: File Upload
1. Navigate to the "Process Video" page
2. Select "Upload Video File" as input method
3. Choose a video file (max 500MB)
4. Click "Start Processing" to begin analysis

### Using the Chatbot

1. After processing a video, navigate to "Chat with Video"
2. Ask questions about the video content:
   - "What topics are covered in this video?"
   - "When is binary search discussed?"
   - "Show me the code examples"
   - "What is the time complexity mentioned?"
3. View timestamps and related content in responses

### Viewing Summaries

1. Navigate to "View Summary" page
2. Browse the comprehensive summary including:
   - Executive summary
   - Learning objectives
   - Detailed breakdown by sections
   - Code examples with explanations
   - Topic timeline
   - Next steps and recommendations

## 📁 Project Structure

```
dsa-video-summarizer/
├── src/
│   ├── video_processor/
│   │   ├── downloader.py          # YouTube video downloader
│   │   ├── extractor.py           # Audio/frame extraction
│   │   └── pipeline.py            # Main processing pipeline
│   ├── ai_engine/
│   │   ├── transcriber.py         # Speech-to-text using Whisper
│   │   ├── content_analyzer.py    # DSA content analysis
│   │   └── summarizer.py          # AI-powered summarization
│   ├── chatbot/
│   │   ├── vector_store.py        # ChromaDB vector storage
│   │   └── query_processor.py     # Query processing and responses
│   ├── frontend/
│   │   └── streamlit_app.py       # Streamlit web interface
│   └── utils/
│       ├── config.py              # Configuration management
│       └── helpers.py             # Utility functions
├── data/
│   ├── videos/                    # Downloaded videos
│   ├── summaries/                 # Generated summaries
│   ├── embeddings/                # Vector embeddings
│   └── temp/                      # Temporary files
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🔧 Configuration

### Video Processing Settings

- **MAX_VIDEO_SIZE_MB**: Maximum video file size (default: 500MB)
- **MAX_PROCESSING_TIME_MINUTES**: Maximum video duration (default: 30 minutes)
- **FRAME_EXTRACTION_INTERVAL**: Interval for frame extraction (default: 30 seconds)

### AI Model Settings

- **Whisper Model**: Configurable model size (tiny, base, small, medium, large)
- **Local LLM Models**: Ollama-based models (CodeLlama:7b recommended for CPU)
- **CPU Optimization**: Efficient inference without GPU acceleration
- **Fallback System**: Rule-based summarizer when LLM is unavailable
- **Embedding Model**: ChromaDB default embedding model

## 🎓 DSA Topics Supported

The system recognizes and analyzes the following DSA topics:

**Data Structures:**
- Arrays, Linked Lists, Stacks, Queues
- Trees (Binary, BST, AVL, Red-Black)
- Heaps, Hash Tables, Graphs
- Tries, Segment Trees, Union Find

**Algorithms:**
- Sorting (Bubble, Merge, Quick, Heap, Insertion)
- Searching (Linear, Binary, DFS, BFS)
- Graph Algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall)
- Dynamic Programming, Greedy Algorithms
- Backtracking, Two Pointers, Sliding Window

**Concepts:**
- Time and Space Complexity Analysis
- Big O Notation
- Algorithm Design Patterns
- Problem-Solving Techniques

## 🧪 Examples

### Sample Queries for Chatbot

```
"What algorithms are discussed in this video?"
"When is dynamic programming mentioned?"
"Show me the Python code examples"
"What is the time complexity of the merge sort implementation?"
"Explain the binary tree traversal methods from the video"
"At what timestamp is the space complexity discussed?"
```

### Sample Video Types

The system works best with:
- Algorithm explanation videos
- Data structure tutorials
- Coding interview preparation
- Computer science lectures
- Programming tutorials with DSA focus

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

### Adding New Features

1. **New DSA Topics**: Add to `config.DSA_TOPICS` list
2. **New Analysis Types**: Extend `ContentAnalyzer` class
3. **New Query Types**: Add patterns to `QueryProcessor.intent_patterns`
4. **New Export Formats**: Extend `VideoSummarizer` class

### Code Style

The project follows PEP 8 guidelines. Use the following tools:

```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy src/
```

## 📊 Performance

### Processing Times (Approximate)

| Video Duration | Processing Time | Components |
|----------------|-----------------|------------|
| 10 minutes     | 3-5 minutes     | Transcription: 60%, Analysis: 30%, Summary: 10% |
| 30 minutes     | 8-12 minutes    | Transcription: 65%, Analysis: 25%, Summary: 10% |
| 60 minutes     | 15-20 minutes   | Transcription: 70%, Analysis: 20%, Summary: 10% |

### Resource Usage

- **Memory**: 2-4GB RAM during processing
- **Storage**: ~100MB per video (temporary files)
- **Network**: YouTube download bandwidth + API calls

## 🚨 Limitations

- **Video Duration**: Maximum 30 minutes (configurable)
- **File Size**: Maximum 500MB (configurable)
- **Languages**: Currently optimized for English content
- **API Costs**: None when using local LLM models
- **Processing Time**: Dependent on video length and hardware
- **Local Models**: Requires sufficient RAM (4-8GB recommended for Ollama)
- **CPU Performance**: CodeLlama:7b optimized for CPU-only systems

## 🔒 Privacy & Security

- **Local Processing**: Video content processed locally
- **API Usage**: Only transcripts sent to OpenAI (no video/audio)
- **Data Storage**: All data stored locally in `data/` directory
- **No External Tracking**: No analytics or external data collection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech recognition
- **Ollama** for local LLM capabilities
- **ChromaDB** for vector storage and search
- **Streamlit** for the web interface
- **yt-dlp** for YouTube video downloading
- **OpenCV** for video frame processing

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Join discussions in existing issues
4. Contact the maintainers

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Support for multiple languages
- [ ] Advanced diagram recognition
- [ ] Interactive code execution
- [ ] Collaborative note-taking
- [ ] Mobile app interface

### Version 2.1 (Future)
- [ ] Real-time video processing
- [ ] Integration with learning platforms
- [ ] Advanced analytics dashboard
- [ ] Multi-user support
- [ ] API endpoints for external integration

---

**Happy Learning! 🎓**
