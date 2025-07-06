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
- OpenAI API key
- FFmpeg (for video processing)
- Tesseract OCR (for text extraction from frames)

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

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

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
- **OpenAI Model**: Uses GPT-3.5-turbo (can be upgraded to GPT-4)
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
- **API Costs**: OpenAI API usage for summarization and chat
- **Processing Time**: Dependent on video length and hardware

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

- **OpenAI** for Whisper and GPT models
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
