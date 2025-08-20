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
