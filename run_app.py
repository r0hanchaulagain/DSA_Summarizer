#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def main():
        
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Check if virtual environment exists
    venv_path = project_root / "thesis_env"
    if not venv_path.exists():
        print("Virtual environment not found!")
        print("Please run the following commands first:")
        print("python -m venv thesis_env")
        print("source thesis_env/bin/activate  # On Windows: thesis_env\\Scripts\\activate")
        print("pip install -r requirements.txt")
        return 1
    
    # Check if .env file exists (optional now)
    env_file = project_root / ".env"
    if not env_file.exists():
        print(".env file not found - creating with default settings")
        print("You can optionally add GEMINI_API_KEY for enhanced features")
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("# Optional: Add Gemini API key for enhanced features\n")
            f.write("# GEMINI_API_KEY=your_key_here\n")
            f.write("DEBUG=False\n")
            f.write("LOG_LEVEL=INFO\n")
        print("Created .env file with default settings")
    
    # Check if data directories exist
    data_dir = project_root / "data"
    if not data_dir.exists():
        print("Creating data directories...")
        for subdir in ["videos", "summaries", "embeddings", "temp"]:
            (data_dir / subdir).mkdir(parents=True, exist_ok=True)
        print("Data directories created")
    
    # Check if requirements are installed
    try:
        import streamlit
        import whisper
        import chromadb
        print("Required packages found")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return 1
    
    # Run the Streamlit application
    print("Starting DSA Video Summarizer...")
    print("Open your browser and go to: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        # Change to project directory
        os.chdir(project_root)
        
        # Add src directory to Python path
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        src_path = str(project_root / "src")
        if current_pythonpath:
            env['PYTHONPATH'] = f"{src_path}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = src_path
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/frontend/streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\nApplication stopped")
        return 0
    except Exception as e:
        print(f"Error running application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())