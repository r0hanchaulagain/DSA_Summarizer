#!/usr/bin/env python3
"""Debug script to troubleshoot Gemini model initialization."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import config
from src.ai_engine.local_llm import GeminiLLM, LocalLLMManager

def debug_gemini_config():
    """Debug Gemini configuration and availability."""
    print("🔍 Debugging Gemini Configuration")
    print("=" * 50)
    
    # Check environment variables
    print("🔑 Environment Variables:")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"✅ GEMINI_API_KEY found: {gemini_key[:10]}...{gemini_key[-4:]}")
        print(f"   Length: {len(gemini_key)} characters")
    else:
        print("❌ GEMINI_API_KEY not found in environment")
    
    # Check config
    print(f"\n⚙️ Config.GEMINI_API_KEY: {'✅ Set' if config.GEMINI_API_KEY else '❌ Not set'}")
    if config.GEMINI_API_KEY:
        print(f"   Value: {config.GEMINI_API_KEY[:10]}...{config.GEMINI_API_KEY[-4:]}")
    
    # Test GeminiLLM directly
    print("\n🧪 Testing GeminiLLM directly:")
    try:
        gemini = GeminiLLM()
        print(f"✅ GeminiLLM instance created")
        
        is_available = gemini.is_available()
        print(f"🔍 is_available(): {is_available}")
        
        if is_available:
            print("✅ Gemini appears to be available")
        else:
            print("❌ Gemini is not available")
            
    except Exception as e:
        print(f"❌ Error creating GeminiLLM: {e}")
    
    # Test LocalLLMManager
    print("\n🏗️ Testing LocalLLMManager:")
    try:
        manager = LocalLLMManager()
        print(f"✅ LocalLLMManager created")
        
        model_info = manager.get_model_info()
        print(f"📊 Model Info: {model_info}")
        
        # Use the new debug method
        print(f"\n{manager.debug_model_status()}")
        
        # Test response generation
        print(f"\n🧠 Testing Response Generation:")
        test_prompt = "Explain binary search in one sentence."
        try:
            response = manager.generate_response(test_prompt, "")
            print(f"✅ Response generated successfully!")
            print(f"🤖 Model used: {type(manager.current_model).__name__}")
            print(f"📝 Response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            
    except Exception as e:
        print(f"❌ Error creating LocalLLMManager: {e}")
        import traceback
        traceback.print_exc()

def check_environment():
    """Check the current environment setup."""
    print("\n🌍 Environment Check:")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    venv = os.getenv("VIRTUAL_ENV")
    if venv:
        print(f"✅ Virtual environment: {venv}")
    else:
        print("ℹ️ No virtual environment detected")
    
    # Check Python path
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found: {env_file}")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if "GEMINI_API_KEY" in content:
                    print("✅ GEMINI_API_KEY found in .env file")
                else:
                    print("❌ GEMINI_API_KEY not found in .env file")
        except Exception as e:
            print(f"⚠️ Error reading .env file: {e}")
    else:
        print("ℹ️ No .env file found")

def suggest_fixes():
    """Suggest fixes for common issues."""
    print("\n🔧 Suggested Fixes:")
    print("=" * 50)
    
    if not config.GEMINI_API_KEY:
        print("1. 🔑 Set GEMINI_API_KEY environment variable:")
        print("   export GEMINI_API_KEY='your_actual_api_key_here'")
        print("   ")
        print("2. 📁 Or create a .env file in the project root:")
        print("   echo 'GEMINI_API_KEY=your_actual_api_key_here' > .env")
        print("   ")
        print("3. 🚀 Restart the application after setting the key")
    else:
        print("✅ GEMINI_API_KEY is configured")
        print("   If Gemini is still not working, check:")
        print("   - API key validity")
        print("   - Network connectivity")
        print("   - API rate limits")

if __name__ == "__main__":
    print("🚀 DSA Video Summarizer - Gemini Debug Tool")
    print("=" * 60)
    
    try:
        debug_gemini_config()
        check_environment()
        suggest_fixes()
        
        print("\n🎉 Debug completed!")
        
    except Exception as e:
        print(f"\n💥 Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
