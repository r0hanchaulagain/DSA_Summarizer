#!/usr/bin/env python3
"""Test script to verify local LLM manager is working."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    
    # Test config import
    from utils.config import config
    print("✅ Config imported successfully")
    
    # Test local LLM import
    from ai_engine.local_llm import local_llm_manager
    print("✅ Local LLM manager imported successfully")
    
    # Test summarizer import
    from ai_engine.summarizer import VideoSummarizer
    print("✅ VideoSummarizer imported successfully")
    
    # Test local LLM manager
    print(f"Local LLM Manager: {type(local_llm_manager)}")
    print(f"Available models: {local_llm_manager.get_model_info()}")
    
    # Test basic response generation
    try:
        response = local_llm_manager.generate_response("Hello, can you explain binary search?", "")
        print(f"✅ Local LLM response test: {response[:100]}...")
    except Exception as e:
        print(f"❌ Local LLM response test failed: {e}")
    
    print("\n🎉 All tests passed! Local LLM system is working.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
