#!/usr/bin/env python3
"""Test script for the dual-model system (Gemini + Ollama fallback)."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.ai_engine.local_llm import local_llm_manager
from src.utils.config import config

def test_dual_model_system():
    """Test the dual-model system."""
    print("🧪 Testing Dual-Model System")
    print("=" * 50)
    
    # Check configuration
    print(f"🔑 Gemini API Key: {'✅ Configured' if config.GEMINI_API_KEY else '❌ Not configured'}")
    print(f"🤖 Ollama Base URL: {config.OLLAMA_BASE_URL}")
    print(f"📦 Ollama Model: {config.OLLAMA_MODEL}")
    
    print("\n📊 Model Information:")
    model_info = local_llm_manager.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\n🧠 Testing Response Generation:")
    
    # Test prompt
    test_prompt = "Explain what a binary search tree is in simple terms."
    test_context = "This is a video about data structures and algorithms."
    
    try:
        print(f"📝 Prompt: {test_prompt}")
        print(f"📚 Context: {test_context}")
        print("\n⏳ Generating response...")
        
        response = local_llm_manager.generate_response(test_prompt, test_context)
        
        print(f"\n✅ Response generated successfully!")
        print(f"🤖 Model used: {type(local_llm_manager.current_model).__name__}")
        print(f"📏 Response length: {len(response)} characters")
        print(f"\n📄 Response preview:")
        print("-" * 40)
        print(response[:300] + "..." if len(response) > 300 else response)
        print("-" * 40)
        
    except Exception as e:
        print(f"\n❌ Error generating response: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
    
    print("\n🔍 Testing Model Availability:")
    for i, model in enumerate(local_llm_manager.llm_models):
        model_name = type(model).__name__
        is_available = model.is_available()
        status = "✅ Available" if is_available else "❌ Not Available"
        print(f"  {i+1}. {model_name}: {status}")
        
        if hasattr(model, 'get_health_status'):
            try:
                health = model.get_health_status()
                if 'status' in health:
                    print(f"     Status: {health['status']}")
            except Exception:
                pass

def test_fallback_behavior():
    """Test fallback behavior when primary model fails."""
    print("\n🔄 Testing Fallback Behavior:")
    print("=" * 50)
    
    # Test with a complex prompt that might trigger fallback
    complex_prompt = """
    Please provide a detailed analysis of the time complexity of the following algorithms:
    1. Binary Search
    2. Merge Sort
    3. Quick Sort
    4. Dijkstra's Algorithm
    
    Include Big O notation, space complexity, and real-world examples.
    """
    
    try:
        print("📝 Testing with complex prompt...")
        response = local_llm_manager.generate_response(complex_prompt, "")
        
        print(f"✅ Complex prompt handled successfully!")
        print(f"🤖 Final model used: {type(local_llm_manager.current_model).__name__}")
        print(f"📏 Response length: {len(response)} characters")
        
    except Exception as e:
        print(f"❌ Complex prompt failed: {e}")

if __name__ == "__main__":
    print("🚀 DSA Video Summarizer - Dual-Model Test")
    print("=" * 60)
    
    try:
        test_dual_model_system()
        test_fallback_behavior()
        
        print("\n🎉 Test completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
