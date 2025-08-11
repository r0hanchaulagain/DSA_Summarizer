#!/usr/bin/env python3
"""Test script to verify Gemini priority in the dual-model system."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.ai_engine.local_llm import local_llm_manager

def test_gemini_priority():
    """Test that Gemini is properly prioritized when available."""
    print("🧪 Testing Gemini Priority in Dual-Model System")
    print("=" * 60)
    
    try:
        # Get model information
        model_info = local_llm_manager.get_model_info()
        
        print("📊 Current Model Configuration:")
        for key, value in model_info.items():
            if key != "model_details":  # Skip the detailed list for cleaner output
                print(f"  {key}: {value}")
        
        print(f"\n🔍 Detailed Model Status:")
        print(local_llm_manager.debug_model_status())
        
        # Test response generation
        print(f"\n🧠 Testing Response Generation Priority:")
        test_prompt = "Explain what a binary search tree is in simple terms."
        
        print(f"📝 Prompt: {test_prompt}")
        print("⏳ Generating response...")
        
        response = local_llm_manager.generate_response(test_prompt, "")
        
        print(f"✅ Response generated successfully!")
        print(f"🤖 Model used: {type(local_llm_manager.current_model).__name__}")
        
        # Check if the right model was used
        if isinstance(local_llm_manager.current_model, local_llm_manager.llm_models[0].__class__):
            print("🎯 ✅ First priority model was used correctly!")
        else:
            print("⚠️ ⚠️ First priority model was NOT used - check fallback logic")
        
        print(f"📏 Response length: {len(response)} characters")
        print(f"📄 Response preview: {response[:200]}...")
        
        # Test multiple queries to see if priority is maintained
        print(f"\n🔄 Testing Multiple Queries for Priority Consistency:")
        queries = [
            "What is time complexity?",
            "Explain recursion briefly",
            "What are data structures?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n  Query {i}: {query}")
            try:
                response = local_llm_manager.generate_response(query, "")
                model_used = type(local_llm_manager.current_model).__name__
                print(f"    ✅ Response generated with {model_used}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        print(f"\n🎯 Final Model Status:")
        print(f"  Current Model: {type(local_llm_manager.current_model).__name__}")
        print(f"  Model Priority: {[type(m).__name__ for m in local_llm_manager.llm_models]}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def check_environment_variables():
    """Check environment variables that affect model selection."""
    print("\n🌍 Environment Variables Check:")
    print("=" * 40)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}...{gemini_key[-4:]}")
        print(f"   Length: {len(gemini_key)} characters")
    else:
        print("❌ GEMINI_API_KEY: Not set")
        print("   💡 Set this to enable Gemini as primary model")
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"🌐 OLLAMA_BASE_URL: {ollama_url}")
    
    ollama_model = os.getenv("OLLAMA_MODEL", "codellama:7b")
    print(f"🤖 OLLAMA_MODEL: {ollama_model}")

if __name__ == "__main__":
    print("🚀 DSA Video Summarizer - Gemini Priority Test")
    print("=" * 70)
    
    try:
        check_environment_variables()
        test_gemini_priority()
        
        print("\n🎉 Priority test completed!")
        print("\n💡 Expected Behavior:")
        print("   • With Gemini API key: Gemini should be used first")
        print("   • Without Gemini API key: Ollama should be used first")
        print("   • Fallback should work automatically if primary model fails")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
