#!/usr/bin/env python3
"""Demonstration of the suggestions button functionality in the chatbot."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.chatbot.query_processor import QueryProcessor

def demo_suggestions():
    """Demonstrate the suggestions functionality."""
    print("💡 Chatbot Suggestions Demo")
    print("=" * 50)
    
    try:
        # Initialize the chatbot
        print("🔧 Initializing chatbot...")
        chatbot = QueryProcessor()
        print("✅ Chatbot initialized successfully!")
        
        # Simulate a video ID (you would normally get this from processing a video)
        video_id = "demo_video_123"
        
        print(f"\n📹 Using video ID: {video_id}")
        
        # Get suggested questions
        print("\n🧠 Generating suggested questions...")
        try:
            suggestions = chatbot.get_suggested_questions(video_id)
            print(f"✅ Generated {len(suggestions)} suggestions:")
            
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
                
        except Exception as e:
            print(f"⚠️ Could not generate dynamic suggestions: {e}")
            print("   Using fallback suggestions...")
            
            # Fallback suggestions (same as in the frontend)
            suggestions = [
                "What topics are covered in this video?",
                "What algorithms are explained?",
                "Show me the code examples",
                "What is the time complexity discussed?",
                "When is [specific topic] mentioned?",
                "Explain the main concept of this video"
            ]
            
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print("\n🎯 How Suggestions Work:")
        print("1. User clicks a suggestion button")
        print("2. Button sets st.session_state.suggested_query = suggestion")
        print("3. Chat interface detects suggested_query and processes it")
        print("4. Suggestion is treated as if user typed it manually")
        print("5. Chatbot generates response using the selected AI model")
        print("6. Response is displayed in chat interface")
        
        print("\n🔄 AI Model Selection:")
        print("• Gemini API (primary) - if configured and available")
        print("• Ollama local LLM (fallback) - if Gemini fails")
        print("• Rule-based system (final fallback) - if all AI models fail")
        
        print("\n💡 Benefits of Dynamic Suggestions:")
        print("• Context-aware questions based on actual video content")
        print("• Personalized suggestions for each video")
        print("• Automatic generation based on detected topics/algorithms")
        print("• Fallback to general suggestions if dynamic generation fails")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_query_processing():
    """Demonstrate how a suggestion would be processed."""
    print("\n🔍 Query Processing Demo")
    print("=" * 50)
    
    try:
        chatbot = QueryProcessor()
        video_id = "demo_video_123"
        
        # Simulate a suggested query
        suggested_query = "What algorithms are explained in this video?"
        
        print(f"📝 Processing suggested query: '{suggested_query}'")
        print(f"📹 For video: {video_id}")
        
        print("\n⏳ Processing steps:")
        print("1. Classify query intent...")
        print("2. Search relevant content...")
        print("3. Generate AI response...")
        print("4. Format and return result...")
        
        print("\n🤖 AI Model Selection:")
        model_info = chatbot.llm_manager.get_model_info()
        print(f"• Primary Model: {model_info.get('primary_model', 'Unknown')}")
        print(f"• Current Model: {model_info.get('current_model', 'Unknown')}")
        print(f"• Available Models: {', '.join(model_info.get('available_models', []))}")
        
        print("\n💡 This demonstrates the complete flow:")
        print("   Suggestion Button → Query Processing → AI Response → User Display")
        
    except Exception as e:
        print(f"❌ Query processing demo failed: {e}")

if __name__ == "__main__":
    print("🚀 DSA Video Summarizer - Suggestions Demo")
    print("=" * 60)
    
    try:
        demo_suggestions()
        demo_query_processing()
        
        print("\n🎉 Demo completed!")
        print("\n📚 To test in the actual application:")
        print("1. Run the Streamlit app: streamlit run src/frontend/streamlit_app.py")
        print("2. Process a video or load an existing summary")
        print("3. Go to 'Chat with Video' page")
        print("4. Click any suggestion button to see it in action!")
        
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
