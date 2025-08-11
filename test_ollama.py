#!/usr/bin/env python3
"""
Simple script to test Ollama connectivity and help troubleshoot setup issues.
Run this script to check if Ollama is working correctly.
"""

import requests
import json
import sys
import time

def test_ollama_connection():
    """Test Ollama connection and provide detailed feedback."""
    base_url = "http://localhost:11434"
    model_name = "codellama:7b"
    
    print("🧪 Testing Ollama Connection...")
    print(f"📍 URL: {base_url}")
    print(f"🤖 Model: {model_name}")
    print("-" * 50)
    
    # Test 1: Basic connectivity
    print("1️⃣ Testing basic connectivity...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama service is running")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama service")
        print("💡 Solution: Run 'ollama serve' to start the service")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout")
        print("💡 Solution: Check if Ollama is responding")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 2: Check available models
    print("\n2️⃣ Checking available models...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        print(f"📋 Available models: {', '.join(model_names)}")
        
        if model_name in model_names:
            print(f"✅ Required model '{model_name}' is available")
        else:
            print(f"❌ Required model '{model_name}' is not available")
            print(f"💡 Solution: Run 'ollama pull {model_name}'")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False
    
    # Test 3: Test model generation
    print("\n3️⃣ Testing model generation...")
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello! Please respond with 'Ollama is working correctly.'",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 100
            }
        }
        
        print("🔄 Generating test response...")
        start_time = time.time()
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            generation_time = end_time - start_time
            
            print(f"✅ Generation successful in {generation_time:.1f} seconds")
            print(f"📝 Response: {response_text}")
            
            if "ollama is working correctly" in response_text.lower():
                print("✅ Test response matches expected output")
            else:
                print("⚠️ Test response doesn't match expected output, but generation works")
                
        else:
            print(f"❌ Generation failed: HTTP {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out after 2 minutes")
        print("💡 Solution: The model might be overloaded or the request is too complex")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False
    
    # All tests passed
    print("\n🎉 All tests passed! Ollama is working correctly.")
    return True

def main():
    """Main function to run the test."""
    print("=" * 60)
    print("🤖 OLLAMA CONNECTION TESTER")
    print("=" * 60)
    
    success = test_ollama_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ OLLAMA SETUP IS WORKING CORRECTLY")
        print("💡 You can now use the DSA Video Summarizer with local AI!")
    else:
        print("❌ OLLAMA SETUP HAS ISSUES")
        print("💡 Please fix the issues above and run this script again")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
