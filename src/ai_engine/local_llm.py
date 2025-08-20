"""Local LLM interface for offline AI processing."""

import os
import logging
import json
import requests
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from utils.config import config
from utils.helpers import logger

logger = logging.getLogger(__name__)

class LocalLLMInterface(ABC):
    """Abstract interface for local LLM models."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response from local LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the local LLM is available."""
        pass

class GeminiLLM(LocalLLMInterface):
    """Gemini API interface for enhanced AI processing."""
    
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-1.5-flash"
        self.timeout = 60
        
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        if not self.api_key:
            return False
        
        # Check if the API key looks valid (basic format check)
        if len(self.api_key) < 10:
            return False
        
        return True
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Gemini API."""
        try:
            if not self.is_available():
                raise Exception("Gemini API key not configured")
            
            # Prepare the full prompt with context
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Use Google's Generative AI library if available, otherwise use direct API
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(full_prompt)
                return response.text.strip()
            except ImportError:
                # Fallback to direct API call
                return self._call_gemini_api_direct(full_prompt)
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            raise Exception(f"Gemini API error: {e}")
    
    def _call_gemini_api_direct(self, prompt: str) -> str:
        """Direct API call to Gemini as fallback."""
        try:
            url = f"{self.base_url}/{self.model}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.9,
                    "maxOutputTokens": 2048
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                content = result['candidates'][0].get('content', {})
                if 'parts' in content and content['parts']:
                    return content['parts'][0].get('text', '').strip()
            
            raise Exception("Invalid response format from Gemini API")
            
        except Exception as e:
            logger.error(f"Direct Gemini API call failed: {e}")
            raise Exception(f"Direct API call failed: {e}")

class OllamaLLM(LocalLLMInterface):
    """Ollama local LLM interface."""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"
        self.timeout = config.OLLAMA_TIMEOUT
        self.retry_attempts = config.OLLAMA_RETRY_ATTEMPTS
        
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                # Also check if the specific model is available
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in available models: {model_names}")
                    return False
                return True
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Ollama at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama API with retry logic and better error handling."""
        for attempt in range(self.retry_attempts + 1):
            try:
                if not self.is_available():
                    raise Exception(f"Ollama is not available at {self.base_url}. Please start Ollama service and ensure model {self.model_name} is loaded.")
                
                # Prepare the full prompt with context
                full_prompt = f"{context}\n\n{prompt}" if context else prompt
                
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                }
                
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                response_text = result.get("response", "").strip()
                
                if not response_text:
                    raise Exception("Ollama returned empty response")
                
                return response_text
                
            except requests.exceptions.Timeout:
                if attempt < self.retry_attempts:
                    continue
                else:
                    error_msg = f"Ollama request timed out after {self.timeout} seconds. The model might be overloaded or the request is too complex."
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except requests.exceptions.ConnectionError:
                if attempt < self.retry_attempts:
                    continue
                else:
                    error_msg = f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama service is running."
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                if attempt < self.retry_attempts:
                    continue
                else:
                    logger.error(f"Failed to generate response after {self.retry_attempts + 1} attempts: {e}")
                    raise Exception(f"Failed to generate response: {e}")
        
        # This should never be reached due to the exception in the loop
        raise Exception("Unexpected error in retry loop")
    
    def _get_available_models(self) -> str:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return ", ".join([model.get("name", "") for model in models])
            return "No models found"
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return "Error retrieving models"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of Ollama service."""
        try:
            # Check if service is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if self.model_name in model_names:
                    return {
                        'status': 'healthy',
                        'base_url': self.base_url,
                        'required_model': self.model_name,
                        'available_models': model_names
                    }
                else:
                    return {
                        'status': 'model_not_loaded',
                        'base_url': self.base_url,
                        'required_model': self.model_name,
                        'available_models': model_names,
                        'error': f'Model {self.model_name} not found in available models',
                        'suggestion': f'Run: ollama pull {self.model_name}'
                    }
            else:
                return {
                    'status': 'service_error',
                    'base_url': self.base_url,
                    'error': f'Service returned status code {response.status_code}',
                    'suggestion': 'Check if Ollama service is running properly'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'status': 'connection_error',
                'base_url': self.base_url,
                'error': 'Cannot connect to Ollama service',
                'suggestion': 'Run: ollama serve'
            }
        except requests.exceptions.Timeout:
            return {
                'status': 'timeout_error',
                'base_url': self.base_url,
                'error': 'Connection timeout',
                'suggestion': 'Check if Ollama service is responding'
            }
        except Exception as e:
            return {
                'status': 'unknown_error',
                'base_url': self.base_url,
                'error': str(e),
                'suggestion': 'Check Ollama service logs for more details'
            }

class RuleBasedSummarizer(LocalLLMInterface):
    """Rule-based summarizer as fallback when LLM is not available."""
    
    def __init__(self):
        self.dsa_keywords = config.DSA_TOPICS
        self.summary_templates = self._load_summary_templates()
    
    def _load_summary_templates(self) -> Dict[str, str]:
        """Load summary templates for different content types."""
        return {
            "executive_summary": "This video covers {topics} with a focus on {primary_topic}. Key algorithms discussed include {algorithms}. The content is suitable for {level} level students.",
            "learning_objectives": "By the end of this video, students should understand: {objectives}",
            "code_examples": "Code examples in {language} demonstrate: {examples}",
            "complexity_analysis": "Time complexity: {time_complexity}, Space complexity: {space_complexity}"
        }
    
    def is_available(self) -> bool:
        """Rule-based system is always available."""
        return True
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using rule-based logic."""
        try:
            # Extract key information from context
            extracted_info = self._extract_key_info(context)
            
            # Generate response based on prompt type
            if "summary" in prompt.lower():
                return self._generate_summary(extracted_info)
            elif "explain" in prompt.lower():
                return self._generate_explanation(prompt, extracted_info)
            elif "code" in prompt.lower():
                return self._generate_code_explanation(extracted_info)
            else:
                return self._generate_general_response(prompt, extracted_info)
                
        except Exception as e:
            logger.error(f"Error in rule-based response generation: {e}")
            return "I'm sorry, I couldn't process that request. Please try rephrasing your question."
    
    def _extract_key_info(self, context: str) -> Dict[str, Any]:
        """Extract key information from context text."""
        info = {
            "topics": [],
            "algorithms": [],
            "languages": [],
            "complexity": {}
        }
        
        # Extract DSA topics
        for topic in self.dsa_keywords:
            if topic.lower() in context.lower():
                info["topics"].append(topic)
        
        # Extract programming languages
        lang_patterns = {
            "python": ["def ", "import ", "print(", "if __name__"],
            "java": ["public class", "System.out.println", "ArrayList"],
            "cpp": ["#include", "using namespace", "cout <<", "vector<"],
            "javascript": ["function", "console.log", "const ", "let "]
        }
        
        for lang, patterns in lang_patterns.items():
            if any(pattern in context for pattern in patterns):
                info["languages"].append(lang)
        
        return info
    
    def _generate_summary(self, info: Dict[str, Any]) -> str:
        """Generate summary using extracted information."""
        topics = ", ".join(info["topics"][:3]) if info["topics"] else "various DSA concepts"
        primary_topic = info["topics"][0] if info["topics"] else "data structures and algorithms"
        algorithms = ", ".join(info["algorithms"][:2]) if info["algorithms"] else "fundamental algorithms"
        level = "intermediate" if len(info["topics"]) > 5 else "beginner"
        
        return self.summary_templates["executive_summary"].format(
            topics=topics,
            primary_topic=primary_topic,
            algorithms=algorithms,
            level=level
        )
    
    def _generate_explanation(self, prompt: str, info: Dict[str, Any]) -> str:
        """Generate explanation based on prompt and context."""
        if "algorithm" in prompt.lower():
            return f"This video explains {', '.join(info['topics'][:2])} algorithms. The implementation details and complexity analysis are covered with practical examples."
        elif "data structure" in prompt.lower():
            return f"The video covers {', '.join(info['topics'][:2])} data structures, including their implementation and usage patterns."
        else:
            return f"The video provides comprehensive coverage of {', '.join(info['topics'][:2])} concepts with practical examples and explanations."
    
    def _generate_code_explanation(self, info: Dict[str, Any]) -> str:
        """Generate code-related explanation."""
        languages = ", ".join(info["languages"]) if info["languages"] else "multiple programming languages"
        return f"Code examples are provided in {languages}. The implementations demonstrate practical application of the theoretical concepts discussed."
    
    def _generate_general_response(self, prompt: str, info: Dict[str, Any]) -> str:
        """Generate general response for other types of queries."""
        return f"Based on the video content, I can see discussions about {', '.join(info['topics'][:3])}. The material includes both theoretical explanations and practical implementations."

class LocalLLMManager:
    """Manager for local LLM models with fallback options."""
    
    def __init__(self):
        self.llm_models = []
        self.current_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available LLM models in order of preference."""
        logger.info("Initializing AI models...")
        
        # Try Gemini first (primary)
        if config.GEMINI_API_KEY:
            gemini = GeminiLLM()
            if gemini.is_available():
                self.llm_models.append(gemini)
                logger.info("Gemini LLM initialized as primary model")
            else:
                logger.warning("Gemini API key configured but model not available")
        else:
            logger.info("No Gemini API key found, skipping Gemini initialization")
        
        # Try Ollama second (fallback)
        ollama = OllamaLLM()
        if ollama.is_available():
            self.llm_models.append(ollama)
            logger.info("Ollama LLM initialized as fallback")
        else:
            logger.warning("Ollama not available")
        
        # Always add rule-based fallback
        rule_based = RuleBasedSummarizer()
        self.llm_models.append(rule_based)
        logger.info("Rule-based summarizer initialized as final fallback")
        
        # Set current model to the first available one
        self.current_model = self.llm_models[0] if self.llm_models else rule_based
        
        # Log final model configuration
        logger.info(f"Model configuration: {len(self.llm_models)} models, current: {type(self.current_model).__name__}")
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the best available model with fallback."""
        last_error = None
        
        # Get available models
        available_models = [model for model in self.llm_models if model.is_available()]
        
        # Try each available model in order until one succeeds
        for model in available_models:
            try:
                response = model.generate_response(prompt, context)
                self.current_model = model
                logger.info(f"Response generated with {type(model).__name__}")
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to generate response with {type(model).__name__}: {e}")
                continue
        
        # If all models failed, return error message
        error_msg = f"All LLM models failed. Last error: {last_error}"
        logger.error(error_msg)
        return "I'm sorry, I'm experiencing technical difficulties with all available AI models. Please try again later."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        available_models = [model for model in self.llm_models if model.is_available()]
        
        return {
            "current_model": type(self.current_model).__name__ if self.current_model else "None",
            "available_models": [type(model).__name__ for model in available_models],
            "total_models": len(self.llm_models),
            "total_available": len(available_models),
            "primary_model": "Gemini" if any(isinstance(model, GeminiLLM) for model in available_models) else "Ollama",
            "fallback_models": [type(model).__name__ for model in available_models if not isinstance(model, GeminiLLM)],
            "model_details": [
                {
                    "name": type(model).__name__,
                    "available": model.is_available(),
                    "type": "primary" if isinstance(model, GeminiLLM) else "fallback"
                }
                for model in self.llm_models
            ]
        }
    
    def debug_model_status(self) -> str:
        """Get detailed debug information about model status."""
        lines = []
        lines.append("🔍 AI Model Debug Information")
        lines.append("=" * 40)
        
        for i, model in enumerate(self.llm_models):
            model_name = type(model).__name__
            is_available = model.is_available()
            status = "✅ Available" if is_available else "❌ Not Available"
            priority = "Primary" if isinstance(model, GeminiLLM) else "Fallback"
            
            lines.append(f"{i+1}. {model_name} ({priority}): {status}")
            
            if hasattr(model, 'api_key') and model.api_key:
                lines.append(f"   API Key: ✅ Configured")
            elif hasattr(model, 'base_url'):
                lines.append(f"   Base URL: {model.base_url}")
        
        lines.append(f"\n🎯 Current Model: {type(self.current_model).__name__}")
        lines.append(f"📊 Total Models: {len(self.llm_models)}")
        lines.append(f"✅ Available Models: {len([m for m in self.llm_models if m.is_available()])}")
        
        return "\n".join(lines)

# Global instance
local_llm_manager = LocalLLMManager()
