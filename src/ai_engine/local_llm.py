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

class OllamaLLM(LocalLLMInterface):
    """Ollama local LLM interface."""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"
        
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama API."""
        try:
            if not self.is_available():
                raise Exception("Ollama is not available. Please start Ollama service.")
            
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
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise

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
        # Try Ollama first
        ollama = OllamaLLM()
        if ollama.is_available():
            self.llm_models.append(ollama)
            logger.info("Ollama LLM initialized successfully")
        
        # Always add rule-based fallback
        rule_based = RuleBasedSummarizer()
        self.llm_models.append(rule_based)
        logger.info("Rule-based summarizer initialized as fallback")
        
        # Set current model
        self.current_model = self.llm_models[0] if self.llm_models else rule_based
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the best available model."""
        try:
            # Try current model first
            if self.current_model and self.current_model.is_available():
                return self.current_model.generate_response(prompt, context)
            
            # Fallback to rule-based system
            for model in self.llm_models:
                if model.is_available():
                    self.current_model = model
                    return model.generate_response(prompt, context)
            
            raise Exception("No LLM models available")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Final fallback
            return "I'm sorry, I'm experiencing technical difficulties. Please try again later."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "current_model": type(self.current_model).__name__,
            "available_models": [type(model).__name__ for model in self.llm_models if model.is_available()],
            "total_models": len(self.llm_models)
        }

# Global instance
local_llm_manager = LocalLLMManager()
