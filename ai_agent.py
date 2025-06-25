import os
import json
import base64
import openai
import requests
import colorama
from colorama import Fore, Style, Back

colorama.init()

class AIAgent:
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.setup_apis()
        self.default_model = self.config.get("default_model", "gemini")
        
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Check if the config contains the old key
                if "gemini_api_key" in config:
                    print("Old config format detected. Overwriting with default config.")
                    raise FileNotFoundError  # Force overwrite
                return config
        except FileNotFoundError:
            print(f"Config file not found or needs update. Creating default config.")
            default_config = {
                "openai_api_key": "your_openai_api_key_here",
                "gemini_api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyAql2s4WHraUgh-xMtEa2gkEnDm4mgmTXQ",
                "claude_api_key": "your_claude_api_key_here",
                "mistral_api_key": "your_mistral_api_key_here",
                "grok_api_key": "your_grok_api_key_here",
                "default_model": "gemini"
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def setup_apis(self):
        # Setup OpenAI
        openai_key = self.config.get("openai_api_key")
        if (openai_key and openai_key != "your_openai_api_key_here"):
            self.openai_client = openai.OpenAI(api_key=openai_key)
            self.openai_available = True
            print("OpenAI API key configured.")
        else:
            print("OpenAI API key not configured.")
            self.openai_available = False
            
        # Setup Gemini
        self.gemini_api_url = self.config.get("gemini_api_url")
        if self.gemini_api_url and "key=" in self.gemini_api_url:
            print("Gemini API configured with direct endpoint URL")
            self.gemini_available = True
        else:
            print("Gemini API URL not properly configured.")
            self.gemini_available = False
            
        # Setup Claude API (Anthropic)
        claude_key = self.config.get("claude_api_key")
        if claude_key and claude_key != "your_claude_api_key_here":
            self.claude_api_key = claude_key
            self.claude_available = True
            print("Claude API key configured.")
        else:
            print("Claude API key not configured.")
            self.claude_available = False
            
        # Setup Mistral AI
        mistral_key = self.config.get("mistral_api_key")
        if mistral_key and mistral_key != "your_mistral_api_key_here":
            self.mistral_api_key = mistral_key
            self.mistral_available = True
            print("Mistral AI API key configured.")
        else:
            print("Mistral AI API key not configured.")
            self.mistral_available = False
            
        # Setup Grok (xAI)
        grok_key = self.config.get("grok_api_key")
        if grok_key and grok_key != "your_grok_api_key_here":
            self.grok_api_key = grok_key
            self.grok_available = True
            print("Grok API key configured.")
        else:
            print("Grok API key not configured.")
            self.grok_available = False
    
    def generate_response_openai(self, prompt):
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return f"Error: {e}"
    
    def generate_response_gemini(self, prompt):
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            print("Sending request to Gemini API...")
            response = requests.post(self.gemini_api_url, json=data, headers=headers)
            
            if response.status_code == 200:
                response_json = response.json()
                # Extract the text from the response
                try:
                    generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                    return generated_text
                except (KeyError, IndexError) as e:
                    print(f"Error parsing Gemini API response: {e}")
                    return f"Error parsing response: {e}. Response structure: {response_json}"
            else:
                error_message = f"Gemini API returned status code {response.status_code}: {response.text}"
                print(error_message)
                return error_message
                
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return f"Error: {e}"
    
    def generate_image_gemini(self, prompt):
        """Generate an image using Google's API or fall back to OpenAI if available"""
        try:
            # Extract base URL and API key from the gemini_api_url
            if not hasattr(self, 'gemini_api_url') or not self.gemini_api_url:
                return {"error": "Gemini API URL not configured", "type": "error"}
            
            # Parse the URL to extract API key
            url_parts = self.gemini_api_url.split('?')
            api_key = url_parts[1].replace('key=', '')
            
            # Try creating text-to-image using Google's Imagen service
            print("Google Imagen service is not yet publicly available. Checking for fallback options...")
            
            # Check if OpenAI is available as a fallback for image generation
            if hasattr(self, 'openai_available') and self.openai_available:
                print("Using OpenAI DALL-E for image generation instead.")
                return self.generate_image_openai(prompt)
            
            # If no image generation service is available, fall back to text description
            text_response = self.generate_response_gemini(f"Describe in vivid detail an image of: {prompt}")
            
            return {
                "error": "Image generation is not available with current API configurations. Generated a description instead.",
                "description": text_response,
                "type": "text"
            }
                
        except Exception as e:
            print(f"Error with image generation: {e}")
            return {"error": f"Error: {e}", "type": "error"}
    
    def generate_image_openai(self, prompt):
        """Generate an image using OpenAI's DALL-E"""
        try:
            print("Sending image generation request to OpenAI DALL-E...")
            response = self.openai_client.images.generate(
                model="dall-e-2",  # or "dall-e-3" for higher quality but slower generation
                prompt=prompt,
                size="512x512",
                response_format="b64_json",
                n=1
            )
            
            # Extract the base64 image data
            image_data = response.data[0].b64_json
            return {"image": image_data, "type": "image/png"}
        except Exception as e:
            print(f"Error with OpenAI image generation: {e}")
            return {"error": f"Error with OpenAI image generation: {e}", "type": "error"}
    
    def analyze_image_gemini(self, image_data, prompt):
        """Analyze an image using Gemini's multimodal capabilities"""
        try:
            # Extract base URL and API key from the gemini_api_url
            if not hasattr(self, 'gemini_api_url') or not self.gemini_api_url:
                return "Gemini API URL not configured"
            
            # Parse the URL to extract key and base URL
            url_parts = self.gemini_api_url.split('?')
            base_url = url_parts[0].replace('models/gemini-2.0-flash:generateContent',
                                          'models/gemini-1.5-flash:generateContent')
            api_key = url_parts[1].replace('key=', '')
            
            vision_url = f"{base_url}?key={api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Ensure image_data is properly formatted
            if not image_data.startswith('data:image'):
                image_data = f"data:image/jpeg;base64,{image_data}"
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt or "Describe this image in detail"},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data.split(',')[1] if ',' in image_data else image_data
                                }
                            }
                        ]
                    }
                ]
            }
            
            print("Sending image analysis request to Gemini API...")
            response = requests.post(vision_url, json=data, headers=headers)
            
            if response.status_code == 200:
                response_json = response.json()
                # Extract the text from the response
                try:
                    text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                    return text
                except (KeyError, IndexError) as e:
                    print(f"Error parsing Gemini vision response: {e}")
                    return f"Error parsing response: {e}"
            else:
                error_message = f"Gemini API returned status code {response.status_code}: {response.text}"
                print(error_message)
                return error_message
                
        except Exception as e:
            print(f"Error with Gemini image analysis: {e}")
            return f"Error: {e}"
    
    def generate_response_claude(self, prompt):
        try:
            headers = {
                "anthropic-version": "2023-06-01",
                "x-api-key": self.claude_api_key,
                "Content-Type": "application/json"
            }
            
            # Updated to use Claude 3.5 Sonnet with streaming
            data = {
                "model": "claude-3-5-sonnet",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
            
            print("Sending request to Claude 3.5 Sonnet API...")
            response = requests.post("https://api.anthropic.com/v1/messages", 
                                    json=data, 
                                    headers=headers,
                                    stream=True)
            
            if response.status_code == 200:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get("type") == "content_block_delta":
                                text = chunk.get("delta", {}).get("text", "")
                                full_response += text
                        except json.JSONDecodeError:
                            continue
                return full_response
            else:
                error_message = f"Claude API returned status code {response.status_code}: {response.text}"
                print(error_message)
                return error_message
                
        except Exception as e:
            print(f"Error with Claude 3.5 API: {e}")
            return f"Error: {e}"
    
    def generate_response_claude_stream(self, prompt):
        try:
            headers = {
                "anthropic-version": "2023-06-01",
                "x-api-key": self.claude_api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "claude-3-5-sonnet",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=data,
                headers=headers,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get("type") == "content_block_delta":
                                yield chunk.get("delta", {}).get("text", "")
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code} - {response.text}"
                    
        except Exception as e:
            yield f"Error: {str(e)}"
            
    def generate_response_mistral(self, prompt):
        try:
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-medium",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300
            }
            
            print("Sending request to Mistral AI API...")
            response = requests.post("https://api.mistral.ai/v1/chat/completions", 
                                     json=data, 
                                     headers=headers)
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response from Mistral")
            else:
                error_message = f"Mistral AI API returned status code {response.status_code}: {response.text}"
                print(error_message)
                return error_message
                
        except Exception as e:
            print(f"Error with Mistral AI API: {e}")
            return f"Error: {e}"
    
    def generate_response_grok(self, prompt):
        try:
            return "Grok API is not publicly available yet. This is a placeholder for future implementation."
        except Exception as e:
            print(f"Error with Grok API: {e}")
            return f"Error: {e}"
    
    def generate_response(self, prompt, model=None, mode="text"):
        """Generate a response based on the prompt and model"""
        model = model or self.default_model
        
        # Handle different modes
        if mode == "image":
            if model == "gemini":
                return self.generate_image_gemini(prompt)
            elif model == "openai" and self.openai_available:
                return self.generate_image_openai(prompt)
            else:
                # Try any available model for image generation
                if self.openai_available:
                    return self.generate_image_openai(prompt)
                else:
                    return self.generate_image_gemini(prompt)  # This will handle fallback
        
        # Print API availability status for debugging
        print(f"API availability - OpenAI: {self.openai_available}, Gemini: {self.gemini_available}, Claude: {self.claude_available}, Mistral: {self.mistral_available}, Grok: {self.grok_available}")
        
        # Text mode (default)
        if model.lower() == "openai" and self.openai_available:
            return self.generate_response_openai(prompt)
        elif model.lower() == "gemini" and self.gemini_available:
            return self.generate_response_gemini(prompt)
        elif model.lower() == "claude" and self.claude_available:
            return self.generate_response_claude(prompt)
        elif model.lower() == "mistral" and self.mistral_available:
            return self.generate_response_mistral(prompt)
        elif model.lower() == "grok" and self.grok_available:
            return self.generate_response_grok(prompt)
        else:
            # Try to use any available model
            if self.gemini_available:
                print("Falling back to Gemini API")
                return self.generate_response_gemini(prompt)
            elif self.openai_available:
                print("Falling back to OpenAI API")
                return self.generate_response_openai(prompt)
            elif self.claude_available:
                print("Falling back to Claude API")
                return self.generate_response_claude(prompt)
            elif self.mistral_available:
                print("Falling back to Mistral AI API")
                return self.generate_response_mistral(prompt)
            elif self.grok_available:
                print("Falling back to Grok API")
                return self.generate_response_grok(prompt)
            else:
                return "No API is properly configured. Please set up API keys in the config.json file."
