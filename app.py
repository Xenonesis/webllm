from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from ai_agent import AIAgent
import os
import base64
import json
import requests

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode
agent = AIAgent()

class OpenAICompatibleManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {"openai_compatible_providers": {}}
    
    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_providers(self):
        return self.config.get("openai_compatible_providers", {})
    
    def chat_completion(self, provider, model, messages, **kwargs):
        providers = self.get_providers()
        if provider not in providers:
            raise ValueError(f"Provider {provider} not configured")
        
        provider_config = providers[provider]
        headers = {
            "Authorization": f"Bearer {provider_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(
            f"{provider_config['base_url']}/chat/completions",
            headers=headers,
            json=data
        )
        
        return response.json()

oai_manager = OpenAICompatibleManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/providers')
def providers():
    return render_template('provider_config.html')

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    available_models = []
    
    # Add original models
    if hasattr(agent, 'gemini_available') and agent.gemini_available:
        available_models.append({"id": "gemini", "name": "Gemini"})
    if hasattr(agent, 'openai_available') and agent.openai_available:
        available_models.append({"id": "openai", "name": "OpenAI"})
    if hasattr(agent, 'claude_available') and agent.claude_available:
        available_models.append({"id": "claude", "name": "Claude 3.5 Sonnet"})
    if hasattr(agent, 'mistral_available') and agent.mistral_available:
        available_models.append({"id": "mistral", "name": "Mistral AI"})
    if hasattr(agent, 'grok_available') and agent.grok_available:
        available_models.append({"id": "grok", "name": "Grok (xAI)"})
    
    # Add OpenAI-compatible providers
    providers = oai_manager.get_providers()
    for provider_name, provider_config in providers.items():
        for model in provider_config.get('models', []):
            available_models.append({
                "id": f"{provider_name}:{model}",
                "name": f"{provider_name.title()} - {model}",
                "provider": provider_name,
                "model": model,
                "type": "openai_compatible"
            })
    
    if not available_models:
        available_models.append({"id": "none", "name": "No APIs Configured"})
    
    return jsonify({
        'available_models': available_models,
        'current_model': agent.default_model,
        'openai_compatible_providers': list(providers.keys())
    })

@app.route('/api/current-model', methods=['GET'])
def get_current_model():
    return jsonify({'model': agent.default_model})

@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    data = request.get_json()
    model = data.get('model')
    
    # Support both original models and OpenAI-compatible models
    valid_models = ['gemini', 'openai', 'claude', 'mistral', 'grok']
    
    # Check if it's an OpenAI-compatible model (format: provider:model)
    if ':' in model or model in valid_models:
        agent.default_model = model
        return jsonify({'success': True, 'model': model})
    else:
        return jsonify({'success': False, 'error': 'Invalid model name'}), 400

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    mode = data.get('mode', 'text')
    model = data.get('model', agent.default_model)
    
    # Check if it's an OpenAI-compatible model
    if ':' in model:
        provider, model_name = model.split(':', 1)
        try:
            messages = [{"role": "user", "content": prompt}]
            response = oai_manager.chat_completion(provider, model_name, messages, max_tokens=1000)
            return jsonify({
                'response': response['choices'][0]['message']['content'],
                'model': model,
                'mode': 'text',
                'provider': provider
            })
        except Exception as e:
            return jsonify({
                'response': f"Error with {provider}: {str(e)}",
                'model': model,
                'mode': 'text',
                'error': True
            })
    
    # Original model handling
    if mode == 'image':
        response = agent.generate_response(prompt, model=model, mode=mode)
        if isinstance(response, dict) and 'image' in response:
            return jsonify({
                'response': response['image'],
                'type': response['type'],
                'model': model,
                'mode': 'image'
            })
        elif isinstance(response, dict) and 'description' in response:
            return jsonify({
                'response': response['description'],
                'error': response.get('error', 'Image generation not available'),
                'model': model,
                'mode': 'text',
                'isFallback': True
            })
        else:
            return jsonify({
                'response': "Error generating image",
                'error': response.get('error', 'Unknown error'),
                'model': model,
                'mode': 'text'
            })
    else:
        response = agent.generate_response(prompt, model=model)
        return jsonify({
            'response': response,
            'model': model,
            'mode': 'text'
        })

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_data = data['image']
    prompt = data.get('prompt', '')
    
    # Analyze image
    response = agent.analyze_image_gemini(image_data, prompt)
    
    return jsonify({
        'response': response,
        'model': 'gemini',
        'mode': 'text'
    })

@app.route('/api/stream')
def stream():
    prompt = request.args.get('prompt')
    if not prompt:
        return 'No prompt provided', 400
        
    def generate():
        for chunk in agent.generate_response_claude_stream(prompt):
            if chunk:
                yield f'data: {json.dumps({"type": "content", "text": chunk})}\n\n'
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/openai-providers', methods=['GET'])
def get_openai_providers():
    return jsonify(oai_manager.get_providers())

@app.route('/api/openai-providers', methods=['POST'])
def add_openai_provider():
    data = request.get_json()
    provider_name = data['name']
    provider_config = {
        'base_url': data['base_url'],
        'api_key': data['api_key'],
        'models': data['models']
    }
    
    oai_manager.config.setdefault('openai_compatible_providers', {})[provider_name] = provider_config
    oai_manager.save_config()
    
    return jsonify({'success': True, 'provider': provider_name})

@app.route('/api/openai-providers/<provider_name>', methods=['DELETE'])
def delete_openai_provider(provider_name):
    providers = oai_manager.config.get('openai_compatible_providers', {})
    if provider_name in providers:
        del providers[provider_name]
        oai_manager.save_config()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Provider not found'}), 404

@app.route('/api/test-provider', methods=['POST'])
def test_provider():
    data = request.get_json()
    provider_name = data['provider']
    
    try:
        response = oai_manager.chat_completion(
            provider_name, 
            data.get('model', 'gpt-3.5-turbo'),
            [{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)