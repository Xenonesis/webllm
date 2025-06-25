from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from ai_agent import AIAgent
import os
import base64
import json

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode
agent = AIAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    available_models = []
    
    # Debug info about availability
    print("Model availability status:")
    print(f"- Gemini: {hasattr(agent, 'gemini_available')} -> {agent.gemini_available if hasattr(agent, 'gemini_available') else 'N/A'}")
    print(f"- OpenAI: {hasattr(agent, 'openai_available')} -> {agent.openai_available if hasattr(agent, 'openai_available') else 'N/A'}")
    print(f"- Claude: {hasattr(agent, 'claude_available')} -> {agent.claude_available if hasattr(agent, 'claude_available') else 'N/A'}")
    print(f"- Mistral: {hasattr(agent, 'mistral_available')} -> {agent.mistral_available if hasattr(agent, 'mistral_available') else 'N/A'}")
    print(f"- Grok: {hasattr(agent, 'grok_available')} -> {agent.grok_available if hasattr(agent, 'grok_available') else 'N/A'}")
    
    # Add models that are available based on API configuration
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
    
    # If no models are available, add a placeholder
    if not available_models:
        available_models.append({"id": "none", "name": "No APIs Configured"})
    
    # For debugging, always include all options during development
    if app.config['DEBUG']:
        # During development, show all options
        all_models = [
            {"id": "gemini", "name": "Gemini"},
            {"id": "openai", "name": "OpenAI"},
            {"id": "claude", "name": "Claude 3.5 Sonnet"},
            {"id": "mistral", "name": "Mistral AI"},
            {"id": "grok", "name": "Grok (xAI)"}
        ]
        
        # Add any that aren't already in available_models
        for model in all_models:
            if not any(m['id'] == model['id'] for m in available_models):
                model['disabled'] = True
                available_models.append(model)
    
    return jsonify({
        'available_models': available_models,
        'current_model': agent.default_model
    })

@app.route('/api/current-model', methods=['GET'])
def get_current_model():
    return jsonify({'model': agent.default_model})

@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    data = request.get_json()
    model = data.get('model')
    
    # Expanded list of valid models
    if model in ['gemini', 'openai', 'claude', 'mistral', 'grok']:
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
    
    # Generate response based on mode
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
            # Handle fallback to text description
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
        # Text generation
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

if __name__ == '__main__':
    app.run(debug=True)