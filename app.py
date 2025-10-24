from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import json
import time

app = Flask(__name__)
CORS(app)

# Hugging Face configuration
HF_API_KEY = os.environ.get('HF_API_KEY', '')
HF_BASE_URL = 'https://api-inference.huggingface.co/models'

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/v1', methods=['GET', 'POST', 'OPTIONS'])
def v1_root():
    """V1 API root endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    if request.method == 'POST':
        return chat_completions()
    
    return jsonify({
        'message': 'Hugging Face OpenAI-compatible API',
        'endpoints': {
            '/v1/chat/completions': 'POST - Chat completions',
            '/v1/models': 'GET - List models'
        }
    })

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        
        # Extract parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta-llama/Llama-3.1-70B-Instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        
        # Convert OpenAI messages to Hugging Face prompt
        prompt = convert_messages_to_prompt(messages)
        
        # Prepare Hugging Face request
        hf_payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        if 'top_p' in data:
            hf_payload['parameters']['top_p'] = data['top_p']
        
        headers = {
            'Authorization': f'Bearer {HF_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Make request to Hugging Face
        model_url = f'{HF_BASE_URL}/{model}'
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    model_url,
                    headers=headers,
                    json=hf_payload,
                    timeout=180
                )
                
                if response.status_code == 200:
                    # Convert Hugging Face response to OpenAI format
                    hf_response = response.json()
                    
                    # Handle different response formats
                    if isinstance(hf_response, list) and len(hf_response) > 0:
                        generated_text = hf_response[0].get('generated_text', '')
                    elif isinstance(hf_response, dict):
                        generated_text = hf_response.get('generated_text', '')
                    else:
                        generated_text = str(hf_response)
                    
                    # Return in OpenAI format
                    openai_response = {
                        "id": f"hf-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": generated_text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(generated_text.split()),
                            "total_tokens": len(prompt.split()) + len(generated_text.split())
                        }
                    }
                    
                    return jsonify(openai_response)
                    
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    error_data = response.json()
                    estimated_time = error_data.get('estimated_time', 20)
                    if attempt < max_retries - 1:
                        time.sleep(min(estimated_time, 30))
                        continue
                    return jsonify({
                        'error': {
                            'message': f'Model is loading. Please wait {estimated_time} seconds and try again.',
                            'type': 'model_loading',
                            'estimated_time': estimated_time
                        }
                    }), 503
                else:
                    return jsonify({
                        'error': {
                            'message': f'Hugging Face API error: {response.text}',
                            'type': 'hf_api_error',
                            'code': response.status_code
                        }
                    }), response.status_code
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue
                return jsonify({
                    'error': {
                        'message': 'Request timeout. Try a smaller model or reduce max_tokens.',
                        'type': 'timeout_error'
                    }
                }), 504
                
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'proxy_error'
            }
        }), 500

def convert_messages_to_prompt(messages):
    """Convert OpenAI chat messages to a single prompt string"""
    prompt = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            prompt += f"System: {content}\n\n"
        elif role == 'user':
            prompt += f"User: {content}\n\n"
        elif role == 'assistant':
            prompt += f"Assistant: {content}\n\n"
    
    prompt += "Assistant: "
    return prompt

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """List popular models"""
    if request.method == 'OPTIONS':
        return '', 204
        
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': 'meta-llama/Llama-3.1-70B-Instruct',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'huggingface'
            },
            {
                'id': 'mistralai/Mistral-7B-Instruct-v0.3',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'huggingface'
            },
            {
                'id': 'NousResearch/Hermes-3-Llama-3.1-70B',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'huggingface'
            },
            {
                'id': 'Qwen/Qwen2.5-72B-Instruct',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'huggingface'
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'huggingface-proxy'})

@app.route('/test-hf', methods=['GET'])
def test_hf():
    """Test Hugging Face API connection"""
    try:
        headers = {
            'Authorization': f'Bearer {HF_API_KEY}'
        }
        response = requests.get(
            'https://huggingface.co/api/whoami-v2',
            headers=headers,
            timeout=10
        )
        return jsonify({
            'hf_api_status': 'valid' if response.status_code == 200 else 'invalid',
            'status_code': response.status_code
        })
    except Exception as e:
        return jsonify({
            'hf_api_status': 'error',
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'Hugging Face to OpenAI API Proxy',
        'endpoints': {
            '/v1/chat/completions': 'POST - Chat completions',
            '/v1/models': 'GET - List models',
            '/health': 'GET - Health check',
            '/test-hf': 'GET - Test HF API'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
