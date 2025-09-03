from flask import Flask, render_template, request, jsonify, session
from adaptive_rag import integrate_adaptive_rag
import os
import re
import uuid
from gtts import gTTS
from dotenv import load_dotenv
from typing import Optional  # Add this import

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(16)

# Initialize the adaptive RAG agent
adaptive_agent = integrate_adaptive_rag()

# ============== AUDIO PROCESSING ==============
class AudioProcessor:
    def __init__(self):
        pass

    def get_language_code_tts(self, language: str) -> str:
        lang_map = {
            'english': 'en', 'hindi': 'hi', 'marathi': 'mr', 'tamil': 'ta',
            'bengali': 'bn', 'gujarati': 'gu', 'kannada': 'kn', 'malayalam': 'ml',
            'punjabi': 'pa', 'telugu': 'te', 'urdu': 'ur', 'odia': 'or',
            'assamese': 'as', 'nepali': 'ne', 'sindhi': 'sd', 'kashmiri': 'ks',
            'sanskrit': 'sa', 'maithili': 'mai', 'dogri': 'doi', 'manipuri': 'mni',
            'bodo': 'brx', 'santhali': 'sat', 'konkani': 'gom'
        }
        return lang_map.get(language.lower(), 'en')

    def text_to_speech(self, text: str, language: str = 'english') -> Optional[str]:  # Fixed type hint
        try:
            lang_code = self.get_language_code_tts(language)
            text = re.sub(r"\*", "", text)
            tts = gTTS(text=text, lang=lang_code, slow=False)

            filename = f"audio_{uuid.uuid4().hex}.mp3"
            filepath = os.path.join('static', 'audio', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            tts.save(filepath)
            return filename
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

audio_processor = AudioProcessor()

# ============== FLASK ROUTES ==============

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('message', '').strip()
    language = data.get('language', 'english')

    if not query:
        return jsonify({'error': 'No message provided'}), 400

    try:
        if 'chat_history' not in session:
            session['chat_history'] = []

        session['chat_history'].append({"role": "user", "content": query})

        # Process query through adaptive rag agent
        answer = adaptive_agent(query, language)

        session['chat_history'].append({"role": "assistant", "content": answer})
        session.modified = True

        return jsonify({'message': answer, 'success': True})
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'message': "Sorry, an error occurred.", 'success': False})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'english')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        audio_filename = audio_processor.text_to_speech(text, language)
        if audio_filename:
            return jsonify({'audio_url': f'/static/audio/{audio_filename}', 'success': True})
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({'error': 'TTS processing failed'}), 500

@app.route('/clear')
def clear_history():
    session['chat_history'] = []
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
