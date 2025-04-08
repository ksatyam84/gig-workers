from flask import Flask, request, jsonify
import ollama
from gtts import gTTS
from playsound import playsound
import os

app = Flask(__name__)

@app.route('/analyze_food', methods=['POST'])
def analyze_food():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_path = os.path.join('/tmp', image.filename)
    image.save(image_path)
    
    res = ollama.chat(
        model='llava:7b',
        messages=[
            {
                'role': "user",
                'content': 'Tell me if this food is healthy or not give me response in short 1 sentence of total calories the image contain and what nutrients it contains',
                'images': [image_path]
            }
        ]
    )

    response_text = res['message']['content']

    # Convert the response to audio
    tts = gTTS(response_text)
    audio_file = "response.mp3"
    tts.save(audio_file)

    # Play the audio
    playsound(audio_file)

    # Clean up the audio file
    os.remove(audio_file)

    return jsonify({'response': response_text}), 200

if __name__ == '__main__':
    app.run(debug=True)