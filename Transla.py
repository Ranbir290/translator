from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
CORS(app, resources={r"/translate": {"origins": "*"}})  # Allow all origins

# Define available language models
LANGUAGE_MODELS = {
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-tl": "Helsinki-NLP/opus-mt-en-tl",  
    "tl-en": "Helsinki-NLP/opus-mt-tl-en",
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
}

# Function to translate text
def translate(text, source_lang, target_lang):
    lang_pair = f"{source_lang}-{target_lang}"

    # Direct translation available
    if lang_pair in LANGUAGE_MODELS:
        model_name = LANGUAGE_MODELS[lang_pair]
    elif f"{source_lang}-en" in LANGUAGE_MODELS and f"en-{target_lang}" in LANGUAGE_MODELS:
        # Use English as an intermediate translation step
        intermediate_text = translate(text, source_lang, "en")
        return translate(intermediate_text, "en", target_lang)
    else:
        return "Error: Translation model not available."

    # Load tokenizer and model
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        return f"Error loading model: {str(e)}"

    # Generate translation
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    except Exception as e:
        return f"Error during translation: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_text():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "Preflight successful"})
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    data = request.get_json()
    text = data.get('text')
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')

    if not text or not source_lang or not target_lang:
        return jsonify({"error": "Missing parameters. Please provide 'text', 'source_lang', and 'target_lang'."}), 400

    translation = translate(text, source_lang, target_lang)
    if "Error" in translation:
        return jsonify({"error": translation}), 500

    return jsonify({"translated_text": translation})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
