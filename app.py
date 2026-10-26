from flask import Flask, render_template, request, jsonify
from pipeline.analyzer import NewsAnalyzer

app = Flask(__name__)
analyzer = NewsAnalyzer()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text_input = request.form.get('news_text', '')
    url_input = request.form.get('news_url', '')

    if not text_input.strip() and not url_input.strip():
        return render_template('index.html', error="Please provide either news text or a URL.")

    result = analyzer.analyze(text_input=text_input, url_input=url_input)
    
    if 'error' in result:
        return render_template('index.html', error=result['error'])

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)