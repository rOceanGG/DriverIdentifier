from flask import Flask, request, jsonify
import util
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
CORS(app)

@app.route('/classify-image', methods = ['GET', 'POST'])
def classify_image():
    imageData = request.form['imageData']

    res = jsonify(util.classifyImage(imageData))

    res.headers.add('Access-Control-Allow-Origin', '*')

    return res

if __name__ == '__main__':
    # Load model artifacts before starting the server
    util.loadArtifacts()
    app.run(port=5000)