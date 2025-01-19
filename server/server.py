from flask import Flask, request, jsonify
import util
app = Flask(__name__)

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