from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/classify-image', methods = ['GET', 'POST'])
def classify_image():
    return "This page classifies images"


if __name__ == '__main__':
    app.run(port=5000)