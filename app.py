from flask import Flask, request, send_file, jsonify, render_template
import pandas as pd
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    print("hello, world!")
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    img = Image.open(file.stream)
    img.save("output.jpg")
    return jsonify({"message": "upload success"})


@app.route("/ocr", methods=["POST"])
def ocr():
    print("do ocr")


@app.route("/download", methods=["GET"])
def download():
    return send_file("output.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
