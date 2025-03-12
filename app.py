import os

from flask import Flask, request, send_file, jsonify, render_template, flash, request, redirect, url_for, send_from_directory
import pandas as pd
from PIL import Image
import io
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    print("hello, world!!")
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    # file = request.files["file"]
    # img = Image.open(file.stream)
    # img.save("output.jpg")
    # return jsonify({"message": "upload success"})

    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist("files")  # Get multiple files
        filenames = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                filenames.append(filename)

        if filenames:
            return redirect(url_for("loading_page", filenames=",".join(filenames)))

        flash("No valid files selected")
        return redirect(request.url)

            # redirected to the uploaded file see def download_file(name)
             #when uploaded 1 file, not in use rn with mult
            #return redirect(url_for('download_file', name=filename))

@app.route("/loading_page", methods=["GET"])
def loading_page():
    filenames = request.args.get("filenames", "").split(",") if request.args.get("filenames") else []
    return render_template("loading.html", filenames=filenames)

@app.route("/ocr", methods=["POST"])
def ocr():
    print("do ocr")
@app.route("/image_gallery", methods=["GET"])
def image_gallery():
    images = os.listdir(app.config["UPLOAD_FOLDER"])
    print(images)
    return render_template("image_gallery.html", images=images)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)
app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)

@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True, "message": f"{filename} deleted"})
    else:
        return jsonify({"success": False, "error": "File not found"}), 404

@app.route("/delete_all", methods=["DELETE"])
def delete_all():
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"success": True, "message": "All files deleted"})
@app.route("/download", methods=["GET"])
def download():
    return send_file("output.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
