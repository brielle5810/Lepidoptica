import os

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify, render_template, flash, request, redirect, url_for, send_from_directory
import pandas as pd
from PIL import Image, ImageOps
import io
from flask_cors import CORS
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PREPROCESS_FOLDER = "preprocess"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PREPROCESS_FOLDER"] = PREPROCESS_FOLDER
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
            preprocess()
            return redirect(url_for("loading_page", filenames=",".join(filenames)))

        flash("No valid files selected")
        return redirect(request.url)

            # redirected to the uploaded file see def download_file(name)
             #when uploaded 1 file, not in use rn with mult
            #return redirect(url_for('download_file', name=filename))

@app.route("/loading_page", methods=["GET"])
def loading_page():
    #filenames = request.args.get("filenames", "").split(",") if request.args.get("filenames") else []
    return render_template("loading.html")

@app.route("/preprocess", methods=["GET","POST"])
def preprocess():
    filenames = os.listdir(app.config["UPLOAD_FOLDER"])

    for filename in filenames:
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        preprocessed_path = os.path.join(app.config["PREPROCESS_FOLDER"], filename)

        try:
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            image = crop_left_half(image)
            image = preproc_image(np.array(image))

            processed_image = Image.fromarray(image)
            processed_image.save(preprocessed_path)

            print(f"Saved preprocessed image: {preprocessed_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return jsonify({"message": f"Error processing {filename}: {e}"}), 500

    return jsonify({"message": "Preprocessing complete", "files": filenames})

def crop_left_half(image):
    image = ImageOps.exif_transpose(image)
    # height, width, _ = image.shape
    width, height = image.size  # Get actual image dimensions

    # width = image.shape[1]
    fraction = 2 / 5
    new_width = int(width * fraction)
    # image.crop((xmin, ymin, xmax, ymax))
    left_half = image.crop((0, 0, new_width, height))

    return left_half
    # output_path = os.path.join(after_path, os.path.basename(image_path))
    # left_half.save(output_path, format="JPEG")

def preproc_image(image):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)
    gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # kernel = np.ones((2, 2), np.uint8)  # smaller kernel for subtle effects
    # processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)  # fill gaps instead of erosion
    # thinning, skeletonization
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=1)

    return erosion
    # # show image
    # # os.mkdir('preprocessed2')
    # os.makedirs('preprocessed2', exist_ok=True)
    # cv2.imwrite('preprocessed2/' + image_path.split('/')[1], erosion)


@app.route("/ocr", methods=["POST"])
def ocr():
    print("do ocr")

@app.route("/image_gallery", methods=["GET"])
def image_gallery():
    #preprocess()
    images = os.listdir(app.config["PREPROCESS_FOLDER"])
    print(images)
    print("====================================")
    return render_template("image_gallery.html", images=images)


@app.route("/preprocessed/<path:name>")
def download_preprocessed_file(name):
    return send_from_directory(app.config["PREPROCESS_FOLDER"], name)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)
app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)

@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    preprocessed_path = os.path.join(app.config["PREPROCESS_FOLDER"], filename)


    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True, "message": f"{filename} deleted"})
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
        return jsonify({"success": True, "message": f"{filename} deleted"})

    else:
        return jsonify({"success": False, "error": "File not found"}), 404

@app.route("/delete_all", methods=["DELETE"])
def delete_all():
    for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
    return jsonify({"success": True, "message": "All files deleted"})
    # for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
    #     file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #
    # return jsonify({"success": True, "message": "All files deleted"})

@app.route("/download", methods=["GET"])
def download():
    return send_file("output.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
