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
STAGE1_FOLDER = "stage1_crop"
for folder in [UPLOAD_FOLDER, STAGE1_FOLDER, PREPROCESS_FOLDER]:
    os.makedirs(folder, exist_ok=True)
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
            return redirect(url_for("loading_page", filenames=",".join(filenames)))

        flash("No valid files selected")
        return redirect(request.url)


@app.route("/loading_page", methods=["GET"])
def loading_page():

    #loading page renders THEN CALLS preprocess! so user waits on loading page
    #filenames = request.args.get("filenames", "").split(",") if request.args.get("filenames") else []
    return render_template("loading.html")


@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    # edited for bath uoloads/processing
    filenames = os.listdir(app.config["UPLOAD_FOLDER"])

    if not filenames:
        return jsonify({"message": "No files to preprocess"}), 400

    #crop and save in stage1 folder
    crop_images_in_batch(filenames)
    #preprocess cropped images and save in preprocess folder
    preprocess_images_in_batch()

    return jsonify({"message": "Batch preprocessing complete", "files": os.listdir(PREPROCESS_FOLDER)})


def crop_images_in_batch(filenames):
    for filename in filenames:
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        cropped_path = os.path.join(STAGE1_FOLDER, filename)

        try:
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            new_width = int(width * (2 / 5))
            cropped_image = image.crop((0, 0, new_width, height))

            cropped_image.save(cropped_path)
        except Exception as e:
            (f"Error cropping {filename}: {e}")

    print("All images cropped successfully!")


def preprocess_images_in_batch():
    # create kernels once, instead of repeatedly for each pic as b4....
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_erode = np.ones((2, 2), np.uint8)

    for filename in os.listdir(STAGE1_FOLDER):
        image_path = os.path.join(STAGE1_FOLDER, filename)
        final_path = os.path.join(PREPROCESS_FOLDER, filename)

        try:
            image = Image.open(image_path)
            image_np = np.array(image)

            # normalize image
            image_np = cv2.normalize(image_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # noise removal
            opening = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel_open, iterations=2)
            opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)

            # grayscale
            gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            processed_image = cv2.erode(gray, kernel_erode, iterations=1)

            # save in preprocess folder
            processed_pil = Image.fromarray(processed_image)
            processed_pil.save(final_path)  # Overwrite the image in PREPROCESS_FOLDER

        except Exception as e:
            print(f"Error preprocessing {filename}: {e}")

    print("All images preprocessed successfully!")

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
# app.add_url_rule(
#     "/uploads/<name>", endpoint="download_file", build_only=True
# )

@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    preprocessed_path = os.path.join(app.config["PREPROCESS_FOLDER"], filename)
    stage1_path = os.path.join(app.config["STAGE1_FOLDER"], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
    if os.path.exists(stage1_path):
        os.remove(stage1_path)
        return jsonify({"success": True, "message": f"{filename} deleted"})
    else:
        return jsonify({"success": False, "error": "File not found"}), 404

    return jsonify({"success": True, "message": f"{filename} deleted"})
@app.route("/delete_all", methods=["DELETE"])
def delete_all():
    #file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    #preprocessed_path = os.path.join(app.config["PREPROCESS_FOLDER"], filename)
    #stage1_path = os.path.join(app.config["STAGE1_FOLDER"], filename)
    for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, STAGE1_FOLDER]:
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
