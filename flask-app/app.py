import os
import secrets

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify, render_template, flash, request, redirect, url_for, \
    send_from_directory, session
import pandas as pd
from PIL import Image, ImageOps
import io
from flask_cors import CORS
#from flask_table import Table, Col
from werkzeug.utils import secure_filename
import shutil
from scipy.ndimage import gaussian_filter
import easyocr
from datetime import datetime
import easyocr

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PREPROCESS_FOLDER = "preprocess"
STAGE1_FOLDER = "stage1_crop"
SAVED_ORIGINALS = "saved_originals"
OCR_OUTPUT = "ocr_output"

reader = easyocr.Reader(['en'], gpu=False)

for folder in [UPLOAD_FOLDER, STAGE1_FOLDER, PREPROCESS_FOLDER, SAVED_ORIGINALS]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STAGE1_FOLDER"] = STAGE1_FOLDER
app.config["PREPROCESS_FOLDER"] = PREPROCESS_FOLDER
app.config["SAVED_ORIGINALS"] = SAVED_ORIGINALS
app.config["OCR_OUTPUT"] = OCR_OUTPUT

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# class ImageTable(Table):
#     image = Col('Image')
#     imageName = Col('Image Name')

@app.route("/", methods=["GET", "POST"])
def index():
    print("hello, world!!")
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        if 'files' not in request.files:
            #flash('No file part')
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

        #flash("No valid files selected")
        return redirect(request.url)

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/loading_page", methods=["GET"])
def loading_page():
    #loading page renders THEN CALLS preprocess! so user waits on loading page
    #filenames = request.args.get("filenames", "").split(",") if request.args.get("filenames") else []
    return render_template("loading.html")

# before deleting files from UPLOAD_FOLDER, move them to SAVED_ORIGINALS
def move_originals(filename):
    original_path = os.path.join(UPLOAD_FOLDER, filename)
    saved_path = os.path.join(SAVED_ORIGINALS, filename)
    
    if os.path.exists(original_path):
        shutil.move(original_path, saved_path)  # Moves file instead of deleting it

@app.route("/edit", methods=["GET", "POST"])
def edit():
    return render_template("edit.html")


@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    # edited for bath uploads/processing
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
            new_width = int(width * (1 / 3))
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
            image = image.convert("RGB")
            image_np = np.array(image)

            # ### Alexia's Work
            # greyscale image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # blurring image
            sigma = 0.5
            image_np = gaussian_filter(image_np, sigma=sigma)

            # thickening the font
            image_np = cv2.bitwise_not(image_np)
            kernal = np.ones((2, 2), np.uint8)
            image_np = cv2.dilate(image_np, kernal, iterations=2)
            image_np = cv2.bitwise_not(image_np)

            # image binarization
            _, image_np = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # save in preprocess folder
            processed_pil = Image.fromarray(image_np)
            processed_pil.save(final_path)

            # ### OLD PREPROCESSING
            # image = Image.open(image_path)
            # image_np = np.array(image)

            # normalize image
            # image_np = cv2.normalize(image_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #
            # # noise removal
            # opening = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel_open, iterations=2)
            # opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)
            #
            # # grayscale
            # gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
            # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            #
            # processed_image = cv2.erode(gray, kernel_erode, iterations=1)
            #
            # # save in preprocess folder
            # processed_pil = Image.fromarray(processed_image)
            # processed_pil.save(final_path)

        except Exception as e:
            print(f"Error preprocessing {filename}: {e}")

        move_originals(filename) # move uploads to saved originals for later use -> editing etc

    #indents are so important guys LOL
    clear_folder(STAGE1_FOLDER) #cleanup
    clear_folder(UPLOAD_FOLDER) #also everytime we preprocess, we process ALL in these, so clear it !
    print("All images preprocessed successfully!")

@app.route("/ocr", methods=["POST"])
def ocr():
    print("do ocr")
    for filename in os.listdir(PREPROCESS_FOLDER):
        name = filename.split(".")[0]
        image_path = os.path.join(PREPROCESS_FOLDER, filename)
        final_path = os.path.join(OCR_OUTPUT, name + ".csv")

        try:
            image = Image.open(image_path)
            image_np = np.array(image)
            results = reader.readtext(image_np)
            ocr_df = pd.DataFrame(results, columns=['bbox', 'text', 'confidence'])
            ocr_df.to_csv(final_path)
            print(final_path + " saved")

        except Exception as e:
            print(f"Error preprocessing {filename}: {e}")

    return jsonify({"success": True, "message": f"OCR Complete"})

@app.route("/reprocess", methods=["POST"])
def reprocess_page():
    selected_images = request.form.getlist("selected_images") 

    if not selected_images:
        return redirect(url_for("image_gallery"))

    return render_template("reprocess.html", images=selected_images)

@app.route("/image_gallery", methods=["GET"])
def image_gallery():
    images = os.listdir(app.config["PREPROCESS_FOLDER"])
    print(images)
    print("====================================")
    return render_template("image_gallery.html", images=images)

@app.route("/loading_reprocess", methods=["GET"])
def loading_reprocess():
    return render_template("loading_reprocess.html")


@app.route("/preprocessed/<path:name>")
def download_preprocessed_file(name):
    return send_from_directory(app.config["PREPROCESS_FOLDER"], name)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

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


@app.route("/apply_reprocessing", methods=["POST"])
def apply_reprocessing():
    #redirects to loading screen before/while reprocessing.
    selected_images = request.form.getlist("selected_images")

    # store selected images and settings w/ session
    session["selected_images"] = selected_images
    session["reprocess_settings"] = {img: {
        "crop_factor": request.form[f"crop_factor_{img}"],
        "processing_strength": request.form[f"processing_strength_{img}"]
    } for img in selected_images}

    return redirect(url_for("loading_reprocess"))


@app.route("/start_reprocessing", methods=["POST"])
def start_reprocessing():
    #processes images after loading screen is shown
    selected_images = session.get("selected_images", [])
    reprocess_settings = session.get("reprocess_settings", {})

    for image_name in selected_images:
        settings = reprocess_settings.get(image_name, {})
        crop_factor = float(settings.get("crop_factor", 0.4))  # Default to 0.4 (2/5)
        processing_strength = settings.get("processing_strength", "medium")

        original_path = os.path.join(SAVED_ORIGINALS, image_name)
        preprocessed_path = os.path.join(PREPROCESS_FOLDER, image_name)

        if not os.path.exists(original_path):
            print(f"Original image for {image_name} not found.")
            continue

        image = Image.open(original_path)
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        new_width = int(width * crop_factor)
        cropped_image = image.crop((0, 0, new_width, height))

        processed_np = np.array(cropped_image)

        processed_np = apply_processing_strength(processed_np, processing_strength)

        processed_pil = Image.fromarray(processed_np)
        processed_pil.save(preprocessed_path)

    # clear session data
    session.pop("selected_images", None)
    session.pop("reprocess_settings", None)

    return jsonify({"message": "Reprocessing complete"}), 200


def apply_processing_strength(image_np, strength):

    kernel_three = np.ones((3, 3), np.uint8) #from open!
    kernel_two = np.ones((2, 2), np.uint8) #from erode!

    if strength == "soft":
        print("Applying soft processing")
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((2, 2), np.uint8)
        noise_removed = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21) 
        thresholded = cv2.adaptiveThreshold(noise_removed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  
        eroded = cv2.erode(thresholded, kernel_two, iterations=1)

    elif strength == "medium":
        print("Applying medium processing")
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # Use slightly stronger kernel and denoising
        # kernel = np.ones((3, 3), np.uint8)
        noise_removed = cv2.fastNlMeansDenoising(gray, None, 10, 10, 7)
        contrast_adjusted = cv2.equalizeHist(noise_removed)  
        thresholded = cv2.threshold(contrast_adjusted, 127, 255, cv2.THRESH_BINARY)[1]
        eroded = cv2.erode(thresholded, kernel_two, iterations=1)

    else:  # strong -> default
        print("Applying strong processing")
        image_np = cv2.normalize(image_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        opening = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel_three, iterations=2)
        opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)

        gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        eroded = cv2.erode(gray, kernel_two, iterations=1) 
    return eroded


@app.route("/output", methods=["GET"])
def output():
    #view the images
    images = os.listdir(app.config["PREPROCESS_FOLDER"])

    print(images)
    pd.set_option('display.max_columns', None)

    ### USE THIS FOR THE FINAL VERSION
    df = pd.read_csv(os.path.join(OCR_OUTPUT, "data.csv"))
    # print("df:\n", df)
    df_list = df.values.tolist()
    print("df_list:\n", df_list, "\n")
    print("====================================")

    for item1, item2 in zip(images, df_list):
        print(item1, item2)

    return render_template("output_gallery.html", images_and_data=zip(images, df_list), headings=df.columns.tolist())


@app.route("/finished", methods=["GET"])
def finished():
    #Tell user their file is downloading
    return render_template('finished.html')


@app.route("/about", methods=["GET"])
def about():
    #About us page
    return render_template('about.html')


if __name__ == "__main__":
    app.secret_key = secrets.token_hex(32)
    app.run(debug=True)
