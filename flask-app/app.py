import os
import secrets

import cv2
import numpy as np
from flask import Flask, request, session, send_file, jsonify, render_template, flash, request, redirect, url_for, \
    send_from_directory, abort, make_response
import pandas as pd
from PIL import Image, ImageOps
from flask_cors import CORS
#from flask_table import Table, Col
from werkzeug.utils import secure_filename
import shutil
from scipy.ndimage import gaussian_filter
import easyocr

import textReader

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
MODIFIED_FOLDER = "modified"
PREPROCESS_FOLDER = "preprocess"
STAGE1_FOLDER = "stage1_crop"
SAVED_ORIGINALS = "saved_originals"
OCR_OUTPUT = "ocr_output"

num_files = 0
num_processed = 0

reader = easyocr.Reader(['en'], gpu=False, recog_network="fine_tuning2")

for folder in [UPLOAD_FOLDER, STAGE1_FOLDER, PREPROCESS_FOLDER, SAVED_ORIGINALS, OCR_OUTPUT, MODIFIED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STAGE1_FOLDER"] = STAGE1_FOLDER
app.config["PREPROCESS_FOLDER"] = PREPROCESS_FOLDER
app.config["MODIFIED_FOLDER"] = MODIFIED_FOLDER
app.config["SAVED_ORIGINALS"] = SAVED_ORIGINALS
app.config["OCR_OUTPUT"] = OCR_OUTPUT

app.secret_key = secrets.token_hex(32)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# class ImageTable(Table):
#     image = Col('Image')
#     imageName = Col('Image Name')

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)

        files = request.files.getlist("files")  # Get multiple files
        filenames = []

        max_files = 500
        if len(files) > max_files:
            # user should not be able to submit if this is the case
            # taken care of with js
            # but if they do... hopefully it at least wont crash
            print(f"Too many files selected. Only {max_files} files have been selected.")
            files = files[:max_files]

        for file in files:
            if file and allowed_file(file.filename):

                filename = secure_filename(file.filename).replace(" ", "_")
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                filenames.append(filename)

        if filenames:
            return redirect(url_for("loading_page", filenames=",".join(filenames)))
        return redirect(request.url)

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/loading_page", methods=["GET"])
def loading_page():
    # oading page renders THEN CALLS preprocess! so user waits on loading page
    return render_template("loading.html")

# before deleting files from UPLOAD_FOLDER, move them to SAVED_ORIGINALS-> need them for reprocessing
def move_originals(filename):
    original_path = os.path.join(UPLOAD_FOLDER, filename)
    saved_path = os.path.join(SAVED_ORIGINALS, filename)
    
    if os.path.exists(original_path):
        shutil.move(original_path, saved_path)  # Moves file instead of deleting it

# @app.route("/edit", methods=["GET", "POST"])
# def edit():
#     return render_template("edit.html")


@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    # edited for batch uploads/processing
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
            new_width = int(width * 0.40)
            cropped_image = image.crop((0, 0, new_width, height))

            cropped_image.save(cropped_path)
        except Exception as e:
            (f"Error cropping {filename}: {e}")

    print("All images cropped successfully!")


def preprocess_images_in_batch():
    # create kernel once, instead of repeatedly for each pic as b4....
    kernel_22 = np.ones((2, 2), np.uint8)

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

            # thickening the font
            image_np = cv2.bitwise_not(image_np)
            image_np = cv2.dilate(image_np, kernel_22, iterations=1)
            image_np = cv2.bitwise_not(image_np)

            # blurring image
            sigma = 0.5
            image_np = gaussian_filter(image_np, sigma=sigma)

            # image binarization
            _, image_np = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # save in preprocess folder
            processed_pil = Image.fromarray(image_np)
            processed_pil.save(final_path)

        except Exception as e:
            print(f"Error preprocessing {filename}: {e}")

        move_originals(filename) # move uploads to saved originals for later use -> editing etc

    #indents are so important guys LOL
    clear_folder(STAGE1_FOLDER) #cleanup
    clear_folder(UPLOAD_FOLDER) #also everytime we preprocess, we process ALL in these, so clear it !
    print("All images preprocessed successfully!")

#Attempting new method to fetch number of remaining files to be processed by OCR for progres bar
#Resources used:
#https://www.w3schools.com/howto/howto_js_progressbar.asp
#https://stackoverflow.com/questions/58996870/update-flask-web-page-with-python-script

@app.route('/progress')
def progress():
    # Send the number of files currently done processing out of the total
    global num_processed
    global num_files
    if num_files != 0:
        #return str(int(float(num_processed) * 100.0 / float(num_files)))
        packaged_data = {'num_processed':num_processed, 'num_files':num_files}
        return jsonify(packaged_data)
        #return str(num_files)

    else:
        return "0"


@app.route("/ocr", methods=["POST"])
def ocr():
    #This version creates individual .csv files for each image
    # for filename in os.listdir(PREPROCESS_FOLDER):
    #     name = filename.split(".")[0]
    #     image_path = os.path.join(PREPROCESS_FOLDER, filename)
    #     final_path = os.path.join(OCR_OUTPUT, name + ".csv")
    #
    #     try:
    #         image = Image.open(image_path)
    #         image_np = np.array(image)
    #         results = reader.readtext(image_np)
    #         ocr_df = pd.DataFrame(results, columns=['bbox', 'text', 'confidence'])
    #         ocr_df.to_csv(final_path)
    #         print(final_path + " saved")
    #
    #     except Exception as e:
    #         print(f"Error preprocessing {filename}: {e}")

    #Attempting to create a version that just fills data in data.csv

    #Update global vars
    global num_processed
    global num_files
    num_files = len(os.listdir(PREPROCESS_FOLDER))
    num_processed = 0

    #Delete data.csv if it exists
    data_path = os.path.join(OCR_OUTPUT, "data.csv")
    confidence_path = os.path.join(OCR_OUTPUT, "confidence.csv")
    final_path = os.path.join(OCR_OUTPUT, "parsed.csv")
    final_confidence_path = os.path.join(OCR_OUTPUT, "parsed_confidence.csv")


    for filename in os.listdir(OCR_OUTPUT):
        if filename == "data.csv":
            os.remove(data_path)
        if filename == "confidence.csv":
            os.remove(confidence_path)
        if filename == "parsed.csv":
            os.remove(final_path)
        if filename == "parsed_confidence.csv":
            os.remove(final_confidence_path)


    #Create the csvs
    data = open(data_path, "w", encoding="utf8")
    data.write("CatalogNumber,Specimen_voucher,Family,Genus,Species,Subspecies,Sex,Country,State,County,Locality name,Elevation min,Elevation max,Elevation unit,Collectors,Latitude,Longitude,Georeferencing source,Georeferencing precision,Questionable label data,Do not publish,Collecting event start,Collecting event end,Date verbatim,Remarks public,Remarks private,Cataloged date,Cataloger First,Cataloger last,Prep type 1,Prep count 1,Prep type 2,Prep number 2,Prep type 3,Prep number 3,Other record number,Other record source,publication,publication")
    data.close()

    confidence = open(confidence_path, "w", encoding="utf8")
    confidence.write("CatalogNumber,Specimen_voucher,Family,Genus,Species,Subspecies,Sex,Country,State,County,Locality name,Elevation min,Elevation max,Elevation unit,Collectors,Latitude,Longitude,Georeferencing source,Georeferencing precision,Questionable label data,Do not publish,Collecting event start,Collecting event end,Date verbatim,Remarks public,Remarks private,Cataloged date,Cataloger First,Cataloger last,Prep type 1,Prep count 1,Prep type 2,Prep number 2,Prep type 3,Prep number 3,Other record number,Other record source,publication,publication")
    confidence.close()


    #Write data to csv
    #Confidence for each data point stored in confidence.csv

    for filename in os.listdir(PREPROCESS_FOLDER):
        name = filename.split(".")[0]
        image_path = os.path.join(PREPROCESS_FOLDER, filename)

        try:
            image = Image.open(image_path)
            image_np = np.array(image)
            results = reader.readtext(image_np)
            ocr_df = pd.DataFrame(results, columns=['bbox', 'text', 'confidence'])
            #ocr_df.to_csv(final_path)

            transcription_lines = "\n\""
            confidence_lines = "\n"

            #num_processed = num_processed + 0.5

            num_cols = len(ocr_df)
            column_counter = 0
            for i in range(num_cols):
                #If there are more sections of data in the dataframe than fields in the csv, start lumping data together in the last column
                #Note: This will not update the confidence for that column with the current implementation
                #temp_str = ""
                #if str(ocr_df.loc[i].at["text"]) != "nan":
                temp_str = str(ocr_df.loc[i].at["text"]).replace("\"", "\"\"")

                if column_counter >= 38:
                    #transcription_lines = transcription_lines + ocr_df[line][1]
                    transcription_lines = transcription_lines + temp_str
                elif column_counter == 0:
                    transcription_lines = transcription_lines + temp_str
                    confidence_lines = confidence_lines + str(ocr_df.loc[i].at["confidence"])
                else:
                    transcription_lines = transcription_lines + "\",\"" + temp_str
                    confidence_lines = confidence_lines + "," + str(ocr_df.loc[i].at["confidence"])
                column_counter = column_counter + 1

            if column_counter < 38:
                for i in range (38 - column_counter):
                    transcription_lines = transcription_lines + "\",\""
                    confidence_lines = confidence_lines + ","

            transcription_lines = transcription_lines + "\""

            data = open(data_path, "a", encoding="utf8")
            data.write(transcription_lines)
            data.close()

            confidence = open(confidence_path, "a", encoding="utf8")
            confidence.write(confidence_lines)
            confidence.close()

            num_processed = num_processed + 1
            print(name + " processed")

        except Exception as e:
            print(f"Error preprocessing {filename}: {e}")

    #TODO: RUN TEXTREADER.PY
    textReader.parsing()

    return jsonify({"success": True, "message": f"OCR Complete"})

@app.route("/reprocess", methods=["POST"])
def reprocess_page():
    selected_images = request.form.getlist("selected_images")

    if not selected_images:
        return redirect(url_for("image_gallery"))

    session['reprocess_images'] = selected_images
    return jsonify({"redirect": url_for("reprocess_page_render")})

@app.route("/reprocess_view")
def reprocess_page_render():
    images = session.get('reprocess_images', [])
    if not images:
        return redirect(url_for("image_gallery"))

    #make a copy of each image in modified folder, this will be outputted nextdoor
    for image in images:
        source = os.path.join(PREPROCESS_FOLDER, image)
        shutil.copy(source, MODIFIED_FOLDER)

    return render_template("reprocess.html", images=images)

@app.route("/update_img_gallery", methods=["POST"])
def update_img_gallery():
    #move all images from modified folder to preprocess, and delete each
    for image in os.listdir(MODIFIED_FOLDER):
        try:
            source = os.path.join(MODIFIED_FOLDER, image)
            shutil.copy(source, PREPROCESS_FOLDER)
            os.remove(source)
            print(f"Image '{source}' removed successfully from '{MODIFIED_FOLDER}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    return jsonify({"success": True, "message": f"All images updated!"})


@app.route("/image_gallery", methods=["GET"])
def image_gallery():
    images = os.listdir(app.config["PREPROCESS_FOLDER"])
    return render_template("image_gallery.html", images=images)

@app.route("/loading_ocr", methods=["GET"])
def loading_ocr():
    return render_template("loading_ocr.html")

@app.route("/preprocessed/<path:name>")
def download_preprocessed_file(name):
    return send_from_directory(app.config["PREPROCESS_FOLDER"], name)

@app.route("/modified/<path:name>")
def download_modified_file(name):
    return send_from_directory(app.config["MODIFIED_FOLDER"], name)

@app.route("/output_gallery/<path:name>")
def download_saved_original_file(name):
    return send_from_directory(app.config["SAVED_ORIGINALS"], name)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

def getImagePairs():
    og_images = sorted(os.listdir(app.config["SAVED_ORIGINALS"]))
    preprocessed_images = sorted(os.listdir(app.config["PREPROCESS_FOLDER"]))
    saved_images = []
    for filename in og_images:
        if filename in preprocessed_images and filename not in saved_images:
            saved_images.append(filename)
    imagePairs = list(zip(saved_images, preprocessed_images))
    return imagePairs

#using htmx to delete rows from the table (and corresponding data, images)
@app.route("/deleterow/<int:row_index>", methods=["DELETE"])
def delete_row(row_index):
    data_path = os.path.join(OCR_OUTPUT, "parsed.csv")
    conf_path = os.path.join(OCR_OUTPUT, "parsed_confidence.csv")

    # load as dataframs to then edit
    try:
        df_data = pd.read_csv(data_path)
        df_conf = pd.read_csv(conf_path, dtype=float).fillna(value=100)
        # correct row index check
        if row_index < 0 or row_index >= len(df_data):
            abort(404, description="Invalid row index")

    except Exception as e:
        abort(500, description="Error reading CSV files: " + str(e))

    #drop from both and reset the indices
    df_data.drop(index=row_index, inplace=True)
    df_conf.drop(index=row_index, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_conf.reset_index(drop=True, inplace=True)

    try:
        # write to csv
        df_data.to_csv(data_path, index=False, header=True)
        df_conf.to_csv(conf_path, index=False, header=True)
    except Exception as e:
        abort(500, description="Error writing to CSV files: " + str(e))

    # issue-> this doesn't remove the images!
    # find and remove the corresponding image files
    imagePairs = getImagePairs()
    original_filename, preprocessed_filename = imagePairs[row_index]

    delete_file(original_filename) #deletes from all folders...

    # for htmx to remove the row from the DOM -> return "" and 204 status code
    response = make_response("", 204)
    # redirect /output route
    response.headers["HX-Redirect"] = url_for("output")
    return response

# data persistency
@app.route("/update_cell", methods=["POST"])
def update_cell():
    data = request.get_json()
    row_index = int(data["row"])
    col_index = int(data["col"])
    new_value = data["value"]
    print(new_value)

    data_path = os.path.join(OCR_OUTPUT, "parsed.csv")

    try:
        df_data = pd.read_csv(data_path, dtype=str) #read all as string to avoid compat type errors

        if row_index < 0 or row_index >= len(df_data):
            return jsonify(success=False, error="Invalid row index"), 404
        if col_index < 0 or col_index >= len(df_data.columns):
            return jsonify(success=False, error="Invalid column index"), 404

        # update the value
        df_data.iloc[:, col_index] = df_data.iloc[:, col_index].astype("object")
        df_data.iat[row_index, col_index] = new_value

        # save back to CSV
        df_data.to_csv(data_path, index=False)
        return jsonify(success=True)

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route("/update_mult_cells", methods=["POST"])
def update_mult_cells():
    data_path = os.path.join(OCR_OUTPUT, "parsed.csv")

    if not os.path.exists(data_path):
        return jsonify(success=False, error="CSV file not found"), 404

    if os.stat(data_path).st_size == 0:
        return jsonify(success=False, error="CSV is empty"), 400

    try:
        updates = request.get_json()

        if not isinstance(updates, list):
            return jsonify(success=False, error="Expected a list of updates"), 400

        df = pd.read_csv(data_path)

        for update in updates: #each cell
            row = int(update.get("row", -1))
            col = int(update.get("col", -1))
            value = update.get("value", "")

            if 0 <= row < len(df) and 0 <= col < len(df.columns):
                # update value for each cell. to be empty for other selected cells when merging
                # or to swap vals in drag
                df.iloc[:, col] = df.iloc[:, col].astype("object")
                df.iat[row, col] = value
            else:
                print(f"invalid update attempted: row={row}, col={col} --skipping")

        if df.columns.empty:
            print("no columns — skipping")
            return jsonify(success=False, error="df has no columns"), 500

        df.to_csv(data_path, index=False)
        return jsonify(success=True)

    except Exception as e:
        print("ERROR:", e)
        return jsonify(success=False, error=str(e)), 500

@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, STAGE1_FOLDER, SAVED_ORIGINALS, MODIFIED_FOLDER]:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"success": True, "message": f"{filename} attempted deletion in all folders"})
    # file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    # preprocessed_path = os.path.join(app.config["PREPROCESS_FOLDER"], filename)
    # stage1_path = os.path.join(app.config["STAGE1_FOLDER"], filename)
    # saved_originals_path = os.path.join(app.config["SAVED_ORIGINALS"], filename)
    # modified_path = os.path.join(app.config["MODIFIED_FOLDER"], filename)
    #
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    # if os.path.exists(preprocessed_path):
    #     os.remove(preprocessed_path)
    # if os.path.exists(stage1_path):
    #     os.remove(stage1_path)
    # if os.path.exists(saved_originals_path):
    #     os.remove(saved_originals_path)
    # if os.path.exists(modified_path):
    #     os.remove(modified_path)

@app.route("/delete_all", methods=["DELETE"])
def delete_all():
    for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, STAGE1_FOLDER, SAVED_ORIGINALS, MODIFIED_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
    return jsonify({"success": True, "message": "All files deleted"})


# @app.before_request
# def clear_on_new_session():
#     if "visited" not in session:
#         session["visited"] = True
#         print("clearing it all!")
#         delete_all()


@app.route("/download", methods=["GET"])
def download():
    return send_file("output.csv", as_attachment=True)

@app.route("/img_reprocessing", methods=["POST"])
def img_reprocessing():
    #processes images after loading screen is shown
    data = request.get_json()

    image_name = data.get("image_name")
    processing_strength = data.get("processing_strength")
    line_value = float(data.get("line_value"))
    crop_factor = float(data.get("crop_factor"))
    print(image_name)
    print(crop_factor)
    print(line_value)
    print(processing_strength)

    original_path = os.path.join(SAVED_ORIGINALS, image_name)
    modified_path = os.path.join(MODIFIED_FOLDER, image_name)

    if not os.path.exists(original_path):
        print(f"Original image for {image_name} not found.")
        return jsonify({"message": "Image not found"}), 200

    image = Image.open(original_path)
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    new_width = int(width * crop_factor)
    cropped_image = image.crop((0, 0, new_width, height))

    processed_np = np.array(cropped_image)

    processed_np = apply_processing_strength(processed_np, processing_strength, line_value)

    processed_pil = Image.fromarray(processed_np)
    processed_pil.save(modified_path)

    return jsonify({"success": "Reprocessing complete"}), 200

def apply_thickness(image_np, thickness):
    # thickening the font
    if thickness == 3:
        return image_np

    image_np = cv2.bitwise_not(image_np)
    kernal = np.ones((2, 2), np.uint8)

    if thickness == 0:
        image_np = cv2.erode(image_np, kernal, iterations=3)
    if thickness == 1:
        image_np = cv2.erode(image_np, kernal, iterations=2)
    if thickness == 2:
        image_np = cv2.erode(image_np, kernal, iterations=1)
    if thickness == 4:
        image_np = cv2.dilate(image_np, kernal, iterations=1)
    if thickness == 5:
        image_np = cv2.dilate(image_np, kernal, iterations=2)
    if thickness == 6:
        image_np = cv2.dilate(image_np, kernal, iterations=3)

    image_np = cv2.bitwise_not(image_np)
    return image_np


def apply_processing_strength(image_np, strength, thickness):
    if strength == "soft":
        print("Applying soft processing")
        # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Thickness
        image_np = apply_thickness(image_np, thickness)

        # image binarization
        image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_np = clahe.apply(image_np)

        # To remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_np = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)

        # Sharpening
        laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
        image_np = cv2.convertScaleAbs(image_np - 0.7 * laplacian)

    elif strength == "medium":
        print("Applying medium processing")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # Use slightly stronger kernel and denoising
        # kernel = np.ones((3, 3), np.uint8)
        image_np = cv2.fastNlMeansDenoising(image_np, None, 10, 10, 7)
        image_np = cv2.equalizeHist(image_np)
        image_np = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)[1]

    else:  # strong -> default
        print("Applying strong processing")

        # greyscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # thickening the font
        image_np = apply_thickness(image_np, thickness)

        # blurring image
        sigma = 0.5
        image_np = gaussian_filter(image_np, sigma=sigma)

        # image binarization
        _, image_np = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image_np


@app.route("/output", methods=["GET"])
def output():
    #NEED TO MARK if IT HAS ALREADY BEEN RUN, or it will overwrite user changes
    # global has_parsed
    #
    # if not has_parsed:
    #     textReader.parsing()
    #     has_parsed = True

    final_path = os.path.join(OCR_OUTPUT, "data.csv")

    #view the images
    og_images = os.listdir(app.config["SAVED_ORIGINALS"])
    saved_images = []
    images = os.listdir(app.config["PREPROCESS_FOLDER"])
    print("IMAGES: ", images)
    print("SAVED ORIGINALS: ", og_images)

    for filename in og_images:
        if filename in images and filename not in saved_images:
            saved_images.append(filename)

    print("OG IMAGES: ", og_images, "\n")
    imagePairs = list(zip(saved_images, images))

    print(imagePairs)
    pd.set_option('display.max_columns', None)

    ### USE THIS FOR THE FINAL VERSION
    df = pd.read_csv(os.path.join(OCR_OUTPUT, "parsed.csv")).fillna('')
    df_list = df.values.tolist()
    print("df_list:\n", df_list, "\n")
    print("====================================")

    for item1, item2 in zip(images, df_list):
        print(item1, item2)

    confidences = pd.read_csv(os.path.join(OCR_OUTPUT, "parsed_confidence.csv"), dtype=float).fillna(value = 100)
    print("confidence_list:\n", df_list, "\n")
    print("====================================")

    for item3 in confidences.values.tolist():
        print(item3)

    # print("df:\n", df)
    return render_template("output_gallery.html",
                           images_and_data=zip(imagePairs, df_list),
                           headings=df.columns.tolist(),
                           confidence_list=confidences.values.tolist())


@app.route("/finished", methods=["GET"])
def finished():
    #Tell user their file is downloading
    return render_template('finished.html')


@app.route("/about", methods=["GET"])
def about():
    #About us page
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
