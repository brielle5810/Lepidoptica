from nanonets import NANONETSOCR
import os
import requests
from PIL import Image, ImageOps

model = NANONETSOCR()
#KATJA:
# MODEL_ID = '52e3f05e-50dd-430a-9b8d-42f28e0942b2'
# API_KEY = '6ef92718-f203-11ef-90b1-a6fe692a5476'
#BRIELLE:
# MODEL_ID = '5c1490c0-0edc-412e-949c-0e68060b76db'
# API_KEY = '10fe9db1-f1f6-11ef-b958-eaf9f4bd4f51'
#ALEX:
# MODEL_ID = '9c3dbc9b-67c2-449a-a8dd-15fe85c9a9e1'
# API_KEY = '92421153-13bf-11f0-b71f-f2abfc032ff7'
#ALEXIA:
MODEL_ID = '82b4a9e4-bf0e-4fcd-9e3f-c24fab0d4039'
API_KEY = '8a12cc28-1815-11f0-b123-ba3b08fd7528'
model.set_token(API_KEY)
# MODEL_ID = '5c1490c0-0edc-412e-949c-0e68060b76db'
# API_KEY = '10fe9db1-f1f6-11ef-b958-eaf9f4bd4f51'

from datetime import datetime

current_batch_day = (datetime.utcnow() - datetime(1970, 1, 1)).days
print(current_batch_day)

url = f'https://app.nanonets.com/api/v2/Inferences/Model/{MODEL_ID}/ImageLevelInferences?start_day_interval=0&current_batch_day={current_batch_day}'
response = requests.post(url, auth=(API_KEY, ''), headers={"Content-Type": "application/json"})
response_data = response.json()
#print (response_data)

PREPROCESSED_DIR = "preprocessed_L_2"
OUTPUT_DIR = "NANONETS_DATA4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for image in response_data["moderated_images"]:
    image_filename = os.path.basename(image["original_file_name"])
    # image_basename, _ = os.path.splitext(image_filename)
    original_image_filename = image_filename

    image_filename = image_filename.replace(" ", "_")
    image_basename, _ = os.path.splitext(image_filename)

    print(f"Processing image {image_filename}...")

    # corresponding image from preprocessed directory
    image_path = os.path.join(PREPROCESSED_DIR, original_image_filename)

    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {PREPROCESSED_DIR}!!!!!!!!!!!!")
        continue

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)

    for box in image["moderated_boxes"]:

        try:

            box_id = box["id"]

            png_filename = f"{image_basename}_{box_id}.png".replace(" ", "_")
            png_path = os.path.join(OUTPUT_DIR, png_filename)

            if not os.path.exists(png_path):
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                print(f"Box {box_id} coordinates: ({xmin}, {ymin}) - ({xmax}, {ymax})")
                ocr_text = box["ocr_text"]

                print(f"Processing box {box_id} from {image_filename}...")
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                cropped_img.save(png_path, format="PNG")
                # gt_filename = f"{image_basename}_{box_id}.gt.txt"
                gt_filename = f"{image_basename}_{box_id}.gt.txt".replace(" ", "_")
                gt_path = os.path.join(OUTPUT_DIR, gt_filename)
                with open(gt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(ocr_text)
            else:
                print(f"File already exists: {png_path}")

        except Exception as e:
            print(f"Error processing image: {e}")

print(f"Processed images saved in '{OUTPUT_DIR}'.")