from nanonets import NANONETSOCR
import os
import requests
from PIL import Image, ImageOps

model = NANONETSOCR()
model.set_token
MODEL_ID =
API_KEY =

from datetime import datetime

current_batch_day = (datetime.utcnow() - datetime(1970, 1, 1)).days
print(current_batch_day)

url = f'https://app.nanonets.com/api/v2/Inferences/Model/{MODEL_ID}/ImageLevelInferences?start_day_interval=0&current_batch_day={current_batch_day}'
response = requests.post(url, auth=(API_KEY, ''), headers={"Content-Type": "application/json"})
response_data = response.json()
#print (response_data)

PREPROCESSED_DIR = "preprocessed"
OUTPUT_DIR = "NANONETS_DATA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for image in response_data["moderated_images"]:
    image_filename = os.path.basename(image["original_file_name"])
    image_basename, _ = os.path.splitext(image_filename)

    # corresponding image from preprocessed directory
    image_path = os.path.join(PREPROCESSED_DIR, image_filename)

    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {PREPROCESSED_DIR}!!!!!!!!!!!!")
        continue

    img = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

    for box in image["moderated_boxes"]:

        box_id = box["id"]

        tiff_filename = f"{image_basename}_{box_id}.tiff"
        tiff_path = os.path.join(OUTPUT_DIR, tiff_filename)

        if not os.path.exists(tiff_path):
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            ocr_text = box["ocr_text"]

            print(f"Processing box {box_id} from {image_filename}...")
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            cropped_img.save(tiff_path, format="TIFF")
            gt_filename = f"{image_basename}_{box_id}.gt.txt"
            gt_path = os.path.join(OUTPUT_DIR, gt_filename)
            with open(gt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(ocr_text)
        else:
            print(f"File already exists: {tiff_path}")


print(f"Processed images saved in '{OUTPUT_DIR}'.")