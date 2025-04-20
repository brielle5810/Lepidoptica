import shutil
from logging import exception

import cv2
import numpy as np
import pytesseract
import os
import PIL
from PIL import Image, ImageEnhance, ImageOps
from symspellpy import SymSpell, Verbosity
from scipy.ndimage import gaussian_filter
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def crop_left_half(image_path):
    print(f'Cropping the image in {image_path}')
    image = Image.open(image_path)
    after_path = 'rotated_and_cropped/'
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    width, height = image.size  # Get actual image dimensions

    fraction = 0.45
    new_width = int(width * fraction)
    left_half = image.crop((0, 0, new_width, height))

    output_path = os.path.join(after_path, os.path.basename(image_path))
    left_half.save(output_path, format="JPEG")

def rotate_180(image_path):
    #this should now just put it 'correct' and not rotate entirely...
    print(f'Rotating the image in {image_path}')
    after_path = 'rotated/'
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    print(after_path + image_path.split('/')[1])
    image.save(after_path + image_path.split('/')[1])

    crop_left_half(after_path + image_path.split('/')[1])

def binarize(image_path):
    #there are 3 possible preprocessing, attempt two worked the best
    print(f'Preprocessing {image_path}')
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)

    # attempt one // BEST VERSION
    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thickness
    image = cv2.bitwise_not(image)
    kernal = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)
    image = cv2.bitwise_not(image)

    # image binarization
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # To remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Sharpening
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    image = cv2.convertScaleAbs(image - 0.7 * laplacian)

    # attempt two
    #greyscale image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # thickening the font
    # image = cv2.bitwise_not(image)
    # kernal = np.ones((2, 2), np.uint8)
    # image = cv2.dilate(image, kernal, iterations=1)
    # image = cv2.bitwise_not(image)
    #
    # # blurring image
    # sigma = 0.5
    # image = gaussian_filter(image, sigma=sigma)
    #
    # # image binarization
    # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # attempt three
    # # normalize image
    # image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # # noise removal
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    # opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)
    # gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # #kernel = np.ones((2, 2), np.uint8)  # smaller kernel for subtle effects
    # #processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)  # fill gaps instead of erosion
    # # thinning, skeletonization
    # kernel = np.ones((2, 2), np.uint8)
    # erosion = cv2.erode(gray, kernel, iterations=1)

    #show image
    os.makedirs('preprocessed', exist_ok=True)
    cv2.imwrite('preprocessed/' + image_path.split('/')[1], image)

if __name__ == '__main__':

    raw_images = 'example/'

    if not os.path.exists(raw_images):
        exception("The raw image file path does not exist")

    if os.path.exists('rotated_and_cropped/'):
        shutil.rmtree('rotated_and_cropped/')
        os.makedirs('rotated_and_cropped/')
    else:
        os.makedirs('rotated_and_cropped/')

    if os.path.exists('rotated/'):
        shutil.rmtree('rotated/')
        os.makedirs('rotated/')
    else:
        os.makedirs('rotated/')

    if os.path.exists('preprocessed'):
        shutil.rmtree('preprocessed')
        os.makedirs('preprocessed')
    else:
        os.makedirs('preprocessed')

    for file in os.listdir(raw_images):
        rotate_180(f'{raw_images}{file}')

    for file in os.listdir('rotated_and_cropped/'):
        binarize(f'rotated_and_cropped/{file}')

