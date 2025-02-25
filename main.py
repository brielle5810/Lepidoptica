import cv2
import numpy as np
import pytesseract
import os
import PIL
from PIL import Image, ImageEnhance, ImageOps
from pip._internal.metadata import pkg_resources
from symspellpy import SymSpell, Verbosity
import json
import sys

import smartcrop

#import keras_ocr
#import tensorflow as tf

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

uncropped_path = 'uncropped/'
cropped_path = 'cropped/'

def crop_left_half(image_path):
    after_path = 'cut_turned/'
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

    image.show()
    print("HELOJIf")

    # height, width, _ = image.shape
    width, height = image.size  # Get actual image dimensions

    # width = image.shape[1]

    fraction = 2 / 5
    new_width = int(width * fraction)
    # image.crop((xmin, ymin, xmax, ymax))
    left_half = image.crop((0, 0, new_width, height))

    output_path = os.path.join(after_path, os.path.basename(image_path))
    left_half.save(output_path, format="JPEG")

    #cv2.imwrite(after_path + image_path.split('/')[1], left_half)

def rotate_180(image_path):
    #this should now just put it 'correct' and not rotate entirely...
    after_path = 'rotated/'
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image.save(after_path + image_path.split('/')[1])
    image.show()
    crop_left_half(after_path + image_path.split('/')[1])

def binarize(image_path):
    image = cv2.imread(image_path)
    # normalize image
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)
    gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #kernel = np.ones((2, 2), np.uint8)  # smaller kernel for subtle effects
    #processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)  # fill gaps instead of erosion
    # thinning, skeletonization
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=1)
    #show image
    # os.mkdir('preprocessed2')
    os.makedirs('preprocessed2', exist_ok=True)
    cv2.imwrite('preprocessed2/' + image_path.split('/')[1], erosion)


def extract_text(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 4')
    #image_to_data(im, lang='eng', config=psm)
    print(text)
    return text


def post_correction(text):
    sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
    dictionary_path = "newDict2.txt"
    eng_Dict = "en-80k.txt"

    dictionary_path2 = "bigramDict.txt"
    sym_spell.load_dictionary(dictionary_path2, term_index=0, count_index=1, separator="$")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, separator="$")
    #sym_spell.load_dictionary(eng_Dict, term_index=0, count_index=1)
    #sym_spell.load_bigram_dictionary(dictionary_path2, term_index=0, count_index=1, separator="$")

    print(text)
    print("====================================================")

    # print first 10 words in dictionary
    #suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
    #suggestions = sym_spell.lookup(text, Verbosity.TOP, max_edit_distance=2, transfer_casing=True)

    suggestions = []
    for i in text.split():
        suggestions.append(sym_spell.lookup(i, Verbosity.TOP, max_edit_distance=2, transfer_casing=True))
        #maybe Verbosity.TOP, or ALL

    # suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
    # for suggestion in suggestions:
    #     print(suggestion.term)

    return suggestions
# def apply_smart_crop(image_path, crop_data):
#     # Open the image
#     image = Image.open(image_path)
#     img_width, img_height = image.size
#
#     # we are using full height to not cut off smaller labels, as is done when everything is left to SmartCrop
#     x, y, width, height = crop_data['x'], crop_data['y'], crop_data['width'], img_height
#
#     # im1 = im.crop((leftstart, pixelsfromtop, howfarright, howfardown))
#     cropped_image = image.crop((x, 0, x + width, height))
#
#     output_dir = "smartcrop"
#     os.makedirs(output_dir, exist_ok=True)
#
#     filename = os.path.basename(image_path)
#     cropped_image.save(os.path.join(output_dir, filename))

# def testSmartCrop(image_path):
#     image = Image.open(image_path)
#     sc = smartcrop.SmartCrop()
#     result = sc.crop(image, 100, 100)
#     output_dir = 'smartcrop'
#     os.makedirs(output_dir, exist_ok=True)
#
#     print(result['top_crop'])
#
#     apply_smart_crop(image_path, result['top_crop'])


if __name__ == '__main__':
    # why so many diff directories?
    # to compare different stages of the image processing and processing techniques
    # this is just a test so... chill
    for file in os.listdir('uncropped/'):
        rotate_180(f'uncropped/{file}')

    for file in os.listdir('cut_turned/'):
        binarize(f'cut_turned/{file}')
        (f'preprocessed/{file}')

        print("====================================================")

        #binarize(f'smartcrop/{file}') erm

    count1 = 0
    text1 = ''
    for file in os.listdir('preprocessed/'):
        print (file)
        text1 = extract_text(f'preprocessed/{file}')
        #print(text1)
        count1 += 1


