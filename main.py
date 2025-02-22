import cv2
import numpy as np
import pytesseract
import os
import PIL
from PIL import Image, ImageEnhance
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
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    width = image.shape[1]

    fraction = 2 / 5
    new_width = int(width * fraction)
    left_half = image[:, :new_width]

    cv2.imwrite(after_path + image_path.split('/')[1], left_half)

def rotate_180(image_path):
    after_path = 'rotated/'
    image = Image.open(image_path)
    image = image.rotate(180)
    image.save(after_path + image_path.split('/')[1])
    crop_left_half(after_path + image_path.split('/')[1])

# def preprocess(image_path):
#     img = cv2.imread(image_path, 0)
#     ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
#
#     imgf = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     cv2.imwrite('preprocessed/' + image_path.split('/')[1], imgf)

#
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
    cv2.imwrite('preprocessed/' + image_path.split('/')[1], erosion)




def extract_text(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 4')
    #image_to_data(im, lang='eng', config=psm)
    print(text)
    return text

# def extract_text_keras(image_path): KERAS BLOWS
#     pipeline = keras_ocr.pipeline.Pipeline()
#     images = [keras_ocr.tools.read(image_path)]
#     prediction_groups = pipeline.recognize(images)
#     text = ''
#     for predictions in prediction_groups:
#         for text in predictions:
#             text += text
#     return text

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
def apply_smart_crop(image_path, crop_data):
    # Open the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # we are using full height to not cut off smaller labels, as is done when everything is left to SmartCrop
    x, y, width, height = crop_data['x'], crop_data['y'], crop_data['width'], img_height

    # im1 = im.crop((leftstart, pixelsfromtop, howfarright, howfardown))
    cropped_image = image.crop((x, 0, x + width, height))

    output_dir = "smartcrop"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    cropped_image.save(os.path.join(output_dir, filename))

def testSmartCrop(image_path):
    image = Image.open(image_path)
    sc = smartcrop.SmartCrop()
    result = sc.crop(image, 100, 100)
    output_dir = 'smartcrop'
    os.makedirs(output_dir, exist_ok=True)

    print(result['top_crop'])

    apply_smart_crop(image_path, result['top_crop'])




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

        #binarize(f'smartcrop/{file}')

    count1 = 0
    text1 = ''
    for file in os.listdir('preprocessed/'):
        print (file)
        text1 = extract_text(f'preprocessed/{file}')
        #print(text1)
        count1 += 1

    # text2 = ''
    # count2 = 0
    # for file in os.listdir('preprocessed/'):
    #     while count2 < 1:
    #         text2 = extract_text(f'preprocessed/{file}')
    #         count2 += 1
    #print(text1)
    #print(text2)
