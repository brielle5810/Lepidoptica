import cv2
import numpy as np
import pytesseract
import os
import PIL
from PIL import Image, ImageEnhance

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

def preprocess(image_path):
    img = cv2.imread(image_path, 0)
    ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

    imgf = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('preprocessed/' + image_path.split('/')[1], imgf)


def binarize(image_path):

    image = cv2.imread(image_path)

    # Normalize image
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.fastNlMeansDenoisingColored(opening, None, 10, 10, 7, 15)

    gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Thinning and Skeletonization
    kernel = np.ones((3, 3), np.uint8) # 5, 5 -> thicker idk
    erosion = cv2.erode(gray, kernel, iterations=1)
    #dilation = cv2.dilate(erosion, kernel, iterations=1)
    #show image
    cv2.imwrite('preprocessed/' + image_path.split('/')[1], erosion)

def extract_text(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 4')
    #image_to_data(im, lang='eng', config=psm)
    return text

if __name__ == '__main__':
    # why so many diff directories?
    # to compare different stages of the image processing and processing techniques
    # this is just a test so... chill
    for file in os.listdir('uncropped/'):
        rotate_180(f'uncropped/{file}')

    for file in os.listdir('cut_turned/'):
        binarize(f'cut_turned/{file}')

    count1 = 0
    text1 = ''
    for file in os.listdir('preprocessed/'):
        while count1 < 1:
            text1 = extract_text(f'preprocessed/{file}')
            count1 += 1

    text2 = ''
    count2 = 0
    for file in os.listdir('cut_turned/'):
        while count2 < 1:
            text2 = extract_text(f'cut_turned/{file}')
            count2 += 1
    print(text1)
    #print(text2)
