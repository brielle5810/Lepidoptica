"""Import the various Python libraries we're gonna need
- [Cv2](https://konfuzio.com/en/cv2/#:~:text=One%20of%20these%20libraries%20that,image%20and%20video%20processing%20functions.): Main module of openCV, allows us to work with and process images
- [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html): Allows us to work with mathematical functions (also arrays!)
- [Matplotlib/pyplot](https://matplotlib.org/cheatsheets/_images/cheatsheets-1.png): Used for data visualization, specifically static, animated, and interactive plots

Tutorials:
*   [Cv2](https://konfuzio.com/en/cv2/#:~:text=One%20of%20these%20libraries%20that,image%20and%20video%20processing%20functions.)
*   [NumPy](https://www.w3schools.com/python/numpy/numpy_intro.asp)
*   [Matplotlib](https://www.w3schools.com/python/matplotlib_intro.asp)
"""

### STEP 1: Symbol contour box ###

import cv2
import numpy as np
from matplotlib import pyplot
import os

### Various functions that we're going to use to actually read the image ###

# Scales a symbol within an image
# Input: 
	# The input image (image)
	# The array of points in the image (contour)
	# The new dimensions of the image as an array (newSize) [9x9]
def scaleSymbol(image, contour, newSize):
	# Get the bounding rect of the contour:
	# cv2.boundingRect takes in an array of points, and returns:
	#	- The (x,y) coordinates of the top-left corner
	#	- The width and height of the bounding rectangle
	[x, y, width, height] = cv2.boundingRect(contour)

	# Now we want the rectangle as it's own image
	symbol = image[y : y + height, x : x + width]
	newWidth = newSize[0]
	newHeight = newSize[1]

	# If the image is scaled too small
	if 0 in symbol.shape:
		return None

	# Resize the symbol
	scaled = cv2.resize(symbol, (newWidth, newHeight), interpolation=cv2.INTER_AREA)

	# Return the scaled symbol
	return scaled

# Copy and paste source image onto dest
def copyImage(source, dest, x, y):
	# Get the height (.shape[0]) and width (.shape[1]) of the original image
	height, width = source.shape[0], source.shape[1]

	# Blit source onto dest
	dest[y : y + height, x : x + width] = source
	return dest

# Get the contours from an image
def getContours(image):
	# Get grayscale image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Find threshed automatically, good for symbol detection
	ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Morphological operation to ensure smaller portions are part of bigger character
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	thresh = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

	# Only find external contours, characters (probably) won't be nested
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	return contours

# Main program to test
if __name__ == '__main__':
	from sys import argv

	# argv check / usage info
	if len(argv) < 2:
		print("No image path specified, using default image.")
		# get the path of the current script
		currentDir = os.path.dirname(os.path.abspath(__file__))
		print(currentDir)

		# combine the path to the current directory with the file name to form a complete filepath.
		imagePath = os.path.join(currentDir, "0.png")
		print(imagePath)

		# Check if the default image exists.
		if not os.path.exists(imagePath):
			print("Default image '0.png' not found. Please ensure it is in the same directory as the script.")
			exit(1)
		image = cv2.imread(imagePath)
	else:
		image = cv2.imread(argv[1])

  # Check if image loading was successful.
	if image is None:
		print("ERROR: Image not loaded")
		exit(1)

	# Scale all symbols down to 9x9
	newWidth, newHeight = 9, 9

	# Get the contours
	contours = getContours(image)
	for contour in contours:
		x, y, width, height = cv2.boundingRect(contour)
		symbol = scaleSymbol(image, contour, (newWidth, newHeight))

		# Clear the area and blit the resized symbol
		# Draw a rectangle using cv2.rectangle:
			# cv2.rectangle(image, startpoint [tuple], endpoint [tuple], color [BGR], thickness)
		cv2.rectangle(image, (x, y), (x + width, y + height), 255, 2)
		copyImage(symbol, image, x, y)

	# Show the image
	pyplot.imshow(image)
	pyplot.show()

	print("All done")