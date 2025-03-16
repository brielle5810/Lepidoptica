# TJS_Project

Our goal is to work with curators Andrei Sourakov, Keith Willmott and Vaughn Shirey to extract and structure collection information from butterfly specimen labels to assist in ongoing digitization efforts. We would be working specifically to parse hand-written and less structured data from specimen labels, building upon Shirey’s work that can already interpret printed and well-structured data. If time permits, we would also plan to automate measuring morphological traits in images, such as abdomen length, thorax length and width, wing length and area. The resulting label data will be uploaded to the FLMNH’s Specify database available online for researchers and museums, and morphological data will contribute to ongoing efforts to score traits for butterflies.

## Install all required packages:
### Method 1: via Powershell terminal + venv:
1. Create a virtual environment (venv) and make sure it's selected for the project (Below is how I know it's activated within Pycharm)
![Screenshot 2025-03-16 125543](readme-images/1.png?raw=true)
2. Open a terminal in flask app directory (where you created a virtual environment)
3. Make sure you're in the right environment (your path will look something like `(venv) PS C:\path\to\your\project>`)
4. Run `pip install -r requirements.txt`
You can do this in the Pycharm Powershell terminal, it's much easier than the 2nd method.
![Screenshot 2025-03-16 125543](readme-images/4.png?raw=true)

### Method 2: via no terminal, only Pycharm + venv:
1. Enter the flask app directory
2. Create a virtual environment (venv) and make sure it's selected for the project
![Screenshot 2025-03-16 125543](readme-images/1.png?raw=true)
4. Open the requirements.txt
5. Select prompt for installing all missing packages. It can be finnicky so repeat until there are no more squiggly lines
![Screenshot 2025-03-16 124950](readme-images/2.png?raw=true)
6. Anything that can't be successfully installed can be installed through the python package manager
![Screenshot 2025-03-16 125751](readme-images/3.png?raw=true)



