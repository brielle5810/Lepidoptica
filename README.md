# Lepidoptica

Our project is focused on aiding the McGuire Center for Lepidoptera & Biodiversity with digitization efforts for their Lepidoptera Specify collection. The McGuire Center for Lepidoptera and Biodiversity is the largest collections-based research and education center in the world focused on butterflies and moths. These collections are used by researchers and students worldwide to track biodiversity patterns and how they relate to topics such as climate change, agricultural pests, evolution, and conservation. To make broader use of the millions of butterfly specimens in world collections, curators are recording these data from labels and making them publicly available for research. It takes a large amount of time for researchers to parse data from labels in specimen photographs and manually enter it into Lepidoptera Specify Collection database (https://specifyportal.floridamuseum.ufl.edu/leps/).

We worked with curators Keith Willmott and Vaughn Shirey to as advisors in this prokect. Our goal was, by the end of the semester, to create an application to extract and parse metadata from butterfly specimen labels; specifically, to interpret inconsistently structured, handwritten information and separate the data into predetermined fields. Our main objectives were to create a program that accepts batches of image files, fine-tune a model to better recognize sex-symbols as well as typed and handwritten labels, predict the accuracy of the interpretation, flag potential errors, and parse the data into specific data fields. The resulting label data will be downloaded in csv format to be uploaded to the FLMNH’s Specify database available online for researchers and museums.

This project will be continued to be worked on in some form in the museum ongoing digitization process. We hope we have given all future Lepidoptica developers a good basis to work on, and we are excited to see how it evolves in the future!

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



