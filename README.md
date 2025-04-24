# Lepidoptica

Our project is focused on aiding the McGuire Center for Lepidoptera & Biodiversity with digitization efforts for their Lepidoptera Specify collection. The McGuire Center for Lepidoptera and Biodiversity is the largest collections-based research and education center in the world focused on butterflies and moths. These collections are used by researchers and students worldwide to track biodiversity patterns and how they relate to topics such as climate change, agricultural pests, evolution, and conservation. To make broader use of the millions of butterfly specimens in world collections, curators are recording these data from labels and making them publicly available for research. It takes a large amount of time for researchers to parse data from labels in specimen photographs and manually enter it into Lepidoptera Specify Collection database (https://specifyportal.floridamuseum.ufl.edu/leps/).

We worked with curators Keith Willmott and Vaughn Shirey to as advisors in this project. Our goal was, by the end of the semester, to create an application to extract and parse metadata from butterfly specimen labels; specifically, to interpret inconsistently structured, handwritten information and separate the data into predetermined fields. Our main objectives were to create a program that accepts batches of image files, fine-tune a model to better recognize sex-symbols as well as typed and handwritten labels, predict the accuracy of the interpretation, flag potential errors, and parse the data into specific data fields. The resulting label data will be downloaded in csv format to be uploaded to the FLMNH’s Specify database available online for researchers and museums.

This project will be continued to be worked on in some form in the museum ongoing digitization process. We hope we have given all future Lepidoptica developers a good basis to work on, and we are excited to see how it evolves in the future!


## Directions
There are two main project directories, listed below. Each of these projects have their own environments, required packages, and readmes with instructions on how to run them. Learn [how to install all the required packages](https://github.com/brielle5810/Lepidoptica/blob/main/how_to_install_packages.md).

### [flask-app/](https://github.com/brielle5810/Lepidoptica/tree/main/flask-app) : web-app project
* Run the web-app through `app.py`
* `Set_up_model.ipynb` is used to copy the custom model into the right local directories so that they can be used by the app. 

### [Nanonet-and-Training/](https://github.com/brielle5810/Lepidoptica/tree/main/Nanonet-and-Training) : nano-net and training project
* [How to Train your Model](https://github.com/brielle5810/Lepidoptica/blob/main/Nanonet-and-Training/How%20to%20Train%20your%20Model.md) is a comprehensive guide on how the training for this model was done. Use alongside `lepidoptica-custom-model.zip`, which contains all the mentioned files
* `main.py` and `box-nanonets.py` is used to prepare data for training. 
