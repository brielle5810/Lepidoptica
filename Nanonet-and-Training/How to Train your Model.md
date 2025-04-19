# How to Train your Model

Through this guide, we're hoping you'll be well equipped to continue (or even restart) fine tuning easyocr. Mainly, I followed what was outlined in the EasyOCR github for training a [custom recognition model](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md). They have a [custom trainer folder](https://github.com/JaidedAI/EasyOCR/tree/master/trainer) in their repository, modelled after the [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) repository, which I downloaded to work on top of.

Before I get into the steps taken to actually train the model, I want to outline the important files, directories, and what we did to train our latest model. This should set the stage, so you understand how everything fits together.

## Directories
This is a map of our training directory, `lepidoptica-custom-model/`. Forewarning: everything has been adjusted to work in Google Colab. So for this to work, you need to unzip and upload everything to Google Drive. There is a default directory, but that can be changed within code (which I will note if necessary). Anything not mentioned is still important, just not enough to go into further details with. 

### :butterfly: data_organization

`nanodata_all/` should be a directory within here where you include all the original nanonet data (images, each with a corresponding txt file with the same name, containing the label [fig.1]).

#### > `NanoNet-To-Data.ipynb`
This program is responsible for taking the nanonet data in `nanodata_all/` and turning it into the format that the trainer expects ( `fine_tuning_data/` with two sub-directories: `nanonet_train/` and `nanonet_val/`. Each sub-directory will have the images, but only one collective `labels.csv` file, containing all the image paths and their label). `fine_tuning_data/` should be moved into `all_data/` before training

*NOTE: Default path is set to `/content/drive/MyDrive/Colab Notebooks/data_folder/tjs-butterfly-imagetotext/lepidoptica-custom-model/data_organization/`. Set to `[your drive path]/lepidoptica-custom-model/data_organization/`*

![fig.1](howtotrain-images/1.png?raw=true “fig.1: Images in original Nanonet format”)
![fig.2](howtotrain-images/2.png?raw=true “fig.2: Images after being read through NanoNet-To-Data.ipynb”)

### :butterfly: config_files
This important folder contains `en_filtered_config.yaml`, a file that controls training configurations. Editing this file will change the batch size, allowed characters, data locations, and more. Currently, none of the files in here are named `en_filtered_config.yaml`. Instead, they're named for which fine_tuned model they were used for, just for version history. Be aware that if you'll train your own version, you'll need to name this file `en_filtered_config.yaml`, or change this name in `trainer.ipynb`. 

#### > `en_filtered_config.yaml`
I will go over the most important parameters I had to change and what they do. Other parameters are still important, but haven't been changed from the original `en_filtered_config.yaml` provided by EasyOcr's repository.

**symbol:** Added symbols `♀♂°`
**lang_char:** Added characters `ÇÑÜÉÁĄÓÃÍÂÚÊÔÄÕçñüéáąóãíâúêôäõ`
**experiment_name:** This is the name given to your model directory when outputted to saved_models. In this verbal example, we'll set it to `fine_tuning`
**train_data:** Location of data directory. In our verbal example, this would be `all_data/fine_tuning_data`
**valid_data:** Location of validation data directory. In our verbal example, this would be `all_data/fine_tuning_data/nanodata_val`
**saved_model:** Location of model that we're building on top of. For `fine_tuning1`, this was `saved_models/english_g2.pth`. For `fine_tuning2`, this was `saved_models/fine_tuning1.pth`.
**select_data:** Location of training data directory, within train_data. In our verbal example, this would be `nanodata_train`

### :butterfly: all_data
This is the data that was used to train my model attempts. As you can see, each directory has a `nanodata_train/` and `nanodata_val/` subdirectory. We would move `fine_tuning_data/`, created by `NanoNet-To-Data.ipynb`, here for organization purposes. 

### :butterfly: saved_models
When the trainer runs, it will create the new model directory here. Each model directory will save a model after each epoch. It will also save the `best_accuracy.pth` and `best_norm_ED.pth`. Finally, there are three log files to show the history of the training and testing. In our verbal example, this directory will be called `fine_tuning` (see `experiment_name` in the config file).

Any model you want to include as a base model to build off of should be included in the root of this directory, as the following models are:

#### > english_g2.pth
This is the model that EasyOcr uses. It was used as the base saved_model for the first rounds of fine_tuning.
#### > fine_tuning1.pth
This model was created after the first round of fine-tuning (`saved_models/fine-tuning1_attempt3/best_accuracy.pth`, renamed to `fine_tuning1.pth`). It was used for the second round of fine-tuning.

### :butterfly: trainer.ipynb
This is the file you should run to do the training. It will read the config file then run `train_dtr.py`. The program runs best when run with a GPU, so Google Colab Pro is recommended. When debugging or making changes to dependent files (`train_dtr.py`, `test_dtr.py`, etc.), you'll need to reload them. There is a few lines before the config function to do so.  If you change anything in config, you'll need to restart your session to reload the file.

*NOTE: Default path is set to `/content/drive/MyDrive/Colab Notebooks/data_folder/tjs-butterfly-imagetotext/lepidoptica-custom-model/`. Set to `[your drive path]/lepidoptica-custom-model/`*

### :butterfly: train_dtr.py
Every other python file is called here. Unfortunately google drive doesn't provide an easy way to edit these files, but Google Colab provides a very simple editor, you'll just need to pull up the file after sifting through the directory list.

## How to Train:

1. Make Nanonet images: Visit [Nanonets.com](https://nanonets.com/), make an account. Upload raw butterfly images. For each word/conneted text without spaces between, make a box and transcribe. You can use the same default container [fig.3]. You will then need to change the Review status of all the images to Approved, or else they won't be considered later. 

![fig.3](howtotrain-images/3.png?raw=true “fig.3: Example of text parsing in Nanonets.com”)

2. Make note of the Nanonet Model ID (found in Workflow Settings) and API key (found in Account Info)

---

---> in local repository `Nanonet-and-Training/`

3. Download all the raw butterfly images documented in your Nanonet account (I have directory `example/` as an example of what it looks like)
4. Rename the raw_images in `main.py` to the correct directory, then run that program. This should crop and preprocess your images, saving the results in `preproccessed/`
5. Input your Nanonet Model ID and API key into `box-nanonets.py`, then run that program. It should create `nanodata_all/`, which will have the image with corresponding label text file as seen previously in [fig. 1].

More information on how to run programs in this virtual environment is in the `README.md` of that directory.

---


---> go to google drive

6. Upload `nanodata_all/` into data_organization
7. Run `NanoNet-To-Data.ipynb`
8. Move the newly created `fine_tuning_data/` into `all_data/`
9. Make sure `en_filtered_config.yaml` is ready and up to date.
10. Run `trainer.ipynb`.
11. Once training is done, you will find the new `fine_tuning/` directory (or whatever name you gave it in config). Within there, you will find the `best_accuracy.pth`.
12. Make a copy of `best_accuracy.pth`, rename to `fine_tuning.pth` (or any other specific name you'd like). 
13. Create two more files, `fine_tuning.py` and `fine_tuning.yaml` (there are examples in `saved_models/fine_tuning1_attempt3` and `saved_models/fine_tuning2_attempt2`). Really I made these based on the EasyOcr provided examples, but I edited the character list. Always review the settings to make sure there's nothing you need to alter.
14. Download `fine_tuning.pth`, `fine_tuning.py`, and `fine_tuning.yaml`, put them in local repository directory `flask-app/models/`

---

---> in local repository `flask-app/`

15. Run `set_up_model.ipynb` to add the necessary files to your local EasyOcr directory
16. Change the name of the model used in reader in `app.py`. For a model named `fine_tuning.pth`, it should look like this: `reader = easyocr.Reader(['en'], gpu=False, recog_network="fine_tuning")`
17. That's all folks!

## History
We actually trained a few different models (all found in `saved_models/`), but these two are the most successful ones. Our program now uses `fine_tuning2.pth`.

### First saved_model - fine_tuning1_attempt3 (fine_tuning1.pth)
- Built on top of saved_model `english_g2.pth`, which is what is used as the easyocr English model
- Added new characters to the character list (97 classes to 130 classes)
- Reduced batch size from 32 to 8
- Adjusted `trainer.py` and the rest of the files to work with google drive
- Data = 3533 images (2827 for training, 706 for validation)
- Accuracy: 73.371, Norm ED: 0.9336, Training Loss: 0, Validation Loss: 0.63248

### Second saved_model - fine_tuning2_attempt2 (fine_tuning2.pth)
- Built on top of `fine_tuning1.pth`, my previously saved model
- Experimented with preprocessing to try and get better metrics
- Data = 5710 images (4569 for training, 1141 for validation)
- Accuracy: 83.523, Norm ED: 0.9349, Training Loss: 0, Validation Loss: 0.59990

**Total images used for training: 9243**
