## Install all required packages using Requirements.txt:

First, make sure you've navigated to either [flask-app/](https://github.com/brielle5810/Lepidoptica/tree/main/flask-app) or [Nanonet-and-Training/](https://github.com/brielle5810/Lepidoptica/tree/main/Nanonet-and-Training). These are two seperate projects, each with their own envirionments and corresponding requirements.txt

### Method 1: via Powershell terminal + venv:
1. Create a virtual environment (venv) with python 3.12 and make sure it's selected for the project (Below is how I know it's activated within Pycharm)
![Screenshot 2025-03-16 125543](readme-images/1.png?raw=true)
2. Open a terminal in the environment directory (Make sure you're in the right environment. Your path will look something like `(venv) PS C:\path\to\your\project>`)
3. Run `pip install -r requirements.txt`
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
