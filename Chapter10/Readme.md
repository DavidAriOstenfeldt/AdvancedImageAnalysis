# Material for Chapter 10

The notebook for the exercise in Chapter 10 is available from here:
[Notebook for mini U-net](https://github.com/vedranaa/teaching-notebooks/blob/main/02506_week10_MiniUnet.ipynb)

You can also open it directly in Google Colab from here:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vedranaa/teaching-notebooks/blob/main/02506_week10_MiniUnet.ipynb)


## Run the notebook on DTU Gbar

It is possible to run the notebook on Gbar on an interactive note. This is done by setting up a Python environment. First you should log onto an interactive GPU node such as `voltash` by running the following command on the command line:

`voltash -X`

To set up a Python environment, you can use the script `env02506.sh`. You should place the file `env02506.sh` in a folder on the Gbar and then run the command line:

`source env02506.sh`

The first time you run this, the script will create a new python environment called `env02506` and activate it.

Then you should install the packages needed which include:

`pip install torch torchvision`

`pip install Pillow`

`pip install notebook`

Now you are good to go. You can navigate to the folder that you wish to store the code, download the notebook from the link above, and open a jupyter notebook by typing the following in the command line:

`jupyter-notebook`
