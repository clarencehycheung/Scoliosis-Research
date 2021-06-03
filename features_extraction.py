# -------------------------Import Libraries------------------------------#
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
import os

# Prompting user to select the parent folder
path = askdirectory(title='Select Folder')

# Loops through the subfolders in the parent folder and finds the EDM-1.csv files
for subdir, dirs, files in os.walk(path):
  for file in files:
    filepath = subdir + os.sep + file
    if filepath.endswith("EDM-1.csv"):
      print(filepath)
