from tkinter.filedialog import askdirectory
import os
import pandas as pd

# import sys
# from io import StringIO

red = [1.0, 0.0, 0.0]
gray = [0.5, 0.5, 0.5]

thresholds = [3, 9.33]

features = {str(thresholds[0]): {"Right": pd.DataFrame(), "Left": pd.DataFrame()},
            str(thresholds[1]): {"Right": pd.DataFrame(), "Left": pd.DataFrame()}}
patch_sets = {"Right": {"positive": pd.DataFrame(), "negative": pd.DataFrame()}, "Left": {"positive": pd.DataFrame(),
                                                                                          "negative": pd.DataFrame()}}
col = ["Right " + str(thresholds[0]) + "mm", "Left " + str(thresholds[0]) + "mm", "Right " + str(thresholds[1]) + "mm",
       "Left " + str(thresholds[1]) + "mm"]
decision_tree = pd.DataFrame(columns=col)

# Prompting user to select the parent folder
path = askdirectory(title='Select Folder')

# Loops through the subfolders in the parent folder
for subdir, dirs, files in os.walk(path):
    for sub_folder in dirs:
        # print('\nfolder: ' + sub_folder)
        filepath = subdir + os.sep + sub_folder + os.sep
        x = []
        for threshold, sides in features.items():
            for side, data_set in sides.items():
                # read features .csv file and split into positive and negative datasets
                features_csv = pd.read_csv(
                    filepath + 'Features-Py-area-' + side + '-' + sub_folder + '-' + str(threshold) + "mm.csv",
                    header=0)
                # decision_tree.loc[:, side + ' ' + threshold + 'mm'] = features_csv.loc[:, 'Curve Class (tree)'][0]
                x.append(features_csv.loc[:, 'Curve Class (tree)'][0])
        row = pd.DataFrame([x], index=[sub_folder], columns=col)
        # print(row)
        decision_tree = pd.concat([decision_tree, row])
print(decision_tree)
decision_tree.to_csv(path + os.sep + 'decision tree results.csv', index=True)
