import splitfolders
import os

splitfolders.ratio(f"{os.getcwd()}/data/train/", output=f"{os.getcwd()}/data/split_train/", seed=1337, ratio=(.8, .2), group_prefix=None) # default values
