import os
import numpy as np
import torch

# class MossDataset(object):
#     def __init__(self, directory):
#         self.directory = directory

#         self.files = 

directory = "../gallant/test_result"
files = os.listdir(directory)
print(files)