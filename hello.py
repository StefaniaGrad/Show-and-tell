import os
import io
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from PIL import Image
import nltk
nltk.download('punkt')
import itertools

print("A")