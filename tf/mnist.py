#!/usr/bin/python
# coding=utf-8

import math

# from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv('/Users/giggle/Downloads/california_housing_train.csv', sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]
tmp = california_housing_dataframe["total_rooms"]
print(type(california_housing_dataframe))
print type(my_feature)
print type(tmp)

