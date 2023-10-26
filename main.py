import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Conv2D,Dense,Flatten,Dropout
from keras.models import Model

fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
X_train,X_test = X_train/255.0,X_test/255.0
print("X_train.shape:",X_train.shape)
