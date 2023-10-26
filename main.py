import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Conv2D,Dense,Flatten,Dropout
from keras.models import Model

fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
X_train,X_test = X_train/255.0,X_test/255.0
print("X_train.shape:",X_train.shape)

X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)
print(X_train.shape)

K = len(set(y_train))
print("Number of classes:",K)

from keras.api._v2.keras import activations
i = Input(shape = X_train[0].shape)
x = Conv2D(32,(3,3),strides = 2,activation = 'relu')(i)
x = Conv2D(64,(3,3),strides = 2,activation = 'relu')(x)
x = Conv2D(128,(3,3),strides = 2,activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512,activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(K,activation = 'softmax')(x)

model = Model(i,x)


