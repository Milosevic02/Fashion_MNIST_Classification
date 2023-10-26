# Fashion MNIST Image Classification

This project focuses on classifying fashion images using the Fashion MNIST dataset. The goal is to build a Convolutional Neural Network (CNN) that can accurately identify different fashion items based on the available data.

## Dataset

The dataset used for this project is the Fashion MNIST dataset. It consists of grayscale images of various fashion items, including T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

## Model Architecture

The CNN model consists of several layers:

1. Input Layer: The input layer receives the image data with a shape of (28, 28, 1) since the images are grayscale.
2. Convolutional Layers: Three convolutional layers with 32, 64, and 128 filters respectively, using a 3x3 kernel and ReLU activation function.
3. Flatten Layer: This layer flattens the output of the last convolutional layer to be fed into the fully connected layers.
4. Dropout Layers: Two dropout layers with a dropout rate of 20% to reduce overfitting.
5. Dense Layers: Two dense layers with 512 units and ReLU activation function.
6. Output Layer: The output layer with 10 units and a softmax activation function to classify the input into one of the 10 fashion categories.

## Model Training

The model is trained using the Adam optimizer and the sparse categorical cross-entropy loss function. The training is done for 15 epochs on the training data.

## Results

The model achieves an accuracy of approximately 93% on the training dataset and 89% on the test dataset. The loss and accuracy plots for both training and validation are visualized to show the training progress.

## Confusion Matrix

The confusion matrix is used to evaluate the model's performance. It shows the count of true positive, false positive, true negative, and false negative predictions for each class.

## Misclassified Samples

A few misclassified samples are randomly selected and displayed along with their true and predicted labels to get an insight into the model's performance.

## Classes

The classes corresponding to the labels are as follows:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

Feel free to explore and modify the code to further improve the model's accuracy or experiment with different architectures and hyperparameters. Happy coding!
 
