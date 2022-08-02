# Midterm Report
## Introduction/Background
## Problem Definition
This project is an exploration of machine learning as a tool to convert handwriting into ASCII characters for data processing in the real world. Despite the best efforts of Silicon Valley, not everything is digital. We still utilize real world documents as a means of tracking very important information. This information usually needs to be processed in a database or backed up to safeguard the data. Important usecases for this project include, but are certainly not limited to, the healthcare, pharmecuticals, insurance, and banking industries of the world. Millions upon millions of legal documents are created everyday that need to be processed at rates that only pure automation can make feasible. If these documents are failed to be processed in time or are improperly processed, they can lead to a multitude of critical issues for organizations and individuals. This exploration aims to help safeguard and provide an additional tool of measurement for these kinds of documents.

## Data Collection
The Dataset used in the project is [**NIST Special Database 19**](https://www.nist.gov/srd/nist-special-database-19). The NIST dataset is the most commonly used dataset for handprinted document and character recognition, which includes over 800,000 images of hand-written characters from 3600 writers, with hand-checked classification. The dataset includes digits, upper case English characters, and lower case English Characters. 

Due to the limited computational power the team has available, we only used a subset of the NIST dataset, which includes 8,000 images of hand-written characters that includes 0 - 9 digits, A - Z uppercase English Letters, and a - z lower case letters, which comes to a total of 62 classes. The dataset is split into 80% training and 20% testing using Sklearn’s train_test_split method and subsequently split into images and labels. The dataset is also randomly shuffled. Each image has a dimension of 100 * 100 pixels, with 3 RGB channels. Each pixel is then typecasted to float32, and divided by 255 for normalization. The training and testing labels are categorized in the one-hot encoding format using Keras.

## Methods
### Simple Convolution Neural Networks
In the midterm checkpoint, we applied a simple CNN model on a subset of the data that only contained 0 - 9 digits. We observed an accuracy of 98.84%. 
![Screen%20Shot%202022-07-08%20at%2018 23 37](https://user-images.githubusercontent.com/17306743/179345093-3914ad03-3c17-428c-b78e-8af1785a4128.png)

However, when the same model is applied to the dataset with digits, uppercase letters and lowercase letters, the model underperformed significantly. Therefore, we increases the complexity of our CNN model, with 6 convolution layers. Corresponding measures to combat overfitting is also implemented using dropout and batch normalization. A fully connected layer is applied in the end. The model summary is as follows:
[] insert model summary of simple model here

We implemented Convolution Neural Networks as our main methodology. The model architecture we used is inspired by previous work done by He et al., 2016, and Yousef et al., 2020. Implementation is achieved using Tensorflow’s Keras.

The first convolution is Conv2D, with 32 filters with kernel size 3 * 3.  The activation function we chose is ReLu(Rectified Linear Activation Function), and the input shape is (1, 28, 28), as defined in the data processing step.

A second convolution is then added, with all hyperparameters being the same except that we used 64 filters in order to identify higher-level features. 

Then, a MaxPooling2D layer is added to reduce the dimensionality while also highlighting the maximum value in each feature map.

The result of the convolution is then flattened from 2-D to 1-D and passed into a hidden dense layer, with an input neuron size of 64 and an output neuron size of 10. 

The following picture is the model architecture summary:

![Screen%20Shot%202022-07-08%20at%2018 23 37](https://user-images.githubusercontent.com/17306743/179345093-3914ad03-3c17-428c-b78e-8af1785a4128.png)

The model is then compiled. We chose categorical cross-entropy to be the loss function since this is a classification problem. By comparing the label given by CNN to the target label, we assign 1 for every correct classification and 0 for incorrect classification. We chose Adam for our activation function, and the metric we chose is Keras’ accuracy, which computes the frequency in which the predicted label matches the target label. For Model training, we chose the batch size to be 32 and epochs to be 10. 


The result of the model is measured with accuracy, as mentioned above. We compared the frequency in which the predicted label agrees with the target label. With our model, we have an accuracy of 98.84%. 


It is very encouraging that our current model resulted in high accuracy. However, considering this is only the subset of the dataset that contains only digits from 0 - 9, we should expect a lower accuracy when we incorporate all letters in the English alphabet. This would mean that we might need more convolution layers, and possibly Regularizing methods such as DropOut in our current model to prevent overfitting. 
## References
He, T., Huang, W., Qiao, Y., & Yao, J. (2016). Text-attentional convolutional neural network for scene text detection. *IEEE Transactions on Image Processing*, *25*(6), 2529–2541. https://doi.org/10.1109/tip.2016.2547588 

Yousef, M., Hussain, K. F., & Mohammed, U. S. (2020). Accurate, data-efficient, unconstrained text recognition with convolutional neural networks. *Pattern Recognition*, *108*, 107482. https://doi.org/10.1016/j.patcog.2020.107482
