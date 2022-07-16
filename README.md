# Midterm Report
The Dataset used in the project is [****NIST Special Database 19****](https://www.nist.gov/srd/nist-special-database-19). The NIST dataset is the most commonly used dataset for handprinted document and character recognition, which includes over 800,000 images of hand-written characters from 3600 writers, with hand-checked classification. The dataset includes digits, upper case English characters, and lower case English Characters. All the images provided by NIST are in 28 pixels * 28 pixels format and grayscaled.

The dataset chosen is a subset of the NIST database. The subset contains over 600,000 hand-written digits(0-9). The rows are each individual image (character), and the columns are each pixel of the image(28 * 28 = 785 columns). The dataset is split into 80% training and 20% testing using Sklearn’s train_test_split method and subsequently split into images and labels. The dataset is also randomly shuffled. 

We then transformed the data from flattened 784 pixels to 2-D shape images 28 * 28 pixels. This is done using NumPy's reshape method, where the training images and testing images are reshaped into (1, 28, 28), with the 1 being the grayscaled color channel, and the two 28s being the image column pixel size and image row pixel size prospectively. Each pixel is then typecasted to float32, and divided by 255 for normalization. The training and testing labels are categorized in the one-hot encoding format using Keras.


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
