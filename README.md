##Foliar Disease Prediction using CNN
This repository contains the code for a machine learning project aimed at predicting foliar diseases using a convolutional neural network (CNN). The developed deep learning model demonstrates high accuracy in classifying various foliar diseases.

#Model Architecture
The CNN model consists of a total of 6 layers, including 2 Conv2D layers, 2 MaxPooling2D layers, and 2 Dense layers. This architecture has been carefully designed to effectively learn and extract relevant features from the input images.

#Model Performance
After training the model on the available dataset, the following accuracies were achieved:

Training Accuracy: 99.83%
Validation Accuracy: 100%
Testing Accuracy: 98.88%
These high accuracies indicate the model's proficiency in predicting foliar diseases accurately.

#Dependencies
The implementation of this project relies on the following Python libraries:

Keras: for building and training deep learning models
NumPy: for efficient numerical computations and array operations
Pandas: for data manipulation and analysis
Matplotlib: for data visualization and plotting
OpenCV: for image processing tasks
Ensure that these libraries are installed in your Python environment before running the code.

#Usage
Clone this repository to your local machine.
Install the required dependencies by running pip install -r requirements.txt.
Prepare your dataset in a compatible format. Refer to the project documentation for specific details on the dataset format and structure.
Execute the Python script foliar_disease_prediction.py to train the CNN model and evaluate its performance.
After training, the model can be used for predictions on new foliar disease images. Refer to the example code provided in predict.py to see how to use the trained model for inference.

#Additional Notes
The dataset used for training and testing the model is not included in this repository. Please ensure that you have access to an appropriate dataset before attempting to run the code.
Feel free to explore and modify the code to suit your specific requirements or experiment with different CNN architectures and hyperparameters.


Happy coding!
