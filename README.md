# Project_Waste_Classification_Using_ML
# Waste-classification-Model-using-Machine-Learning
Overview
This project uses a Convolutional Neural Network (CNN) to classify waste into two categories: Organic and Recyclable. The model is trained on a dataset of waste images, with data augmentation applied to enhance the generalization of the model. After training, the model can predict the type of waste when provided with an image, improving waste management efficiency.

Dataset
The dataset is divided into two directories:

TRAIN: Contains images of organic and recyclable waste for training the model.
TEST: Contains images for evaluating the model's performance.
Each directory has subfolders for each category (Organic and Recyclable). Images are resized to 150x150 pixels before being fed into the model.

Project Structure
plaintext
Copy code
waste_classification/
│
├── waste_classification_model_recyclable_organic.h5   # Saved trained model
├── train/                                             # Training dataset
│   ├── O/                                             # Organic images
│   └── R/                                             # Recyclable images
├── test/                                              # Test dataset
│   ├── O/                                             # Organic test images
│   └── R/                                             # Recyclable test images
├── waste_classification.py                            # Model training and evaluation script
└── README.md                                          # Project documentation

Requirements
Install the necessary libraries using the following command:
  pip install -r requirements.txt

requirements.txt:
  tensorflow
  numpy
  Pillow
How to Run the Project
Train the Model: The script waste_classification.py contains the code for training the CNN model on the dataset.

Run the script to train the model:
  python waste_classification.py

Evaluate the Model: After training, the model is evaluated on the test dataset. You can monitor accuracy and loss during training.

Predict Waste Type: Use the following code to predict the waste type from an image:

  waste_type, confidence = predict_waste_type('path_to_image.jpg')
  print(f'Predicted Waste Type: {waste_type}, Confidence: {confidence}')
  
Model Summary
Model Type: CNN (Convolutional Neural Network)
Input Shape: (150, 150, 3)
Number of Classes: 2 (Organic, Recyclable)
Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metric: Accuracy
Example Prediction
  Predicted Waste Type: Organic with confidence: 96.47%
