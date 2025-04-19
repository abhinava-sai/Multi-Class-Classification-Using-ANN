# Multi-Class-Classification-Using-ANN


Multi-Class Classification Using Artificial Neural Network with Fashion MNIST
 

Abstract:    
This study presents a comprehensive analysis of a multi-class classification model using an Artificial Neural Network (ANN) to classify images from the Fashion MNIST dataset. This dataset contains 70,000 grayscale images across 10 different clothing categories. The proposed model is designed to achieve high accuracy in predicting the class of the clothing items while efficiently handling the complexities of image data.


1.Introduction

1.1 Background
The Fashion MNIST dataset serves as a challenging benchmark for image classification tasks, providing a rich source of data to evaluate machine learning models. With the rise of deep learning, Artificial Neural Networks have become a popular choice for image classification due to their ability to learn complex patterns.

1.2 Motivation
The motivation behind this study includes:
- Evaluating the performance of ANN on multi-class image classification tasks.
- Analyzing the effectiveness of various techniques such as dropout and batch normalization to improve model performance.
- Understanding how the architecture of the ANN impacts classification accuracy.
  
1.3 Paper Structure
This paper is organized as follows:
- Section 2: Data Acquisition and Preprocessing
- Section 3: Model Development
- Section 4: Experimental Results
- Section 5: Analysis and Discussion
- Section 6: Conclusion and Future Work


2. Data Acquisition and Preprocessing

2.1 Dataset Overview
The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images, classified into 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

2.2 Data Loading and Normalization
        
Load and preprocess the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

Normalize the images from 0-255 to 0-1
X_train, X_test = X_train / 255.0, X_test / 255.0

Flatten the data for ANN (samples, features)
X_train_flattened = X_train.reshape(-1, 28   28)
X_test_flattened = X_test.reshape(-1, 28   28)
      
2.3 Data Visualization
Visualizing sample images helps understand the data distribution and identify class characteristics:         
Visualize sample images from Fashion MNIST
plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.xticks([])  
    plt.yticks([])  
    plt.grid(False)  
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.title(class_names[y_train[i]])
plt.show()

      
3.Model Development
   
3.1 ANN Architecture
The model is structured as follows:
- Input layer with 128 neurons and Leaky ReLU activation.
- Two hidden layers, each with 128 neurons, batch normalization, and dropout layers to prevent overfitting.
- Output layer with 10 neurons and softmax activation for multi-class classification.

3.2 Model Compilation
       
Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      
3.3 Model Training
Training the model using the training dataset and validating it on the test dataset, with early stopping implemented to avoid overfitting:
       
Train the model
history = model.fit(X_train_flattened, y_train, 
                    validation_data=(X_test_flattened, y_test), 
                    epochs=50, batch_size=32, callbacks=[early_stopping])
      

4. Experimental Results

4.1 Training and Validation Performance
The following visualizations represent the model's accuracy and loss during training:
          
Visualize training accuracy and loss over epochs
plt.figure(figsize=(14,5))

Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
      
4.2 Model Evaluation
The test accuracy and confusion matrix provide insights into the model’s classification performance:
      
Evaluate the model
test_loss, test_acc = model.evaluate(X_test_flattened, y_test)
print(f'Test Accuracy: {test_acc}')

Confusion matrix visualization
y_pred = model.predict(X_test_flattened)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
      
4.3 Classification Report
A detailed classification report provides metrics like precision, recall, and F1-score for each class:
         
Classification report
report = classification_report(y_test, y_pred_classes, output_dict=True)
df_report = pd.DataFrame(report).transpose()

Plot the precision, recall, and f1-score for each class
plt.figure(figsize=(12,8))
df_report.iloc[:-3, :-1].plot(kind='bar', figsize=(12, 8), colormap='viridis')
plt.title('Precision, Recall, F1-Score per Class')
plt.xticks(range(10), class_names, rotation=45)
plt.show()


5. Analysis and Discussion

5.1 Performance Metrics
The model achieved a commendable test accuracy, showcasing the effectiveness of the ANN architecture. The confusion matrix highlights the model's strengths and weaknesses in predicting specific classes.

5.2 Comparative Analysis
The use of techniques such as batch normalization and dropout contributed to improved model stability and generalization. The model effectively handled class imbalances, as indicated by the classification report.


6. Conclusion and Future Work
This study demonstrates the potential of Artificial Neural Networks in multi-class classification tasks using the Fashion MNIST dataset. The findings underscore the importance of model architecture and regularization techniques.


 Future Work
1. Explore deeper architectures such as Convolutional Neural Networks (CNNs) for enhanced performance.
2. Investigate data augmentation techniques to improve model robustness.
3. Implement hyperparameter tuning to optimize model parameters.
