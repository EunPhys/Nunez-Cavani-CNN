# Nunez-Cavani-CNN
Convolutional neural network image classifier. Optimised to decipher between two of footballs biggest GOATS, Darwin Nunez and Edinson Cavani.

DATA FORMAT:

Train Data Shape: (1851, 150, 150, 3)
Train Labels Shape: (1851,)
Test Data Shape: (200, 150, 150, 3)
Test Labels Shape: (200,)

CNN MODEL STRUCTURE: 

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 128)       73856     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 36, 36, 128)      0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 256)       295168    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 17, 17, 256)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 512)       1180160   
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 512)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 dropout (Dropout)           (None, 25088)             0         
                                                                 
 dense (Dense)               (None, 512)               12845568  
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 14,397,057
Trainable params: 14,397,057
Non-trainable params: 0
_________________________________________________________________

TRAINING EPOCHS:

Epoch 1/10
47/47 [==============================] - 68s 1s/step - loss: 0.7232 - accuracy: 0.5946 - val_loss: 0.8840 - val_accuracy: 0.0000e+00
Epoch 2/10
47/47 [==============================] - 68s 1s/step - loss: 0.6721 - accuracy: 0.6061 - val_loss: 0.9562 - val_accuracy: 0.0054
Epoch 3/10
47/47 [==============================] - 66s 1s/step - loss: 0.6672 - accuracy: 0.6054 - val_loss: 0.6581 - val_accuracy: 0.6442
Epoch 4/10
47/47 [==============================] - 66s 1s/step - loss: 0.6647 - accuracy: 0.6250 - val_loss: 1.1434 - val_accuracy: 0.0027
Epoch 5/10
47/47 [==============================] - 66s 1s/step - loss: 0.5757 - accuracy: 0.7122 - val_loss: 0.8705 - val_accuracy: 0.5040
Epoch 6/10
47/47 [==============================] - 66s 1s/step - loss: 0.4689 - accuracy: 0.7784 - val_loss: 0.6468 - val_accuracy: 0.7035
Epoch 7/10
47/47 [==============================] - 66s 1s/step - loss: 0.4251 - accuracy: 0.8176 - val_loss: 0.9935 - val_accuracy: 0.5472
Epoch 8/10
47/47 [==============================] - 66s 1s/step - loss: 0.4077 - accuracy: 0.8250 - val_loss: 0.3853 - val_accuracy: 0.8032
Epoch 9/10
47/47 [==============================] - 66s 1s/step - loss: 0.3392 - accuracy: 0.8480 - val_loss: 0.1937 - val_accuracy: 0.9191
Epoch 10/10
47/47 [==============================] - 66s 1s/step - loss: 0.3381 - accuracy: 0.8486 - val_loss: 0.5162 - val_accuracy: 0.7574

TRAINING AND TEST SUMMARY

Training Loss: 0.2644
Training Accuracy: 0.8865

Test Loss: 0.6664938926696777
Test Accuracy: 0.7250000238418579

ADDITIONAL TEST METRICS

Accuracy: 0.725
Precision: 0.6956521739130435
Recall: 0.8
F1 Score: 0.7441860465116279
