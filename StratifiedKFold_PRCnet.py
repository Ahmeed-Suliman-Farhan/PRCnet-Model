import os
import math
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
from sklearn import metrics
from livelossplot.inputs.keras import PlotLossesCallback
 

 
def load_data(dir_path):
    
#    Load resized images as np.arrays to workspace

    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                   # img = cv2.imread(dir_path + path + '/' + file)
                
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels



X_data, y_data, labels = load_data('archive51/')




def save_new_images(x_set, y_set ,folder_name):
    i = 0
    print(folder_name)
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+ 'glioma/'+str(i)+'.jpg', img)
        if imclass == 1:
            cv2.imwrite(folder_name+'meningioma/'+str(i)+'.jpg', img)
        if imclass == 2:
            cv2.imwrite(folder_name+'notumor/'+str(i)+'.jpg', img)
        if imclass == 3:
            cv2.imwrite(folder_name+'pituitary/'+str(i)+'.jpg', img)
        i += 1



from sklearn.model_selection import StratifiedKFold
# Instantiate the cross validator
skf = StratifiedKFold(n_splits=5 , random_state=43 , shuffle=True)
# Loop through the indices the split() method returns
index=1
for  train_indices, val_indices  in skf.split(X_data,y_data):
    
    print ("Training on fold " + str(index) + "/10...")
    # Generate batches from indices
    xtrain, xval = X_data[train_indices], X_data[val_indices]
    ytrain, yval = y_data[train_indices], y_data[val_indices]
    print(val_indices)
    save_new_images(xtrain, ytrain,folder_name='Data_cross_7023/fold_' + str(index) + '/train/')
    save_new_images(xval, yval,folder_name='Data_cross_7023/fold_' + str(index) + '/val/')
    index=index+1



names=sorted(os.listdir('Data_cross_7023/fold_1/train')) 
print(names)
BATCH_SIZE = 64



def create_model():
    inputs = keras.Input(shape=(224,224,1))
    L1 = layers.Conv2D(32, 3,strides=(2,2), padding="same")(inputs)
    L1 = layers.BatchNormalization()(L1)
    L1 = layers.Activation('relu')(L1)
    # --------------------------------------

    #-----------------------------------------
    L2 = layers.Conv2D(64, 3, padding="same")(L1)
    L2 = layers.BatchNormalization()(L2)
    L2 = layers.Activation('relu')(L2)
    #-----------------------------------------
    L3 = layers.MaxPooling2D(2)(L2)
    #---------------------------------------------
    L4 = layers.Conv2D(64, 3, padding="same")(L3)
    L4 = layers.BatchNormalization()(L4)
    L4 = layers.Activation('relu')(L4)

    L5 = layers.Conv2D(64, 5, padding="same")(L3)
    L5 = layers.BatchNormalization()(L5)
    L5 = layers.Activation('relu')(L5)

    L6 = layers.Conv2D(64, 7, padding="same")(L3)
    L6 = layers.BatchNormalization()(L6)
    L6 = layers.Activation('relu')(L6)

    L7 = layers.Concatenate()([L3,L4,L5,L6])
    L7 = layers.BatchNormalization()(L7)
    #---------------------------------------------------

    L8 = layers.Conv2D(128, 3,strides=(2,2), padding="same")(L7)
    L8 = layers.BatchNormalization()(L8)
    L8 = layers.Activation('relu')(L8)

    L9 = layers.Conv2D(128, 5,strides=(2,2), padding="same")(L7)
    L9 = layers.BatchNormalization()(L9)
    L9 = layers.Activation('relu')(L9)

    L10 = layers.Conv2D(128, 7, strides=(2,2), padding="same")(L7)
    L10 = layers.BatchNormalization()(L10)
    L10 = layers.Activation('relu')(L10)


    L3_1 = layers.Conv2D(128, 3,strides=(2,2), padding="same")(L3)
    L3_1 = layers.BatchNormalization()(L3_1)
    L3_1 = layers.Activation('relu')(L3_1)

    L7_1 = layers.MaxPooling2D(2)(L7)
 

    L11 = layers.Concatenate()([L3_1,L7_1,L8,L9,L10])
    L11 = layers.BatchNormalization()(L11)
    #-----------------------------------------
    L12 = layers.MaxPooling2D(2)(L11)
    #-------------------------------------------

    L13 = layers.Conv2D(256, 3,strides=(2,2), padding="same")(L12)
    L13 = layers.BatchNormalization()(L13)
    L13 = layers.Activation('relu')(L13)

    L14 = layers.Conv2D(256, 5,strides=(2,2), padding="same")(L12)
    L14 = layers.BatchNormalization()(L14)
    L14 = layers.Activation('relu')(L14)

    L15 = layers.Conv2D(256, 7, strides=(2,2), padding="same")(L12)
    L15 = layers.BatchNormalization()(L15)
    L15 = layers.Activation('relu')(L15)
    #===============================================
    L3_11 = layers.Conv2D(256, 3,strides=(2,2), padding="same")(L3)
    L3_11 = layers.BatchNormalization()(L3_11)
    L3_11 = layers.Activation('relu')(L3_11)

    L3_11 = layers.MaxPooling2D(2)(L3_11)

    L3_2 = layers.Conv2D(256, 3,strides=(2,2), padding="same")(L3_11)
    L3_2 = layers.BatchNormalization()(L3_2)
    L3_2 = layers.Activation('relu')(L3_2)



    L7_11 = layers.Conv2D(256, 3,strides=(2,2), padding="same")(L7_1)
    L7_11 = layers.BatchNormalization()(L7_11)
    L7_11 = layers.Activation('relu')(L7_11)

    L7_2 = layers.MaxPooling2D(2)(L7_11)

    L12_1 = layers.MaxPooling2D(2)(L12)
 

    L16 = layers.Concatenate()([L3_2,L7_2,L12_1,L13,L14,L15])
    L16 = layers.BatchNormalization()(L16)
 
    x = layers.GlobalAveragePooling2D()(L16)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
 

    model = keras.Model(inputs, outputs, name="Our_model")

 
    optim=keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
   

    return model



for num in range(1, 6):
 
    train_gen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255,brightness_range=[0.1,0.7],
                                                           rotation_range=20,
                                                           horizontal_flip=True,
                                                           vertical_flip=True,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2)
                                                       
                                                       
    valid_gen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_batches=train_gen.flow_from_directory(
        'Data_cross_7023/fold_' + str(num) + '/train',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        color_mode="grayscale",
        classes=names
        )
    val_batches=valid_gen.flow_from_directory(
        'Data_cross_7023/fold_' + str(num) + '/val',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False,
        color_mode="grayscale",
        classes=names
        )
     
    n_epochs = 250
    model= None
 
    model = create_model()

  
     

   
  
    plot_loss_1 = PlotLossesCallback()

# ModelCheckpoint callback - save best weights
    tl_checkpoint_1=None
    tl_checkpoint_1 = ModelCheckpoint(filepath='6_9_2022_Our_model_7023_StratifiedKFold_5_fold_' + str(num) + '.weights.best.hdf5',
                                      save_best_only=True,
                                      verbose=1)

# EarlyStopping - monitors the performance of the model and stopping the training process prevents overtraining

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=40,
                               restore_best_weights=True,
                               mode='min')



    history = model.fit(train_batches,
                         batch_size=BATCH_SIZE,
                         epochs=n_epochs,
                         validation_data=val_batches,
                         callbacks=[tl_checkpoint_1, early_stop],
                         verbose=2)
                         
                              
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1) 
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.grid()
    plt.legend(fontsize=15)

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='valid acc')
    plt.grid()
    plt.legend(fontsize=15)   
    
    
    print("----------Our Model K-fold : {}".format(num)) 
 
    preds_ft = model.predict(val_batches)
    pred_classes_ft = np.argmax(preds_ft, axis=1)
    true_classes = val_batches.classes
    acc_ft = accuracy_score(true_classes, pred_classes_ft)
   
    print("Our Model Accuracy for data : {:.2f}%".format(acc_ft * 100))
    
    print('Accuracy score is :', np.round(metrics.accuracy_score(true_classes, pred_classes_ft),4))
    print('Precision score is :', np.round(metrics.precision_score(true_classes, pred_classes_ft, average='weighted'),4))
    print('Recall score is :', np.round(metrics.recall_score(true_classes, pred_classes_ft, average='weighted'),4))
    print('F1 Score is :', np.round(metrics.f1_score(true_classes, pred_classes_ft, average='weighted'),4))
 
    print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(true_classes, pred_classes_ft),4))

    print('\t\tClassification Report:\n', metrics.classification_report(true_classes, pred_classes_ft))
    
    model.load_weights('6_9_2022_Our_model_7023_StratifiedKFold_5_fold_' + str(num) + '.weights.best.hdf5') 
        
    print("-------------Load best_ weights---------------")
 
    preds_ft = model.predict(val_batches)
    pred_classes_ft = np.argmax(preds_ft, axis=1)
    true_classes = val_batches.classes
    acc_ft = accuracy_score(true_classes, pred_classes_ft)
   
    print("Our Model Accuracy for data : {:.2f}%".format(acc_ft * 100))
     
    print('Accuracy score is :', np.round(metrics.accuracy_score(true_classes, pred_classes_ft),4))
    print('Precision score is :', np.round(metrics.precision_score(true_classes, pred_classes_ft, average='weighted'),4))
    print('Recall score is :', np.round(metrics.recall_score(true_classes, pred_classes_ft, average='weighted'),4))
    print('F1 Score is :', np.round(metrics.f1_score(true_classes, pred_classes_ft, average='weighted'),4))
 
    print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(true_classes, pred_classes_ft),4))

    print('\t\tClassification Report:\n', metrics.classification_report(true_classes, pred_classes_ft))
                            