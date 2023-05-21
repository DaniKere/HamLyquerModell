import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import numpy as np
import os
from tqdm import tqdm
import cv2
import random
TRAIN_PATH ='Synthetic_MICCAI2020_dataset\Video_01'
TEST_PATH  ='Synthetic_MICCAI2020_dataset\Video_01\test'

IMG_WIDTH = 701
IMG_HEIGHT = 538
IMG_CHANEL = 3
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
X_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,IMG_CHANEL), dtype=np.uint8)
Y_train = np.zeros((len(train_ids),IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = cv2.imread(path + '/images/' + id_ + '.png')
    #img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img  
                                    
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = cv2.imread(path + '/masks/' + id_)
    #mask_ = np.expand_dims(cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    mask = np.maximum(mask, mask_)            
    Y_train[n] = mask 
    
# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANEL), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = cv2.imread(path + '/images/' + id_ + '.png')
    sizes_test.append([img.shape[0], img.shape[1]])
    #img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')
    
image_x = random.randint(0, len(train_ids))
cv2.imshow(X_train[image_x])
plt.show()
cv2.imshow(np.squeeze(Y_train[image_x]))
plt.show()

#
##Build the model
#inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANEL))
#s= tf.keras.layers.Lambda(lambda x: x/255)(inputs)
#
##Contraction path
#c1 =  tf.keras.layers.Conv2D(16, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(s)
#c1 = tf.keras.layers.Dropout(0.1)(c1)
#c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
#
#c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#c2 = tf.keras.layers.Dropout(0.1)(c2)
#c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
# 
#c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#c3 = tf.keras.layers.Dropout(0.2)(c3)
#c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
# 
#c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#c4 = tf.keras.layers.Dropout(0.2)(c4)
#c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
# 
#c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#c5 = tf.keras.layers.Dropout(0.3)(c5)
#c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
#
##Expansive path 
#u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#u6 = tf.keras.layers.concatenate([u6, c4])
#c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#c6 = tf.keras.layers.Dropout(0.2)(c6)
#c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
# 
#u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#u7 = tf.keras.layers.concatenate([u7, c3])
#c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#c7 = tf.keras.layers.Dropout(0.2)(c7)
#c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
# 
#u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#u8 = tf.keras.layers.concatenate([u8, c2])
#c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#c8 = tf.keras.layers.Dropout(0.1)(c8)
#c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
# 
#u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
#c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#c9 = tf.keras.layers.Dropout(0.1)(c9)
#c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
# 
#outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
#model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()
#
#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose = 1, save_best_only=True)
#callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#             tf.keras.callbacks.TensorBoard(log_dir='logs')]
#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
#
#
##
### Load and preprocess the image data
##train_dataset = keras.preprocessing.image_dataset_from_directory(
##    "Synthetic_MICCAI2020_dataset\Video_01\green_screen",
##    batch_size=32,
##    image_size=(701, 538),
##    shuffle=True,
##)
##validation_dataset = keras.preprocessing.image_dataset_from_directory(
##    "Synthetic_MICCAI2020_dataset\Video_01\images",
##    batch_size=32,
##    image_size=(701, 538),
##    shuffle=True,
##)
##
###UNET
### Define the neural network architecture
##model = keras.Sequential([
##    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),
##    keras.layers.MaxPooling2D(),
#    keras.layers.Flatten(),
#    keras.layers.Dense(10, activation='softmax')
#])
#
#
#
#
#
## Train the neural network
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
#
## Model save
#model.save('model.h5')
#
#