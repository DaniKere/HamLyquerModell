import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import numpy as np
import os
import time
from tqdm import tqdm
import cv2
import random
from PIL import Image
from skimage.io import imread, imshow
from skimage.transform import resize
TEST_PATH = 'Synthetic_MICCAI2020_dataset/Video_01/images'
TRAIN_PATH ='Synthetic_MICCAI2020_dataset/Video_01/green_screen'
MASK_PATH  ='Synthetic_MICCAI2020_dataset/Video_01/ground_truth'
IMG_WIDTH =  512 #701
IMG_HEIGHT = 512 #538
IMG_CHANEL = 3
train_ids = os.listdir(MASK_PATH)
test_ids = os.listdir(MASK_PATH)
X_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,IMG_CHANEL), dtype=np.uint8) #feltölti 0-kal
Y_train = np.zeros((len(train_ids),IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH +'/' + id_
    img = imread(path)[:,:,:IMG_CHANEL]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    
    mask= np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)    
    t_path=MASK_PATH +'/' + id_    
    # Replace with the desired threshold value
    #binary_image = cv2.cvtColor(imread(t_path), cv2.COLOR_RGB2GRAY)    
    bi = imread(t_path)[:,:,:1]
    bi = resize(bi,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.reshape(bi,(IMG_HEIGHT, IMG_WIDTH, 1))
    
    Y_train[n] = mask  

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANEL), dtype=np.uint8)
sizes_test = []
print('load test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path_husos = TEST_PATH +'/' + id_
    img = imread(path_husos)
    img =resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
print('Done!')
    
image_x = random.randint(0, len(train_ids))
#imshow(X_train[image_x])

#plt.imshow(np.squeeze(Y_train[image_x]))
#plt.show()
if os.path.exists("kamu_model_for_hus.h5"):
    print("Betölt")
else:
    #Build the model
    start = time.time()
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANEL))
    s= tf.keras.layers.Lambda(lambda x: x/255)(inputs)

    #Contraction path
    c1 =  tf.keras.layers.Conv2D(filters=16, kernel_size=3,activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    #model.save('/model_for_hus.h5')
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_hus.h5', verbose = 1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir = '/')]
    results = model.fit(X_train, Y_train, validation_split=0.0001, batch_size=16, epochs=5, callbacks=callbacks)

    try:
        model.save('model_for_hus.h5')
    except Exception as err:
        print(err)

    try:
        model.save_weights('model_for_hus_weights.h5')
    except Exception as err:
        print(err)
    
    end = time.time()
    tdiff = end - start
    print('Finnished building model in {tdiff}')
    print('With params: {results}')
    print('')
    

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val   = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test  = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t   = (preds_val   > 0.5).astype(np.uint8)
preds_test_t  = (preds_test  > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

