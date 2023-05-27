import model as md
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

train_ids = os.listdir(md.MASK_PATH)
test_ids = os.listdir(md.MASK_PATH)
X_train = np.zeros((len(train_ids), md.IMG_HEIGHT, md.IMG_WIDTH, md.IMG_CHANEL), dtype=np.uint8) #feltölti 0-kal
Y_train = np.zeros((len(train_ids), md.IMG_HEIGHT, md.IMG_WIDTH, 1), dtype = np.bool)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = md.TRAIN_PATH +'/' + id_
    img = imread(path)[:,:,:md.IMG_CHANEL]  
    img = resize(img, (md.IMG_HEIGHT, md.IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    
    mask= np.zeros((md.IMG_HEIGHT, md.IMG_WIDTH, 1), dtype=np.bool)    
    t_path=md.MASK_PATH +'/' + id_    
    # Replace with the desired threshold value
    #binary_image = cv2.cvtColor(imread(t_path), cv2.COLOR_RGB2GRAY)    
    bi = imread(t_path)[:,:,:1]
    bi = resize(bi,(md.IMG_HEIGHT, md.IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.reshape(bi,(md.IMG_HEIGHT, md.IMG_WIDTH, 1))
    
    Y_train[n] = mask

X_test = np.zeros((len(test_ids), md.IMG_HEIGHT, md.IMG_WIDTH, md.IMG_CHANEL), dtype=np.uint8)
sizes_test = []
print('load test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path_husos = md.TEST_PATH +'/' + id_
    img = imread(path_husos)
    img =resize(img, (md.IMG_HEIGHT, md.IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
print('Done!')
    
image_x = random.randint(0, len(train_ids))
#imshow(X_train[image_x])

#plt.imshow(np.squeeze(Y_train[image_x]))
#plt.show()
if os.path.exists("model_for_hus.h5"):
    print("Betölt")
else:
    #Build the model
    start = time.time()
    
    model = md.create_model()

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

