import model as md
import make_markup_images as mp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import time
import random
import numpy as np
from skimage.io import imshow
import os
from itertools import chain


image_folder = [f for f in os.listdir(md.FOLDER_PATH) if os.path.isdir(os.path.join(md.FOLDER_PATH, f))]
TRAIN_PATH =""
MASK_PATH  =""

X_train = np.zeros((1, md.IMG_HEIGHT, md.IMG_WIDTH, md.IMG_CHANEL), dtype=np.uint8) #feltölti 0-kal
Y_train = np.zeros((1, md.IMG_HEIGHT, md.IMG_WIDTH, 1), dtype = np.bool)

for all_f in image_folder: 
          
    TRAIN_PATH =   md.FOLDER_PATH +"/"+ all_f +"/images"      
    MASK_PATH  =   md.FOLDER_PATH +"/"+ all_f +"/ground_truth"
    if "video".lower() in all_f.lower():                    
        X_train_temp, Y_train_temp = md.load_traning_images(TRAIN_PATH, MASK_PATH, len(f for f in os.listdir(md.FOLDER_PATH) if os.path.isfile(os.path.join(md.FOLDER_PATH, f))))
        
    X_train = list(chain(X_train, X_train_temp))
    Y_train = list(chain(Y_train, Y_train_temp))
    
X_test_ = md.load_images(md.TEST_PATH, 300)


print('Done!')
  

start = time.time()

model = md.create_model()

if md.model_exists():
    print("Betölt")
    model.load_weights(md.WEIGHTS_PATH)
else:
    print("Build")

    #Build the model
    model.summary()
    #model.save('/model_for_hus.h5')
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_hus.h5', verbose = 1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir = '/')]
    results = model.fit(X_train, Y_train, validation_split=0.0001, batch_size=16, epochs=5, callbacks=callbacks)
    print('With params: {}'.format(results))

    md.save_model(model)
    end = time.time()
    tdiff = end - start
    print('Finnished building model in {}'.format(tdiff))
    
end = time.time()
tdiff = end - start
print('Finnished model in {}'.format(tdiff))
print('')


preds_train = model.predict(X_train)
preds_val   = model.predict(X_train)
preds_test  = model.predict(X_test )

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
ix = random.randint(0, len(preds_train) - 1)


#print('type: {} -> {}'.format(type(preds_train), type(preds_train[ix])))
img         = X_test[ix]
overlayable = np.squeeze(((preds_train[ix] > .5) * 255).astype(np.uint8))
plt.figure("in")
imshow(img)
imshow(mp.create_overlayed_image(img, overlayable))
plt.show()
a=3




