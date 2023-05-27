import model as md
import make_markup_images as mp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import time
import cv2
import random
from PIL import Image
import numpy as np
from skimage.io import imshow


X_train, Y_train = md.load_traning_images(md.TRAIN_PATH, md.MASK_PATH, 10)
X_test = md.load_images(md.TEST_PATH, 10)

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
    

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val   = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test  = model.predict(X_test, verbose=1)

 
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t   = (preds_val   > 0.5).astype(np.uint8)
#preds_test_t  = (preds_test  > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train) - 1)
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(preds_train[ix]))
plt.show()

print('type: {}'.format(type(preds_train[ix])))
imshow(mp.markup_from_image(Image.fromarray((preds_train[ix] * 255).astype(np.uint8))))
plt.show()


# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(preds_val[ix]))
plt.show()



