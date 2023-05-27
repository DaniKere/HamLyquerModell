import model as md
import make_markup_images as mp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import time
import random
import numpy as np
from skimage.io import imshow


#X_train = np.zeros((0, md.IMG_HEIGHT, md.IMG_WIDTH, md.IMG_CHANEL), dtype=np.uint8) #feltölti 0-kal
#Y_train = np.zeros((0, md.IMG_HEIGHT, md.IMG_WIDTH, 1), dtype=bool)

start = time.time()
model = md.create_model()

if md.model_exists():
    print("Betölt")
    model.load_weights(md.WEIGHTS_PATH)
else:
    print("Build")

    #Build the model
    model.summary()
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_hus.h5', verbose = 1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir = '/')]
    results = md.train_model(model=model, validation_split=0.0001, batch_size=16, epochs=5, callbacks=callbacks)
    print('With params: {}'.format(results))

    md.save_model(model)
    end = time.time()
    tdiff = end - start
    print('Finnished building model in {}'.format(tdiff))
    
end = time.time()
tdiff = end - start
print('Finnished model in {}'.format(tdiff))
print('')


X_test = md.load_images(md.TEST_PATH, 300)

#preds_train = model.predict(X_train[0])
#preds_val   = model.predict(X_train[0])
preds_test  = model.predict(X_test [0])

 
preds_train_t = (preds_test > 0.5).astype(np.uint8)
ix = random.randint(0, len(preds_train_t) - 1)


#print('type: {} -> {}'.format(type(preds_train), type(preds_train[ix])))
img         = X_test[ix]
overlayable = np.squeeze(((preds_train_t[ix] > .5) * 255).astype(np.uint8))
plt.figure("in")
imshow(img)
imshow(mp.create_overlayed_image(img, overlayable))
plt.show()
a=3




