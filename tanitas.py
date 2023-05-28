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

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir = '/')]

if md.model_exists():
    print("Betölt")
    model.load_weights(md.WEIGHTS_PATH)
else:
    print("Build")

    #Build the model
    model.summary()
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



# ecaluation
md.evaluate_model(model=model, batch_size=16, callbacks=callbacks)


X_test = md.load_images(md.TEST_PATH, 300)
preds_test  = model.predict(X_test)

 

ix = random.randint(0, len(preds_test) - 1)

for i in range(0, len(preds_test)-1):
    img         = X_test[i]
    overlayable = np.squeeze(((preds_test[i] > .5) * 255).astype(np.uint8))
    imshow(mp.create_overlayed_image(img, overlayable))
    plt.show()

