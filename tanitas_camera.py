import model as md
import make_markup_images as mp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import time
import random
import numpy as np
from skimage.io import imshow
import cv2
#X_train = np.zeros((0, md.IMG_HEIGHT, md.IMG_WIDTH, md.IMG_CHANEL), dtype=np.uint8) #feltölti 0-kal
#Y_train = np.zeros((0, md.IMG_HEIGHT, md.IMG_WIDTH, 1), dtype=bool)


model = md.create_model()

if md.model_exists():
    print("Load Model Weights")
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
    print('Finished model')


a="c"

if a=="c":

    video_=cv2.VideoCapture(0)
    
    if not video_.isOpened():
        print('Faild to open the camera')
    else:
        
        while True:
            ret, frame = video_.read()
        
            org_=md.load_images_camera(frame)
            cv2.imshow("db",org_[0])
            preds_test  = model.predict(org_)
            
            cv2.imshow("in",frame)
            overlayable = np.squeeze(((preds_test[0] > .5) * 255).astype(np.uint8))    
            cv2.imshow("out",mp.create_overlayed_image(org_[0], overlayable))    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
else:
    X_test = md.load_images(md.TEST_PATH, ind=md.INDIVIDUAL_IMG_PATH)
    preds_test  = model.predict(X_test)    

    #ix = random.randint(0, len(preds_test) - 1)
    if len(X_test) == 1:
        img  = X_test[0]
        overlayable = np.squeeze(((preds_test[0] > .5) * 255).astype(np.uint8))
        imshow(mp.create_overlayed_image(img, overlayable))
        plt.show()
#else:   
#        for i in range(0, len(preds_test)-1):
#            imgs = X_test[i]
#            overlayable = np.squeeze(((preds_test[i] > .5) * 255).astype(np.uint8))
#            imshow(mp.create_overlayed_image(imgs, overlayable))
#            plt.show()
