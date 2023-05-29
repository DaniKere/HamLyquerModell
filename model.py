import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.io import imshow

# Image pathsSynthetic_MICCAI2020_dataset\others\Video_17\images
TESTS_PATH =         'Synthetic_MICCAI2020_dataset/others'
#TRAIN_PATH =          #'Synthetic_MICCAI2020_dataset/Video_01/green_screen'
#MASK_PATH  =          #'Synthetic_MICCAI2020_dataset/Video_01/ground_truth'
FOLDER_PATH = 'Synthetic_MICCAI2020_dataset'
IMG_WIDTH =  512 #701
IMG_HEIGHT = 512 #538
IMG_CHANEL = 3

TRAIN_SEP = 100
physical_devices = tf.config.list_physical_devices('GPU')
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
except:
    print('Invalid device or cannot modify virtual devices once initialized')

# Not used
#MODEL_PATH   = 'models/model_for_hus.h5'
WEIGHTS_PATH = 'models/model_for_hus_weights.h5'

numeric = 0
def debug_print():
    global numeric
    #print(numeric)
    numeric += 1

def model_exists():
    return os.path.exists(WEIGHTS_PATH)

def create_model():
    debug_print()
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANEL))
    s= tf.keras.layers.Lambda(lambda x: x/255)(inputs)

    debug_print()
    #Contraction path
    c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3,activation='relu', kernel_initializer='he_normal', padding='same')(s)
    debug_print()
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    debug_print()
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    debug_print()
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    debug_print()
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    debug_print()
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    debug_print()
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    debug_print()
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    debug_print()
    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    debug_print()
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    debug_print()
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    debug_print()
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    debug_print()
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    debug_print()
    return model

def train_model(model: tf.keras.Model, validation_split, batch_size, epochs, callbacks):
    image_folder = [f for f in os.listdir(FOLDER_PATH) if os.path.isdir(os.path.join(FOLDER_PATH, f))]
    results = []
    for all_f in image_folder:
        if "video" in all_f.lower():     
            train_path =   FOLDER_PATH +"/"+ all_f +"/images"      
            mask_path  =   FOLDER_PATH +"/"+ all_f +"/ground_truth"               
            X_train, Y_train = load_traning_images(train_path, mask_path)
            for x in range(0, len(X_train), TRAIN_SEP):
                x_tmp = X_train[x:x+TRAIN_SEP]
                y_tmp = Y_train[x:x+TRAIN_SEP]
                results.append(model.fit(x_tmp, y_tmp, validation_split=validation_split, batch_size=batch_size, epochs=epochs, callbacks=callbacks))
    return results

def evaluate_model(model: tf.keras.Model, batch_size, callbacks):
    image_folder = [f for f in os.listdir(TESTS_PATH) if os.path.isdir(os.path.join(TESTS_PATH, f))]
    results = []
    for all_f in image_folder:
        if "video" in all_f.lower():     
            train_path =   TESTS_PATH +"/"+ all_f +"/images"      
            mask_path  =   TESTS_PATH +"/"+ all_f +"/ground_truth"               
            X_eval, Y_eval = load_traning_images(train_path, mask_path)
            results.append(model.evaluate(X_eval, Y_eval, batch_size=batch_size, callbacks=callbacks))
    return results

def save_model(model: tf.keras.models.Model):
    model.save_weights(WEIGHTS_PATH)

def load_traning_images(train_path, mask_path, maxnum = -1):
    train_ids = os.listdir(mask_path)
    if maxnum >= 0 and len(train_ids) > maxnum:
        load_len = maxnum
    else:
        load_len = len(train_ids)

    X_train = np.zeros((load_len, IMG_HEIGHT, IMG_WIDTH, IMG_CHANEL), dtype=np.uint8) #feltölti 0-kal
    Y_train = np.zeros((load_len, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

    for n, id_ in tqdm(enumerate(train_ids), total=load_len):
        if n >= load_len:
            break
        path = train_path + '/' + id_
        img = imread(path)[:,:,:IMG_CHANEL]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img  #Fill empty X_train with values from img
        
        mask   = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        t_path = mask_path + '/' + id_   
        bi     = imread(t_path)[:,:,:1]
        bi     = resize(bi, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask   = np.reshape(bi, (IMG_HEIGHT, IMG_WIDTH, 1))
        
        Y_train[n] = mask

    return X_train, Y_train

def load_images(img_path, maxnum = -1):
    test_ids = os.listdir(img_path)
    if maxnum >= 0 and len(test_ids) > maxnum:
        load_len = maxnum
    else:
        load_len = len(test_ids)

    X_test = np.zeros((load_len, IMG_HEIGHT, IMG_WIDTH, IMG_CHANEL), dtype=np.uint8)

    print('load test images')
    for n, id_ in tqdm(enumerate(test_ids), total=load_len):
        if n >= load_len:
            break

        path_husos = img_path + '/' + id_
        img = imread(path_husos)
        img =resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img

    return X_test

def load_images_camera(frame):
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANEL), dtype=np.uint8)    
    frame = frame[:,:,:IMG_CHANEL]
 #   imshow(frame)
    img =resize(frame, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[0] = img
    return X_test