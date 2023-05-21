import Synthetic_MICCAI2020_dataset.model as smod
import cv2
import numpy as np

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (538, 701))
    return np.expand_dims(new_array, axis=0).shape

model = smod.new_model()

prediction = model.predict([prepare('Synthetic_MICCAI2020_dataset/others/Video_15/images/000.png'), None])
print(prediction)