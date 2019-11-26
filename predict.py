
import cv2
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
img_width, img_height = 160, 160
inf_model = tf.keras.models.load_model('best.hdf5')
# def preprocess_image(img):
#     if (img.shape[0] != 150 or img.shape[1] != 150):
#         img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
#     img = (img / 127.5)
#     img = img - 1
#     img = np.expand_dims(img, axis=0)
#     return img


mypath = "predict/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)
# predicting images
dog_counter = 0
cat_counter = 0

for file in onlyfiles:
    img = image.load_img(mypath + file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = inf_model.predict_classes(images, batch_size=10)
    classes = classes[0][0]
    if classes == 1:
        print(file + ": " + 'dog')
        dog_counter += 1
    elif classes == 0:
        print(file + ": " + 'cat')
        cat_counter += 1
print("Total Dogs :", dog_counter)
print("Total Cats :", cat_counter)