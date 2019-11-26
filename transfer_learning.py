from __future__ import absolute_import, division, print_function
# nhét thư viện vào này
import os
# tensorflow phiên bản 1.10.0 nhé
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version is ", tf.__version__)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

# tải dataset vê
zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                                   fname="cats_and_dogs_filtered.zip", extract=True)
# get đường dẫn của file đã tải về
base_dir, _ = os.path.splitext(zip_file)
print(base_dir)
# cấu hình đường dẫn của thư mục training
train_dir = os.path.join(base_dir, 'train')
print(train_dir)
# cấu hình đường dẫn của thư mục validate
validation_dir = os.path.join(base_dir, 'validation')
print('day la duong dan: ')
print(validation_dir)
# Cấu hình đường dẫn của thư mục chưa ảnh mèo
train_cats_dir = os.path.join(train_dir, 'cats')
print('Total training cat images:', len(os.listdir(train_cats_dir)))

# Cấu hình đường dẫn của thư mục chưa ảnh chó
train_dogs_dir = os.path.join(train_dir, 'dogs')
print('Total training dog images:', len(os.listdir(train_dogs_dir)))

# Cấu hình đường dẫn của thư mục validate chứa ảnh mèo
validation_cats_dir = os.path.join(validation_dir, 'cats')
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))

# Cấu hình đường dẫn của thư mục dùng để validate chứa ảnh chó
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

image_size = 150  # tất cả ảnh được resize về 160x160
# số sample được đưa vào xử lý
batch_size = 32
# sinh thêm nhiều dạng dữ liệu nữa sử dụng ImageDataGenerator của tensorflow
# Rescale image
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Source directory for the training images
    target_size=(image_size, image_size),
    batch_size=batch_size,
    #  sử dụng binary_crossentropy loss, we need binary labels
    class_mode='binary')

classes = train_generator.class_indices
list_classes = list(classes.keys())
print(classes)
print(list_classes)
# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # Source directory for the validation images
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary')

IMG_SHAPE = (image_size, image_size, 3)
# load base model từ  MobileNet đã được huấn luyện trước rồi này
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                             include_top=False,
                                            weights='imagenet')
#Không train lại model có sẵn
base_model.trainable = False
# thêm các layer ở lớp cao nhất
# base_model.summary()
model = tf.keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation='sigmoid')
])
# sigmoid active function thường dùng cho binary classification

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.summary()
# Set callback để lưu model và tensorboard
filepath="best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only = False, save_best_only=True, mode='min')
logdir="mobilenet"
tfboard = TensorBoard(log_dir=logdir)

callbacks_list = [checkpoint, tfboard]
epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size
"""
# Cho phép train lại base model 
base_model.trainable = True

# Hiển thi số lớp trong base m
print("Number of layers in the base model: ", len(base_model.layers))

# Set điều kiện lấy layer  
fine_tune_at = 100

# Cho phóp train từ layer thứ 100 trở 
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
"""
history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              callbacks = callbacks_list,
                              )

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
