import os, sys, glob, pathlib
import tensorflow as tf
#from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import CategoricalAccuracy
import numpy as np
#import matplotlib.pyplot as plt

myepochs = 2000

# Ex: sbatch sbatch_VGG16_1a.sh 1aa1

try:
  myargs = sys.argv[1][2:]
except:
  print('No valid arguments presented')
  quit()

print(myargs)

modelnum = sys.argv[1]

if myargs[0] == 'a':
  print(0.25)
  mydropout = 0.25
elif myargs[0] == 'b':
  print(0.30)
  mydropout = 0.30
elif myargs[0] == 'c':
  print(0.35)
  mydropout = 0.35
elif myargs[0] == 'd':
  print(0.40)
  mydropout = 0.45
elif myargs[0] == 'e':
  print(0.45)
  mydropout = 0.45
else:
  print('choose proper dropout type: a, b, c, d, e')
  #quit()

if myargs[1] == '3':
  print(0.0006)
  mylr = 0.0006
elif myargs[1] == '2':
  print(0.0005)
  mylr = 0.0005
elif myargs[1] == '1':
  print(0.0004)
  mylr = 0.0004
else:
  print('choose proper learning rate type: 1, 2, 3')

#rootdir = '/ocean/projects/mcb180013p/shared/abalonevision/ImageData/'

mymonitor = 'val_categorical_accuracy'
workdir = './mbanerjee/'
rootdir = workdir + 'sex_determination/'
data_dir_train = pathlib.Path(rootdir + 'Training/')
#data_dir_validation = pathlib.Path(rootdir + 'large/Validation/')
data_dir_test = pathlib.Path(rootdir + 'Test/')
checkpoint_filepath = workdir + 'tfmodelcheckpoints/mbmodel'+ modelnum + '_weights.{epoch:02d}-vca-{val_categorical_accuracy:.4f}-ca-{categorical_accuracy:.4f}.hdf5'
#checkpoint_filepath = workdir + 'gordon/tfmodelcheckpoints/'+ modelnum + '_weights.{epoch:02d}-vba-{val_binary_accuracy:.4f}-ba-{binary_accuracy:.4f}.hdf5'

# Predefined image dimensions
batch_size = 32
img_height = 377
img_width = 321

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  label_mode='categorical',
  subset='both',
  seed=1320,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  label_mode='categorical',
  #seed=1320,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#print(val_ds)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# VGG16 model
num_classes = 2
model = tf.keras.Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))

model.add(tf.keras.layers.Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.Dropout(mydropout))
model.add(tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.Dropout(mydropout))
#model.add(tf.keras.layers.Dense(1000, activation=tf.keras.layers.Softmax()))

#model.add(tf.keras.layers.Dense(4096*2, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
#model.add(tf.keras.layers.Dense(4096*2*2, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
#model.add(tf.keras.layers.Dense(4096*2*2*2, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate=mylr),
    loss=tf.keras.losses.CategoricalCrossentropy(), #from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.FalseNegatives()])
    #loss=tf.keras.losses.CategoricalCrossCrossentropy(),
    #metrics=['accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=1,
    monitor=mymonitor,
    mode='max',
    save_best_only=False,
    initial_value_threshold=0.70)

model_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor=mymonitor,
    min_delta=0,
    patience=100,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=200
)

h = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=myepochs,
  callbacks=[model_checkpoint_callback, model_earlystop]
)

