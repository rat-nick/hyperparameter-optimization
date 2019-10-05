import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
import distutils
if distutils.version.LooseVersion(tf.__version__) < '1.14':
    raise Exception('This notebook is compatible with TensorFlow 1.14 or higher, for TensorFlow 1.13 or lower please use the previous version at https://github.com/tensorflow/tpu/blob/r1.13/tools/colab/fashion_mnist.ipynb')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# add empty color dimension
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)

from google.colab import drive
drive.mount('/content/drive')


def create_model(front_end_layers, init_filters, kernel_size, pooling_size, dense_layers, dense_units):
  model = tf.keras.models.Sequential()
  
  #front-end layers (conv to pool pairs)
  for i in range(front_end_layers):
    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(init_filters * pow(2, i), (kernel_size, kernel_size), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pooling_size, pooling_size), strides=(pooling_size, pooling_size)))
    model.add(tf.keras.layers.Dropout(0.25))    

  #back-end dense layers
  for i in range(dense_layers):
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.5))
  
  #prediction layer
  model.add(tf.keras.layers.Dense(10))
  model.add(tf.keras.layers.Activation('softmax'))
  return model
  
  # import os

# resolver = tf.contrib.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.contrib.distribute.initialize_tpu_system(resolver)
# strategy = tf.contrib.distribute.TPUStrategy(resolver)

#list of hyperparameter values
front_end_layers = [1,2]
init_filters = [32, 64]
kernel_size = [3, 5]
pool_size = [2, 3]
dense_layers = [1, 2]
dense_units = [256, 512]

for fel in front_end_layers:
  for dl in dense_layers:
    for du in dense_units:
      for inf in init_filters:
        for ks in kernel_size:
          for ps in pool_size:
            # with strategy.scope():
            logdir = str(fel) + "x" + str(ks) + "init" + str(inf) + "x" + str(ps) + "_" + str(dl) + "x" + str(du)  
              
            mc=ModelCheckpoint(filepath="/content/drive/My Drive/diplomski/" + logdir + "/best_model.h5", monitor="val_loss", verbose=1, mode="min", save_best_only=True)
            es=EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/content/drive/My Drive/diplomski/"+logdir)          
            model = create_model(fel, inf, ks, ps, dl, du)
            model.compile(
              optimizer=tf.keras.optimizers.Adam(),
              #optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

            model.fit(
              x_train.astype(np.float32), y_train.astype(np.float32),
              epochs=1000,
              # steps_per_epoch=46,
              # batch_size=128 * 8,
              # validation_freq=10,
              validation_data=(x_val.astype(np.float32), y_val.astype(np.float32)),
              callbacks=[tensorboard_callback,es, mc]
            )
            #K.clear_session()

prefix = "/content/drive/My Drive/diplomski/"
results = {}
for fel in front_end_layers:
  for dl in dense_layers:
    for du in dense_units:
      for inf in init_filters:
        for ks in kernel_size:
          for ps in pool_size:
            saved_model = create_model(fel, inf, ks, ps, dl, du)
            logdir = str(fel) + "x" + str(ks) + "init" + str(inf) + "x" + str(ps) + "_" + str(dl) + "x" + str(du)
            saved_model.load_weights(prefix+logdir+"/best_model.h5")
            saved_model.compile(
              optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy']
            )
            score = saved_model.evaluate(x_test.astype(np.float32), y_test.astype(np.float32), verbose=0)
            print("%s: %.2f%%" % (logdir, score[1]*100))
            results[logdir]=score[1]*100			