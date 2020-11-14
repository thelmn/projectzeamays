#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

# %%
from exp3_data import datasets, viz
from exp3_seg_models import unet1, unet2, unet3

# %%
# with tf.device('CPU'):
model = unet3()
model.summary()

# %%
TRAIN_BATCH = 10
SHUFFLE_BUFFER = 200
base_dir = "./data/final"
annotations_file = "./data/final/rois.json"
train_ds, eval_ds = datasets(base_dir, annotations_file)

# %%
# i = 0
# for image, mask in train_ds:
#     viz( (image.numpy(), mask.numpy()) )
#     i += 1
#     if i == 1:
#         break
# %%
train_ds = train_ds.shuffle(SHUFFLE_BUFFER).batch(TRAIN_BATCH).apply(tf.data.experimental.ignore_errors())
eval_ds = eval_ds.batch(1).apply(tf.data.experimental.ignore_errors())

# %%
lr = 1e-2
epochs=10
decay = lr/epochs
model_count = 2

optm = optimizers.Adam(
    learning_rate=lr,
    decay=decay,
)
model.compile(
    optimizer=optm,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=['accuracy'],
)

# %%
log_dir = f"./logs/lmn_detection/{model_count}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback]
)

# %%
model_path = f"./models/unet_seg/{model_count}"
model.save(model_path)

# %%
# i = 0
# for image, mask in train_ds:
#     viz( (image.numpy().astype(np.int), mask.numpy()) )
#     y = model.predict(np.expand_dims(image.numpy(), axis=0))
#     mask = np.argmax(y, axis=-1).squeeze()
#     viz( (image.numpy().astype(np.int), mask) )
#     i += 1
#     if i == 1:
#         break

# %%
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# %%
# tf.config.list_physical_devices('GPU')
# %%
