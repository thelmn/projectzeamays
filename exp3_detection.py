# %%
import argparse

#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

# %%
from exp3_data import datasets, viz
from exp3_seg_models import unet1, unet2, unet3

# %%
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for exponential lr decay')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to trains')
parser.add_argument('--model_i', type=int, default=1, help='model count, for saving and logs')
parser.add_argument('--n_batch', type=int, default=32, help='batch size')
parser.add_argument('--model', type=str, default='unet3', help='model to train: unet1, unet2, unet3')
parser.add_argument('--base_dir', type=str, default='./data/final', help='data folder')
parser.add_argument('--annot_file', type=str, default='./data/final/rois_final.json', help='file with box annotations')

args = parser.parse_args()

print("config:", args)

lr = args.lr
epochs = args.epochs
decay = lr/epochs
model_count = args.model_i
batch_size = args.n_batch
base_dir = args.base_dir
annot_file = args.annot_file
model_variant = args.model


# %%
# with tf.device('CPU'):
if model_variant == 'unet1':
    model = unet1()
elif model_variant == 'unet2':
    model = unet2()
else:
    model = unet3()

model.summary()

# %%

train_ds, eval_ds = datasets(base_dir, annot_file)

# %%
# i = 0
# for image, mask in train_ds:
#     viz( (image.numpy(), mask.numpy()) )
#     i += 1
#     if i == 1:
#         break
# %%
SHUFFLE_BUFFER = 300
train_ds = train_ds.shuffle(SHUFFLE_BUFFER).batch(batch_size).apply(tf.data.experimental.ignore_errors())
eval_ds = eval_ds.batch(1).apply(tf.data.experimental.ignore_errors())

# %%
# lr_schedule = optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=lr,
#     decay_steps=epochs,
#     decay_rate=decay
# )
# %%
optm = optimizers.Adam(
    learning_rate=lr,
    decay=decay
    )

model.compile(
    optimizer=optm,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=['accuracy'],
)

# %%


log_dir = f"./logs/lmn_detection/{model_count}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
best_model_path = f"./models/unet_seg/{model_count}_best"
mcp_save = tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4, mode='min')

model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback, earlyStopping, mcp_save, reduce_lr_loss]
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
