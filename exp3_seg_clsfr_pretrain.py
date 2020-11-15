# %%
import argparse

# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

# %%
from exp3_data import datasets, nlbclass_datasets, viz, viz_class
from exp3_seg_models import unet1, unet2, unet3, unet3_enc_classifier

# %%
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='decay rate for exponential lr decay')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to trains')
parser.add_argument('--model_i', type=int, default=1, help='model count, for saving and logs')
parser.add_argument('--n_batch', type=int, default=32, help='batch size')
parser.add_argument('--n_train', type=int, default=None, help='Number of batches to take for training')
parser.add_argument('--base_dir', type=str, default='./data/NLB', help='data folder')

args = parser.parse_args()

lr = args.lr
decay = args.lr_decay
epochs = args.epochs
model_count = args.model_i
batch_size = args.n_batch
n_train = args.n_train
base_dir = args.base_dir

# %%
# with tf.device('CPU'):
model = unet3_enc_classifier()
model.summary()

# %%
SHUFFLE_BUFFER = 200

# %%
import os
classes = []
for subdir in sorted(os.listdir(base_dir)):
    if os.path.isdir(os.path.join(base_dir, subdir)):
        classes.append(subdir)
num_class = len(classes)
class_indices = dict(zip(classes, range(len(classes))))

# %%
train_ds, eval_ds = nlbclass_datasets(
    base_dir, 
    n_classes=num_class, 
    batch_size=batch_size,
    n_train=n_train if n_train else 2000
)

# %%
# i = 0
# for images, labels in train_ds:
#     image = images[0]
#     label = labels[0]
#     viz_class( (image.numpy(), label.numpy()), classes )
#     i += 1
#     if i == 1:
#         break
# %%
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
eval_ds = eval_ds.apply(tf.data.experimental.ignore_errors())

# %%
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=epochs,
    decay_rate=decay
)

optm = optimizers.Adam(
    learning_rate=lr_schedule
)

model.compile(
    optimizer=optm,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # metrics=['accuracy'],
)

# %%
log_dir = f"./logs/lmn_nlb_pretrain/{model_count}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback]
)

# %%
model_path = f"./models/unet_seg_nlb_pretrain/{model_count}"
model_weights_path = f"./models/unet_seg_nlb_pretrain/{model_count}_weights"
model.save(model_path)
model.save_weights(model_weights_path)

# %%
