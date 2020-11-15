# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

# %%
from exp3_data import datasets, nlbclass_datasets, viz, viz_class
from exp3_seg_models import unet1, unet2, unet3, unet3_enc_classifier

# %%
# with tf.device('CPU'):
model = unet3_enc_classifier()
model.summary()

# %%
import os
TRAIN_BATCH = 32
SHUFFLE_BUFFER = 200
base_dir = "./data/NLB"

# %%
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
    batch_size=TRAIN_BATCH
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
# train_ds = train_ds.batch(TRAIN_BATCH).apply(tf.data.experimental.ignore_errors())
# eval_ds = eval_ds.batch(1).apply(tf.data.experimental.ignore_errors())

# %%
lr = 1e-2
epochs=5
decay = lr/epochs
model_count = 1

optm = optimizers.Adam(
    learning_rate=lr,
    decay=decay,
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
