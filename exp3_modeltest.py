#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

# %%
from exp3_data import datasets, viz, viz_nmasks, smooth_mask, contour_mask
from exp3_seg_models import unet1, unet2, unet3

# %%
model_count = 1
batch_size = 10

base_dir = './data/final'
annot_file = './data/final/rois_final.json'
# annot_file = './data/final/rois_whealthy.json'

# %%
saved = None
with tf.device('CPU'):
    saved = tf.keras.models.load_model(f"./models/unet_seg/{model_count}")

# %%
train_ds, eval_ds = datasets(base_dir, annot_file)

# %%
SHUFFLE_BUFFER = 300
train_ds = train_ds.shuffle(SHUFFLE_BUFFER).batch(batch_size).apply(tf.data.experimental.ignore_errors())

# %%
sample = train_ds.take(1)
# %%
images, masks, pred_masks = None, None, None
for batch in sample:
    print(len(batch))
    images, masks = batch
    pred_masks = saved.predict(images)
print(f"predicted {len(pred_masks)} images")
# %%
pred_masks = (np.argmax(pred_masks, axis=-1)).astype(np.int8)

# %%
import cv2
import numpy as np
for image, mask, pred_mask in zip(images, masks, pred_masks):
    pred_mask = np.repeat(pred_mask, 4, axis=0)
    pred_mask = np.repeat(pred_mask, 4, axis=1)
    pred_mask_1 = pred_mask == 1
    pred_mask_2 = pred_mask == 2

    pred_mask = np.stack((pred_mask_1, pred_mask_2), axis=-1)

    mask = mask.numpy()
    mask = np.repeat(mask, 4, axis=0)
    mask = np.repeat(mask, 4, axis=1)
    mask = np.stack((mask == 1, mask == 2), axis=-1)

    smooth_1 = contour_mask(np.float32(pred_mask_1))
    smooth_2 = contour_mask(np.float32(pred_mask_2))
    print(smooth_1.shape, smooth_2.shape)
    smooth = np.stack((smooth_1, smooth_2), axis=-1)
    # print(smooth.shape)
    viz_nmasks(image.numpy(), [mask, pred_mask, smooth])

# %%
