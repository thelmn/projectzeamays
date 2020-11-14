"""
Convert a folder of images into a tfrecord file
"""
#%%
import os
import tensorflow as tf
import numpy as np
import json 

#%%

def generate_tfrecord(base_folder, json_ds_name, class_set, out_file_name):
    out_file = os.path.join(base_folder, out_file_name)
    json_file = os.path.join(base_folder, json_ds_name)
    class_map = dict(zip(class_set, range(len(class_set))))

    with open(json_file) as f:
        with tf.io.TFRecordWriter(out_file) as writer:
            obj = json.load(f)
            for v in obj:
                fname = v["file_name"]
                fname = os.path.join(base_folder, fname)
                image_data = tf.io.read_file(fname)

                boxes = np.asarray(v["boxes"], dtype=np.float32)
                classes = np.array([class_map[cl] for cl in v["class"]], dtype=np.int32)

                N = boxes.shape[0]
                is_crowd = np.zeros((N, ), dtype=np.int8)

                print(type(tf.io.serialize_tensor(boxes)))
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'file': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.convert_to_tensor(v["file_name"]).numpy()])),
                    'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data.numpy()])),
                    'boxes': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(boxes).numpy()])),
                    'class': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(classes).numpy()])),
                    'is_crowd': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(is_crowd).numpy()])),
                }))
                writer.write(example.SerializeToString())
                
#%%
DATADIR = "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/final"
CLASSES = ['healthy', 'faw', 'zinc_def']
generate_tfrecord(DATADIR, 'rand_rois.json', CLASSES, 'rand_rois.tfrecord')

# %%
tfrecs = [
    "/media/lmn/41BA76B7045343B9/dev/school/projectzeamays/data/final/rand_rois.tfrecord"
]
tf_feature_desc = {
    'file': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'boxes': tf.io.FixedLenFeature([], tf.string),
    'class': tf.io.FixedLenFeature([], tf.string),
    'is_crowd': tf.io.FixedLenFeature([], tf.string),
}
ds = tf.data.TFRecordDataset(tfrecs)

for tfexample in ds.take(10):
    example = tf.io.parse_single_example(tfexample, features=tf_feature_desc)
    fname = example['file'].numpy()
    im = tf.io.decode_image(example['image'].numpy()).numpy()
    boxes = tf.io.parse_tensor(example['boxes'], tf.float32).numpy()
    klass = tf.io.parse_tensor(example['class'], tf.int32).numpy()
    is_crowd = tf.io.parse_tensor(example['is_crowd'], tf.int8).numpy()
    im = im.astype("float32")
    print(
        'example: ',
        fname,
        im.shape, im.dtype, 
        boxes.shape, boxes.dtype, 
        klass.shape, klass.dtype, 
        is_crowd.shape, is_crowd.dtype,
        )

# %%
