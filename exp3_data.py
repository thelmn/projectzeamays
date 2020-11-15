# %%
import os
import json
import random
import numpy as np
import tensorflow as tf
from utils import np_box_ops as boxops
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#%%
class RandRoi():
    def __init__(self, base_dir, classes, annotations_file, train_split=0.9):
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = base_dir
        self.classes = classes
        self.annotations_file = annotations_file
        assert os.path.isdir(self.imgdir), self.imgdir

        self.rois, self.eval_rois = self.load(train_split)

    def load(self, train_split=0.9):
        json_file = self.annotations_file
        with open(json_file) as f:
            obj = json.load(f)

        ret_train = []
        ret_eval = []

        for v in obj:
            # make this a training point with prob train_split
            if random.uniform(0,1) < train_split:
                split = 'train'
            else: 
                split = 'eval'

            _fname = v["file_name"]
            fname = os.path.join(self.imgdir, _fname)
            assert os.path.exists(fname), fname

            roidb = {"file_name": fname, "image_id": _fname}
            roidb["boxes"] = np.asarray(v["boxes"], dtype=np.float32)
            N = roidb["boxes"].shape[0]
            if N == 0:
                continue
            # roidb["segmentation"] = segs
            roidb["class"] = np.array([self.classes[cl] for cl in v["class"]], dtype=np.int64)
            roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
            if split == 'eval':
                ret_eval.append(roidb)
            
            # attempt to balance the zinc class
            if split == 'train' and len(roidb['class']) > 0 and roidb['class'][0] == self.classes['zinc']:
                for i in range(random.randint(1,14)):
                    ret_train.append(roidb)
            
        return ret_train, ret_eval

    def training_roidbs(self):
        return self.rois
    
    def inference_roidbs(self):
        return self.eval_rois

def load_image(filename):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_image(raw)
    image = tf.cast(image, tf.float32)
    # image = tf.image.resize(image, (256, 256))
    tf.debugging.assert_shapes([(image, (256,256,3))])
    return image

def roidb_gen(roidbs):
    def gen():
        for roidb in roidbs:
            yield preprocess_bbox(roidb)
    return gen
#%%
def preprocess_bbox(roidb, n_classes=3, iou_thresh=0.5):
    x,y = np.meshgrid(range(0, 64*4, 4),range(0, 64*4, 4))
    x = np.reshape(x, (4096))
    y = np.reshape(y, (4096))
    xy = np.stack((x,y,x+4,y+4), axis=-1)
    # xy = xy.squeeze()
    target_rois = roidb['boxes']
    ious = boxops.ioa_1(xy, target_rois)
    ious = np.max(ious, axis=1)
    ious = np.reshape(ious, (64,64))
    class_i = roidb['class'][0] if len(roidb['class']) > 0 else 0
    mask = ious > iou_thresh
    ious[mask] = class_i
    ious[~mask] = 0
    return roidb['file_name'], ious

# %%
def datasets(base_dir, annotations_file, classes=['healthy', 'faw', 'zinc']):
    class_map = dict(zip(classes, range(len(classes))))
    roi = RandRoi(base_dir, class_map, annotations_file)
    train_roidbs = roi.training_roidbs()
    eval_roidbs = roi.inference_roidbs()

    train_ds = tf.data.Dataset.from_generator(
        roidb_gen(train_roidbs),
        (tf.string, tf.int8),
    )
    eval_ds = tf.data.Dataset.from_generator(
        roidb_gen(eval_roidbs),
        (tf.string, tf.int8),
    )

    train_ds = train_ds.map(
        lambda filename, mask: (load_image(filename), mask)
    )
    eval_ds = eval_ds.map(
        lambda filename, mask: (load_image(filename), mask)
    )
    return train_ds, eval_ds

def nlbclass_datasets(base_dir, n_classes=2, batch_size=32, eval_split=10, n_train=2000):
    
    gen = tf.keras.preprocessing.image.ImageDataGenerator()
    ds = tf.data.Dataset.from_generator(
        lambda: gen.flow_from_directory(base_dir, batch_size=batch_size),
        output_types=(tf.float32, tf.float32), 
        output_shapes=([batch_size,256,256,3], [batch_size,n_classes])
    )

    def onehot_to_index(images, labels):
        labels = tf.argmax(labels, axis=-1)
        return images, labels
    
    ds = ds.map(onehot_to_index)
        
    eval_ds = ds.take(eval_split)
    train_ds = ds.skip(eval_split).take(n_train)
    
    return train_ds, eval_ds
#%%
def mix_healthy(base_dir, annot_file, out_file, prop=0.3):
    healthy = os.listdir( os.path.join(base_dir, 'healthy') )
    count = 0

    with open(annot_file, 'r') as f:
        with open(out_file, 'w') as out_f:
            obj = json.load(f)

            ret = []

            for v in obj:
                ret.append(v)

                if random.uniform(0,1) <= prop:
                    ret.append({
                            'file_name': os.path.join('healthy', healthy[count]),
                            'boxes': [[0,0,0,0]],
                            'class': [0]
                        })
                    count += 1
            json.dump(ret, out_f)
            
# %%
_cmap_red_aplha = {
    'red': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 1, 1),
    ),
    'green': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 0, 0),
    ),
    'blue': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 0, 0),
    ),
    'alpha': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 1, 1),
    ),
}

_cmap_blue_aplha = {
    'red': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 0, 0),
    ),
    'green': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 0, 0),
    ),
    'blue': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 1, 1),
    ),
    'alpha': (
        (  0, 0, 0),
        (0.5, 0, 0),
        (  1, 1, 1),
    ),
}

cmap_red_aplha = LinearSegmentedColormap('cmap_red_aplha', _cmap_red_aplha)
cmap_blue_aplha = LinearSegmentedColormap('cmap_blue_aplha', _cmap_blue_aplha)

plt.register_cmap(cmap=cmap_red_aplha)
plt.register_cmap(cmap=cmap_blue_aplha)

def viz_class(entry, class_list):
    image, class_i = entry
    clazz = class_list[class_i]
    plt.imshow(image)
    plt.title(f'{clazz}, {class_i}')

def viz(entry, n_classes=3):
    image, mask = entry
    plt.imshow(image)
    mask = mask.astype(np.int)
    mask = np.repeat(mask, 4, axis=0)
    mask = np.repeat(mask, 4, axis=1)
    plt.imshow(mask == 1, cmap='cmap_red_aplha', vmin=0, vmax=1)
    plt.imshow(mask == 2, cmap='cmap_blue_aplha', vmin=0, vmax=1)
    plt.show()
