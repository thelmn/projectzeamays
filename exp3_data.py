# %%
import os
import json
import random
import cv2
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

        ret = []

        for v in obj:
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
            ret.append(roidb)
        
        n = int(len(ret) * train_split)
        ret_train = ret[:n]
        ret_eval = ret[n:]
        
        return ret_train, ret_eval

    def training_roidbs(self):
        print("Training images:", len(self.rois))
        return self.rois
    
    def inference_roidbs(self):
        print("Eval images:", len(self.eval_rois))
        return self.eval_rois

def load_image(filename):
    tf.print(filename)
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

# %%
def oversample_zinc(base_dir, annot_file, out_file, prop=0.3):
    with open(annot_file, 'r') as f:
        with open(out_file, 'w') as out_f:
            obj = json.load(f)

            ret = []

            for v in obj:
                ret.append(v)
                # attempt to balance the zinc class
                if len(v['class']) > 0 and v['class'][0] == 'zinc':
                    for i in range(random.randint(1,4)):
                        ret.append(v)
            print(len(obj), len(ret))
            json.dump(ret, out_f)
# %%
def fix_no_boxes(base_dir, annot_file, out_file):
    with open(annot_file, 'r') as f:
        with open(out_file, 'w') as out_f:
            obj = json.load(f)
            ret = []
            for v in obj:
                clazz = v['file_name'].split('/')[0]
                assert clazz in ['healthy','faw','zinc'], v['file_name']

                if len(v['class']) == 0:
                    v['class'] = [clazz]
                if len(v['boxes']) == 0:
                    v['boxes'] = [[0,0,0,0]]
                ret.append(v)
            json.dump(ret, out_f)

# %%
def count_classes(base_dir, annot_file):
    counts = {
        'healthy': 0,
        'faw': 0,
        'zinc': 0
    }
    with open(annot_file, 'r') as f:
        obj = json.load(f)
        for v in obj:
            clazz = v['class'][0]
            assert clazz in list(counts.keys()), v['file_name']
            counts[clazz] += 1
    print(counts)
# %%
def shuffle_ds(base_dir, annot_file, out_file):
    with open(annot_file, 'r') as f:
        with open(out_file, 'w') as out_f:
            obj = json.load(f)
            print('before shuffle: ', len(obj))
            random.shuffle(obj)
            print('after shuffle: ', len(obj))
            json.dump(obj, out_f)

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
                            'class': ['healthy']
                        })
                    count += 1
            print(len(obj), len(ret))
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

#%%
def viz_nmasks(image, masks, n_classes=3):
    n = len(masks)
    n_sq = int(np.floor(np.sqrt(n+1)))
    fig, ax = plt.subplots(n_sq, n_sq)
    image = image.astype(np.int)
    # print('image', image.dtype, np.min(image), np.max(image))
    ax[0,0].imshow(image)
    fig.tight_layout = True
    plt.imshow(image)
    for i, mask in enumerate(masks):
        ax_i = (i+1)//n_sq
        ax_j = (i+1)%n_sq
        print('mask_i', i, mask.dtype, np.min(mask), np.max(mask))
        # mask = mask.astype(np.int)
        ax[ax_i,ax_j].imshow(image)
        ax[ax_i,ax_j].imshow(mask[:,:,0], cmap='cmap_red_aplha', vmin=0, vmax=1)
        ax[ax_i,ax_j].imshow(mask[:,:,1], cmap='cmap_blue_aplha', vmin=0, vmax=1)
    plt.show()

# %%
def smooth_mask(mask):
    smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return smooth

def contour_mask(mask):
    mask = np.array(mask, dtype=int)
    # print(mask.dtype, mask.shape, np.min(mask), np.max(mask))
    if np.min(mask) == np.max(mask):
        return mask
    mask = np.repeat(mask, 4, axis=0)
    mask = np.repeat(mask, 4, axis=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (255, 0, 0) # green - color for contours
        color = (0, 255, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 2, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 2, 8)
    print(np.count_nonzero(drawing), np.product(drawing.shape))
    drawing = drawing[:,:,1]
    drawing = cv2.resize(drawing, (256,256), interpolation=cv2.INTER_CUBIC)
    print(drawing.shape, np.count_nonzero(drawing), np.product(drawing.shape))
    return drawing

# %%
