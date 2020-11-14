#%%
import os
import json
import random

#%%
def random_rois(parent, img_dim=256):
    with open(os.path.join(parent, "rand_rois.json"), "w") as f_json:
        obj = []
        for folder in os.listdir(parent):
            folder = os.path.join(parent, folder)
            if not os.path.isdir(folder):
                continue
            folder_name = os.path.basename(folder)
            for f in os.listdir(folder):
                f = f"{folder_name}/{f}"
                if f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".jpeg") or f.endswith(".JPEG"):
                    n_rois = random.randint(1, 4)
                    classes = [folder_name]*n_rois
                    boxes = [ [0,0,0,0] ]*n_rois
                    for i in range(n_rois):
                        x1 = random.random() * img_dim
                        y1 = random.random() * img_dim
                        x2 = random.random() * img_dim
                        y2 = random.random() * img_dim
                        boxes[i] = [ min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2) ]

                    obj.append({
                        'file_name': f,
                        'class': classes,
                        'boxes': boxes,
                        })
        json.dump(obj, f_json)

#%%
random_rois('./data/final')

# %%
