# %%
import tensorflow as tf
import numpy as np

# %%
with tf.device("CPU"):
    resnet = tf.keras.applications.ResNet50(
        include_top=False, 
        input_shape=(256, 256, 3),
        weights=None)

# %%
print(resnet.summary())
# %%
for var in resnet.weights:
    print(f"{var.name}\n")

# %%
with tf.device("CPU"):
    saved = tf.saved_model.load("./models/simplecnn/")
#%%
for var in saved.variables:
    print(f"{var.shape}\n")

# %%
checkpoint = tf.train.Checkpoint(model=saved)
checkpoint.save("./test_ckpt")

# %%
saved = tf.saved_model.load("./test_ckpt-1")

#%%
import os
save_dir = './test_2'
save_dir = os.path.abspath( os.path.join(os.getcwd(), save_dir) )
if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
    os.mkdir( save_dir )
for var in saved.variables:
    name = var.name
    dirs_var = name.split('/')

    dirs = dirs_var[0]
    var_name = dirs_var[1]

    var_name = var_name.split(':')[0]
    secs = dirs.split('_')

    file_path = save_dir
    for sec in secs:
        if '/' not in sec:
            file_path = os.path.join(file_path, sec)
            if not (os.path.exists(file_path) and os.path.isdir(file_path)):
                os.mkdir(file_path)
    
    if var_name in ['moving_mean', 'moving_variance']:
        sec = var_name.split('_')
        var_name = 'EMA'
        sec = sec[1]
        file_path = os.path.join(file_path, sec)
        if not (os.path.exists(file_path) and os.path.isdir(file_path)):
            os.mkdir(file_path)
        pass

    try:
        # print("var_name: ", var.numpy().dtype)
        np.save(os.path.join(file_path, var_name), var.numpy())
    except AttributeError as e:
        print(f"failed to save var {var}: {e}")

# %%
import zipfile
fname = os.path.basename(save_dir)
zipf = zipfile.ZipFile(f'{save_dir}.npz', 'w', zipfile.ZIP_DEFLATED)

for root, dirs, files in os.walk(save_dir):
    for file in files:
        file = os.path.join(root, file)
        zipf.write(file, arcname=os.path.relpath(file, save_dir))
zipf.close()

# %%
with np.load(f'./{fname}.npz', allow_pickle=True) as data:
    for (file_name, arr) in data.items():
        print(f'{file_name}: {arr.shape}')
