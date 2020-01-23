import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


### OUTPUTS
# NYU:
# (480, 640, 3) uint8 0 255
# (480, 640) float32 0.0 0.035614558
# 100%|██████████| 197/197 [00:08<00:00, 23.95it/s]
# global min/max RGB: 0/255
# global min/max D: 0.0/0.15259021520614624
# mean/std RGB: [82.81281129 69.98840933 67.55189782]/[68.93296759 68.55954934 69.47609191]
# mean/std D: 0.022469453513622284/0.01488421019166708
# ===
#
# IN:
# (9600, 640, 3) float32 0.0 0.99607843
# (9600, 640) float32 0.007721065 0.06131075
# 100%|██████████| 50/50 [00:44<00:00,  1.14it/s]
# global min/max RGB: 0.0/255.0
# global min/max D: 0.0/0.12114137411117554
# mean/std RGB: [130.67242 128.60162 125.68408]/[49.410275 50.143192 51.87701 ]
# mean/std D: 0.039533697068691254/0.014748318120837212


IN_PATH = "/Users/florian/intnet-bedrooms-png-sample/"
NYU_PATH = "/Volumes/dell/nyu-depth-v2/sync/bedroom_0004/"

rgb_imgs = [x for x in os.listdir(NYU_PATH) if x[:4] == "rgb_"]
depth_imgs = [x for x in os.listdir(NYU_PATH) if x[:11] == "sync_depth_"]

rgb_imgs.sort()
depth_imgs.sort()

print("NYU:")

# load sample image from NYU
sample_rgb = plt.imread(os.path.join(NYU_PATH, rgb_imgs[0]))
print(sample_rgb.shape, sample_rgb.dtype, sample_rgb.min(), sample_rgb.max())

# sample depth
sample_depth = plt.imread(os.path.join(NYU_PATH, depth_imgs[0]))
print(sample_depth.shape, sample_depth.dtype, sample_depth.min(),
      sample_depth.max())

## find global min/max/mean/std on color and depth

nyu_mean_rgb = []
nyu_std_rgb = []
nyu_mean_d = []
nyu_std_d = []

nyu_min_rgb = np.array([np.inf])
nyu_max_rgb = np.array([-np.inf])
nyu_min_d = np.array([np.inf])
nyu_max_d = np.array([-np.inf])

for i in trange(len(rgb_imgs)):
    sample_rgb = plt.imread(os.path.join(NYU_PATH, rgb_imgs[i]))
    sample_depth = plt.imread(os.path.join(NYU_PATH, depth_imgs[i]))

    nyu_min_rgb = np.min([nyu_min_rgb, sample_rgb.min()])
    nyu_max_rgb = np.max([nyu_max_rgb, sample_rgb.max()])
    nyu_min_d = np.min([nyu_min_d, sample_depth.min()])
    nyu_max_d = np.max([nyu_max_d, sample_depth.max()])

    nyu_mean_rgb.append(sample_rgb.mean(axis=(0,1)))
    nyu_std_rgb.append(sample_rgb.std(axis=(0,1)))
    nyu_mean_d.append(sample_depth.mean())
    nyu_std_d.append(sample_depth.std())


print(f"global min/max RGB: {nyu_min_rgb}/{nyu_max_rgb}")
print(f"global min/max D: {nyu_min_d}/{nyu_max_d}")
print(f"mean/std RGB: {np.mean(nyu_mean_rgb, axis=0)}/{np.mean(nyu_std_rgb, axis=0)}")
print(f"mean/std D: {np.mean(nyu_mean_d)}/{np.mean(nyu_std_d)}")
print ("===\n\nIN:")

## IN
rgb_imgs = [x for x in os.listdir(IN_PATH) if x[-8:] == "-rgb.png"]
depth_imgs = [x for x in os.listdir(IN_PATH) if x[-6:] == "-d.png"]
rgb_imgs.sort()
depth_imgs.sort()

# load sample image from NYU
sample_rgb = plt.imread(os.path.join(IN_PATH, rgb_imgs[0]))
print(sample_rgb.shape, sample_rgb.dtype, sample_rgb.min(), sample_rgb.max())

# sample depth
sample_depth = plt.imread(os.path.join(IN_PATH, depth_imgs[0]))
print(sample_depth.shape, sample_depth.dtype, sample_depth.min(),
      sample_depth.max())



in_mean_rgb = []
in_std_rgb = []
in_mean_d = []
in_std_d = []

in_min_rgb = np.array([np.inf])
in_max_rgb = np.array([-np.inf])
in_min_d = np.array([np.inf])
in_max_d = np.array([-np.inf])

for i in trange(len(rgb_imgs)):
    sample_rgb = plt.imread(os.path.join(IN_PATH, rgb_imgs[i]))*255
    sample_depth = plt.imread(os.path.join(IN_PATH, depth_imgs[i]))
    print (sample_depth.dtype)

    in_min_rgb = np.min([in_min_rgb, sample_rgb.min()])
    in_max_rgb = np.max([in_max_rgb, sample_rgb.max()])
    in_min_d = np.min([in_min_d, sample_depth.min()])
    in_max_d = np.max([in_max_d, sample_depth.max()])

    in_mean_rgb.append(sample_rgb.mean(axis=(0,1)))
    in_std_rgb.append(sample_rgb.std(axis=(0,1)))
    in_mean_d.append(sample_depth.mean())
    in_std_d.append(sample_depth.std())

print(f"global min/max RGB: {in_min_rgb}/{in_max_rgb}")
print(f"global min/max D: {in_min_d}/{in_max_d}")
print(f"mean/std RGB: {np.mean(in_mean_rgb, axis=0)}/{np.mean(in_std_rgb, axis=0)}")
print(f"mean/std D: {np.mean(in_mean_d)}/{np.mean(in_std_d)}")