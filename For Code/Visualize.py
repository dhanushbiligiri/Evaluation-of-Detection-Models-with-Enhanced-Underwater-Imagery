import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import random

# Replace with your actual file paths
regular_folder = '/home/campus30/dnarsipu/Downloads/For Code/Regular/images/train'
enhanced_folder = '/home/campus30/dnarsipu/Downloads/For Code/Enhanced/images/train'

# Get a list of image filenames in the regular train folder
regular_images = os.listdir(regular_folder)
# Randomly select 5 images
selected_images = random.sample(regular_images, 5)

# Create a grid for the subplots
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 5, height_ratios=[0.5, 0.5])

# Adjust the space between rows
gs.update(wspace=0.005, hspace=0)

# Create the subplots
for i, img_name in enumerate(selected_images):
    reg_img_path = os.path.join(regular_folder, img_name)
    enh_img_path = os.path.join(enhanced_folder, img_name)

    # Regular image subplot
    ax_reg = plt.subplot(gs[0, i])
    img = Image.open(reg_img_path)
    ax_reg.imshow(img)
    ax_reg.axis('off')
    if i == 0:
        ax_reg.set_title('Regular Images', rotation='vertical', x=-0.1, y=0.1, fontsize=16)

    # Enhanced image subplot
    ax_enh = plt.subplot(gs[1, i])
    img = Image.open(enh_img_path)
    ax_enh.imshow(img)
    ax_enh.axis('off')
    if i == 0:
        ax_enh.set_title('Enhanced Images', rotation='vertical', x=-0.1, y=0.1, fontsize=16)

# Show the plot
plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.20, wspace=0.05, hspace=0)
plt.show()
