import torch
from model import AIMnet  # Ensure AIMnet is defined or imported correctly
import os
from PIL import Image
import shutil
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Grayscale

def prepare_la_input(image_path):
    # Assuming 'la' needs a grayscale version of the image repeated across 3 channels
    img = Image.open(image_path).convert('RGB')  # Open the image and ensure it is in RGB
    grayscale_transform = Grayscale()  # Create a Grayscale transform
    la_input = grayscale_transform(img)  # Apply the transformation
    la_input = ToTensor()(la_input)  # Convert to tensor
    la_input = la_input.repeat(3, 1, 1)  # Repeat the grayscale image across 3 channels
    la_input = la_input.unsqueeze(0)  # Add a batch dimension
    return la_input


def load_model(model_path, model):
    # Load the state dictionary from the file
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
    
    # Adjust the keys if they start with 'module.' (used in models trained with DataParallel)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the adjusted state dictionary into the model
    model.load_state_dict(new_state_dict)

model_path = r'/home/campus30/dnarsipu/Downloads/Research - Ali/Semi-UIR-masterEnhancement/pretrained/model.pth'
model = AIMnet()  # Initialize your model
load_model(model_path, model)


def prepare_la_input(image_path):
    # This is a placeholder function. You need to replace it with actual processing steps
    # that prepare whatever input 'la' is supposed to represent.
    img = Image.open(image_path).convert('RGB')  # Adjust this according to actual needs
    la_tensor = ToTensor()(img)  # Example transformation, adjust as necessary
    return la_tensor.unsqueeze(0)  # Add batch dimension if needed

def enhance_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for label_file in os.listdir(labels_dir):
        src_label_path = os.path.join(labels_dir, label_file)
        dest_label_path = os.path.join(output_labels_dir, label_file)
        shutil.copy2(src_label_path, dest_label_path)

    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        img_tensor = ToTensor()(img).unsqueeze(0)
        la_input = prepare_la_input(image_path)  # Prepare 'la' input for each image

        with torch.no_grad():
            output = model(img_tensor, la_input)  # Now 'la_input' is defined
            enhanced_tensor = output[0] if isinstance(output, tuple) else output

        enhanced_img = ToPILImage()(enhanced_tensor.squeeze(0))
        enhanced_img.save(os.path.join(output_images_dir, image_file))

# Assuming other setup code and function calls remain the same



train_input_dir = r'/home/campus30/dnarsipu/Downloads/Research - Ali/RUOD800_yolo_split/test'
train_output_dir = r'/home/campus30/dnarsipu/Downloads/Research - Ali/RUOD800_yolo_split/Enhanced Test'
enhance_images(train_input_dir, train_output_dir)