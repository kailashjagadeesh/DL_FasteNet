#!/usr/bin/env python3
import time
import os
import sys

import cv2
import numpy as np
from numpy import random as nprand
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from helpers import helpers
from ImageLoader import ImageLoader
from FasteNet_Net_v2 import FasteNet_v2

from FasteNet_Vanilla_Net import FasteNet_Vanilla
from FasteNet_Large_Net import FasteNet_Large

# params
DIRECTORY = 'C:\AI\DATA' # os.path.dirname(__file__)
DIRECTORY2 = 'C:\AI\WEIGHTS'
SHUTDOWN_AFTER_TRAINING = False

VERSION_NUMBER = 5
MARK_NUMBER = 505

BATCH_SIZE = 100
NUMBER_OF_IMAGES = 700
NUMBER_OF_CYCLES = 5

# instantiate helper object
helpers = helpers(mark_number=MARK_NUMBER, version_number=VERSION_NUMBER, weights_location=DIRECTORY2)
device = helpers.get_device()

def generate_dataloader(index):
    # holder for images and groundtruths and lowest running loss
    images = []
    truths = []

    from_image = int(index * NUMBER_OF_IMAGES/NUMBER_OF_CYCLES)
    to_image = int((index+1) * NUMBER_OF_IMAGES/NUMBER_OF_CYCLES)

    # read in the images
    for i in range(from_image, to_image):
        image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{i}.png')
        truth_path = os.path.join(DIRECTORY, f'Dataset/label/label_{i}.png')

        if os.path.isfile(image_path): 

            # read images
            image = TF.to_tensor(cv2.imread(image_path))[0]
            truth = TF.to_tensor(cv2.imread(truth_path))[0]
            
            # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
            image /= torch.max(image + 1e-6)
            truth /= torch.max(truth + 1e-6)

            images.append(image)
            truths.append(truth)

    print(f'Attempted to load images {from_image} to {to_image}, actually loaded {len(images)} images.')


    # feed data to the ImageLoader and start the dataloader to generate batches
    dataset = ImageLoader(images, truths, (images[0].shape[0], images[0].shape[1]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader

# uncomment this to one to view the output of the dataset
# helpers.peek_dataset(dataloader=dataloader)

# set up net
FasteNet = FasteNet_v2().to(device)

# get latest weight file
weights_file = helpers.get_latest_weight_file()
if weights_file != -1:
    FasteNet.load_state_dict(torch.load(weights_file))

# set up loss function and optimizer and load in data
loss_function = nn.MSELoss()
optimizer = optim.Adam(FasteNet.parameters(), lr=1e-6, weight_decay=1e-2)

# get network param number and gflops
# image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{1}.png')
# image = TF.to_tensor(cv2.imread(image_path))[0].unsqueeze(0).unsqueeze(0)[..., :1600].to(device)
# image /= torch.max(image + 1e-6)
# helpers.network_stats(FasteNet, image)


# HARD NEGATIVE MINING
# HARD NEGATIVE MINING
# HARD NEGATIVE MINING

# FOR INFERENCING
# FOR INFERENCING
# FOR INFERENCING

# set frames to render > 0 to perform inference
torch.no_grad()
FasteNet.eval()
frames_to_render = 100
start_time = time.time()


# set to true for inference
for _ in range(frames_to_render):
    index = nprand.randint(0, 996)

    image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{index}.png')
    
    # read images
    image = TF.to_tensor(cv2.imread(image_path))[0]
    
    # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
    image /= torch.max(image + 1e-6)
    
    input = image.unsqueeze(0).unsqueeze(0).to(device)[..., :1600]
    saliency_map = FasteNet.forward(input)
    torch.cuda.synchronize()

    # draw contours on original image and prediction image
    contour_image, contour_number = helpers.saliency_to_contour(input=saliency_map, original_image=input, fastener_area_threshold=0, input_output_ratio=8)

    # use this however you want to use it
    image_image = np.array(cv2.imread(image_path)[..., 0][:, :1600], dtype=np.float64)
    image_image /= 205 # approximate multiplier, I don't actually know the scale anymore at this point
    fused_image = np.transpose(np.array([contour_image, image_image]), [1, 0])

    # set to true to display images
    if True:
        figure = plt.figure()

        figure.add_subplot(2, 2, 1)
        plt.title(f'Input Image: Index {index}')
        plt.imshow(input.squeeze().to('cpu').detach().numpy(), cmap='gray')
        figure.add_subplot(2, 2, 2)
        plt.title('Saliency Map')
        plt.imshow(saliency_map.squeeze().to('cpu').detach().numpy(), cmap='gray')
        figure.add_subplot(2, 2, 3)
        plt.title('Predictions')
        plt.imshow(fused_image)
        plt.title(f'Predicted Number of Fasteners in Image: {contour_number}')
        
        plt.show()

end_time = time.time()
duration = end_time - start_time
print(f"Average FPS = {frames_to_render / duration}")
