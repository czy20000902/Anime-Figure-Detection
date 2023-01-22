from os.path import exists
import sys
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/input.mp4')
parser.add_argument('--output_path', type=str, default='data/output.mp4')
args = parser.parse_args()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

#Load pretrained model
if exists('fasterrcnn_anime.pth'):
    model.load_state_dict(torch.load('fasterrcnn_anime.pth'))

model = model.cuda()
model.eval()

# Read image
img = cv2.imread(args.input_path)
# Preprocess
data_transform = transforms.ToTensor()
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
img = img.cuda()
# Detect
output = model(img)

# Extract result
boxes = output[0]['boxes'].data.cpu().numpy()
scores = output[0]['scores'].data.cpu().numpy()

# Convert to numpy array
img = img[0].cpu().numpy().transpose(1, 2, 0)
img *= 255
img = np.ascontiguousarray(img, dtype=np.uint8)

# Draw bounding boxes
for i in range(len(boxes)):
    if scores[i] > 0.8:
        print(scores[i])
        box = boxes[i]
        x1, y1, x2, y2 = box
        print(box)
        cv2.rectangle(img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=20)

# Show and save results
cv2.imshow('img', img)
cv2.imwrite(args.output_path)
cv2.waitKey(0)
cv2.destroyAllWindows()
