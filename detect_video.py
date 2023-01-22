from os.path import exists
import sys
import torch
import torchvision
from torchvision import transforms
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

data_transform = transforms.ToTensor()

input_video = cv2.VideoCapture(args.input_path)
fps = input_video.get(cv2.CAP_PROP_FPS)
size = (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter("output_video.mp4",
                              cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # encoder
                              fps,
                              size)


success, frame = input_video.read()
while success:  # Loop until all frames have been processed
    img = data_transform(frame)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()

    output = model(img)
    # Extract results
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()

    img = img[0].cpu().numpy().transpose(1, 2, 0)
    print(len(img))
    print(len(img[0]))
    img *= 255
    img = np.ascontiguousarray(img, dtype=np.uint8)
    # Draw bounding boxes
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            print(scores[i])
            box = boxes[i]
            x1, y1, x2, y2 = box
            print(box)
            cv2.rectangle(frame, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)
    # Write to output video
    videoWriter.write(frame)
    success, frame = input_video.read()
videoWriter.release()
