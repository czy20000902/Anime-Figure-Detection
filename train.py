from os.path import exists
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor, AnchorGenerator
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import argparse
from datasets.voc import VOCDetection
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='watermark')
parser.add_argument('--store_name', type=str, default='fasterrcnn_anime.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

data_transform = transforms.Compose([transforms.ToTensor()])

train_set = VOCDetection(args.data_path, image_set="train", transform=data_transform)
valid_set = VOCDetection(args.data_path, image_set="trainval", transform=data_transform)
test_set = VOCDetection(args.data_path, image_set="val", transform=data_transform)

train_loader = DataLoader(train_set, batch_size=args.batch_size)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

num_classes = 2  # 1 class (person) + background

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)


if exists(args.store_name):
    state_dict = torch.load(args.store_name)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in fasterrcnn_resnet50_fpn.state_dict()})

# Loop
for epoch in range(args.epochs):
    for index, (images, tags) in enumerate(train_loader):
        # Set the model to training mode
        model.train()
        optimizer.zero_grad()
        images = [image.to(device) for image in images]
        dic = {}
        targets = []
        dic['boxes'] = torch.Tensor([[float(tags['annotation']['object'][0]['bndbox']['xmin'][0]),
                                      float(tags['annotation']['object'][0]['bndbox']['ymin'][0]),
                                      float(tags['annotation']['object'][0]['bndbox']['xmax'][0]),
                                      float(tags['annotation']['object'][0]['bndbox']['ymax'][0])]]).to(device)
        dic['labels'] = torch.LongTensor([0]).to(device)
        targets.append(dic)

        # Forward pass
        loss_dict = model(images, targets)
        # loss_dict = model(data, tags)
        # Compute the loss
        losses = sum(loss for loss in loss_dict.values())
        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        losses.backward()

        # Update the weights
        optimizer.step()

        torch.save(model.state_dict(), args.store_name)
        print('index', index, 'in epoch', epoch)
    print('epoch :', epoch)
