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
parser.add_argument('--data_path', type=str, default='voc_dataset')
parser.add_argument('--store_name', type=str, default='fasterrcnn_anime.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()


def collate_fn(batch):
    return tuple(zip(*batch))


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2  # 1 class (person) + background

model = fasterrcnn_resnet50_fpn()
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)
model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
                                                            num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler)
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

data_transform = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.ToTensor()])

train_set = VOCDetection(args.data_path, image_set="train", transform=data_transform)
# valid_set = VOCDetection(args.data_path, image_set="trainval")
# test_set = VOCDetection(args.data_path, image_set="val")

train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn)
# valid_loader = DataLoader(valid_set)
# test_loader = DataLoader(test_set)

# Loop
for epoch in range(args.epochs):
    for images, targets in train_loader:
        # Set the model to training mode
        model.train()
        optimizer.zero_grad()
        # targets = {}
        # targets['boxes'] = torch.Tensor([[float(tags['annotation']['object'][0]['bndbox']['xmin'][0]),
        #                                   float(tags['annotation']['object'][0]['bndbox']['ymin'][0]),
        #                                   float(tags['annotation']['object'][0]['bndbox']['xmax'][0]),
        #                                   float(tags['annotation']['object'][0]['bndbox']['ymax'][0])]]).to(device)
        # targets['labels'] = torch.LongTensor([0]).to(device)
        # targets = [targets]

        images = [image.to(device) for image in images]
        tags = []
        for target in targets:
            boxes = torch.Tensor([[float(target['annotation']['object'][0]['bndbox']['xmin']),
                                   float(target['annotation']['object'][0]['bndbox']['ymin']),
                                   float(target['annotation']['object'][0]['bndbox']['xmax']),
                                   float(target['annotation']['object'][0]['bndbox']['ymax'])]]).to(device)
            print(boxes)
            labels = torch.LongTensor([1]).to(device)
        tags.append({'boxes': boxes, 'labels': labels})
        # Forward pass
        loss_dict = model(images, tags)
        # loss_dict = model(data, tags)
        # Compute the loss
        losses = sum(loss for loss in loss_dict.values())
        for name in loss_dict:
            print(loss_dict[name].item())
        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        losses.backward()

        # Update the weights
        optimizer.step()

        torch.save(model.state_dict(), args.store_name)
