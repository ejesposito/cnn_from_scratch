import argparse
import torch

from PIL import Image
from torchvision import transforms


class Predictor(object):

    def __init__(self, model, cuda):
        # model to gpu if it's available
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device(cuda)
        self.model = model.to(self.device)

    def predict(self, image):
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.CenterCrop([54, 54]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            image = transform(image)
            images = image.unsqueeze(dim=0).to(self.device)

            length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits = self.model.eval()(images)

            length_prediction = length_logits.max(1)[1]
            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]
            digit3_prediction = digit3_logits.max(1)[1]
            digit4_prediction = digit4_logits.max(1)[1]

            print('length:', length_prediction.item())
            print('digits:', digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(), digit4_prediction.item())
