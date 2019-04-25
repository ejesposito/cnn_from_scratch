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
        # Create the tranorm to apply to imates
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        with torch.no_grad():
            # transform the image
            image = transform(image)
            images = image.unsqueeze(dim=0).to(self.device)
            # eval
            pred_number_digits, pred_d1, pred_d2, pred_d3, pred_d4 = self.model.eval()(images)
            # find predctions
            number_digits = pred_number_digits.max(1)[1]
            d1 = pred_d1.max(1)[1]
            d2 = pred_d2.max(1)[1]
            d3 = pred_d3.max(1)[1]
            d4 = pred_d4.max(1)[1]
            # compute score of prediction confidence
            score = pred_number_digits.max(1)[0] + pred_d1.max(1)[0] + pred_d2.max(1)[0] + pred_d3.max(1)[0] + pred_d4.max(1)[0]
            """
            print('score: {}'.format(score))
            print('number digits:', number_digits.item())
            print('digits:', d1.item(), d2.item(), d3.item(), d4.item())
            """
            # return predictions
            return number_digits.item(), d1.item(), d2.item(), d3.item(), d4.item(), score
