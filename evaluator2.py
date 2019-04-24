import torch
import torch.utils.data
from torchvision import transforms

from dataset2 import TorchDataSet


class Evaluator(object):
    def __init__(self, test):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = torch.utils.data.DataLoader(TorchDataSet(test['image'],
                                                                test[['number_digits', 'd1', 'd2', 'd3', 'd4']],
                                                                transform),
                                                   batch_size=128, shuffle=False)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')

    def evaluate(self, model):
        num_correct = 0
        needs_include_length = False

        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
                images, length_labels, digits_labels = images.to(self.device), length_labels.to(self.device), [digit_labels.to(self.device) for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits = model.eval()(images)

                length_prediction = length_logits.max(1)[1]
                digit1_prediction = digit1_logits.max(1)[1]
                digit2_prediction = digit2_logits.max(1)[1]
                digit3_prediction = digit3_logits.max(1)[1]
                digit4_prediction = digit4_logits.max(1)[1]

                if needs_include_length:
                    num_correct += (length_prediction.eq(length_labels) &
                                    digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3])).cpu().sum()
                else:
                    num_correct += (digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3])).cpu().sum()

        accuracy = num_correct.item() / len(self._loader.dataset)
        return accuracy
