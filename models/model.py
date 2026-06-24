import torch.nn as nn
from torchvision import models

def build_model(num_classes=10):
    """
    Builds an EfficientNet-B0 model for EuroSAT classfication.
    """

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Replace the final classification layer

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_model(num_classes=10)
    print(model)