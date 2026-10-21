import torch
import torch.nn as nn
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_transfer_model(model_name: str = "resnet50", strategy: str = "partial"):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )
        
        if strategy == "frozen":
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif strategy == "partial":
            for name, param in model.named_parameters():
                if not ("layer4" in name or "fc" in name):
                    param.requires_grad = False
        elif strategy == "full":
            for param in model.parameters():
                param.requires_grad = True

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )
        
        if strategy == "frozen":
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        elif strategy == "partial":
            for name, param in model.named_parameters():
                if not ("denseblock4" in name or "norm5" in name or "classifier" in name):
                    param.requires_grad = False
        elif strategy == "full":
            for param in model.parameters():
                param.requires_grad = True

    return model