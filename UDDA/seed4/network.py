import torch 
import torch.nn as nn
import torch.nn.functional as f

class Encoder(nn.Module):
    """
    MLP encoder model for our model
    """
    def __init__(self, num_fts):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_fts, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ClassClassifier(nn.Module):
    def __init__(self, num_cls):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(64, num_cls)

    def forward(self, x):
        x = self.classifier(x)
        return x

