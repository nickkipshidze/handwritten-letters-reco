import torch, os
from torch import nn

class AlphaModel(nn.Module):
    def __init__(self, in_shape, hidden_units, out_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=in_shape, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_shape),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.layer_stack.forward(X)
    
class ConvAlphaModel(nn.Module):
    def __init__(self, in_shape, hidden_units, out_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=5, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*121, out_features=out_shape),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.classifier.forward(
            self.conv_block_2.forward(
                self.conv_block_1.forward(X)
            )
        )

class Model:
    def __init__(self, path):
        if os.path.exists(path):
            self.model = torch.load(path, map_location=torch.device("cpu"))
        else:
            raise FileNotFoundError("Path to model does not exist")
        self.class_names = ["ა", "ბ", "გ", "დ", "ე", "ვ", "ზ", "თ", "ი", "კ", "ლ", "მ", "ნ", "ო", "პ", "ჟ", "რ", "ს", "ტ", "უ", "ფ", "ქ", "ღ", "ყ", "შ", "ჩ", "ც", "ძ", "წ", "ჭ", "ხ", "ჯ", "ჰ"]
    
    def predict(self, X):
        X = torch.Tensor(X).reshape(28, 28).unsqueeze(0).unsqueeze(0) / 255
        X = X.flip(2)
        # X = torch.unsqueeze(X.flatten(), 0)
        y_logits = self.model.forward(X)
        return self.class_names, y_logits[0].tolist()