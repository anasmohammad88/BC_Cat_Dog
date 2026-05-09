import torch
import torch.nn as nn

class CatDogClassification(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            
            self.con1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1=nn.BatchNorm2d(out_channels)
            self.act=nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.con2=nn.Conv2d(in_channels= out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2=nn.BatchNorm2d(out_channels)
           
            # Residual if needed
            if in_channels != out_channels:
                self.proj=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                self.proj=nn.Identity()
                
        def forward(self, x):
            out = self.dropout(self.act(self.bn1(self.con1(x))))
            out = self.bn2(self.con2(out))
            out += self.proj(x)
            return self.act(out)
            
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        in_channels = 16
        model_layers = []
        
        for layers in [32,64,128,256]:
            model_layers.append(self.Block(in_channels=in_channels, out_channels=layers))
            model_layers.append(nn.MaxPool2d(2))
            in_channels = layers
        
        self.features = nn.Sequential(*model_layers)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 1)  # logits output
        )     
        

    def forward(self, x):
        return self.classifier(self.features(self.stem(x)))