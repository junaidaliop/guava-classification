# models/model.py

import torch.nn as nn
import timm

class GuavaClassifier(nn.Module):
    def __init__(self, num_classes=3):
        """
        Initializes the GuavaClassifier with an optional pretrained rdnet_base backbone.
        
        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(GuavaClassifier, self).__init__()
        
        # Initialize the rdnet_base model from timm (load pretrained weights by default)
        self.model = timm.create_model('rdnet_base', pretrained=True)  # or False if you want to train from scratch

        # To freeze
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the classifier head with a new linear layer
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'fc'):
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes)
            print("Replaced 'head.fc' with new Linear layer.")
        else:
            raise AttributeError("The model does not have 'head.fc' attribute.")
        
        # Ensure that the new classifier head's parameters are trainable
        for param in self.model.head.parameters():
            param.requires_grad = True
        print("Set 'head' parameters to require gradients.")

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output logits.
        """
        x = self.model(x)
        return x