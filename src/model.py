import torch
import torch.nn as nn
import timm


class EfficientNetBinary(nn.Module):
    """
    EfficientNet-based binary classifier for DR detection
    
    Args:
        model_name: EfficientNet variant (e.g., 'efficientnet_b3')
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, model_name='efficientnet_b3', pretrained=True, dropout=0.3):
        super(EfficientNetBinary, self).__init__()
        
        # Load pretrained backbone (without classification head)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            n_features = features.shape[1]
        
        # Binary classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 2)  # 2 classes for binary classification
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class ViTBinary(nn.Module):
    """
    Vision Transformer-based binary classifier for DR detection
    
    Args:
        model_name: ViT variant (e.g., 'vit_base_patch16_224')
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, dropout=0.3):
        super(ViTBinary, self).__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        n_features = self.backbone.num_features
        
        # Binary classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Dropout(dropout),
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class DenseNetBinary(nn.Module):
    """
    DenseNet-based binary classifier for DR detection
    
    Args:
        model_name: DenseNet variant (e.g., 'densenet121')
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
    """
    
    def __init__(self, model_name='densenet121', pretrained=True, dropout=0.3):
        super(DenseNetBinary, self).__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Binary classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def create_model(model_name='efficientnet_b3', pretrained=True, dropout=0.3, device='cuda'):
    """
    Factory function to create model
    
    Args:
        model_name: Name of the model architecture
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        Model instance
    """
    
    if 'efficientnet' in model_name:
        model = EfficientNetBinary(model_name, pretrained, dropout)
    elif 'vit' in model_name:
        model = ViTBinary(model_name, pretrained, dropout)
    elif 'densenet' in model_name:
        model = DenseNetBinary(model_name, pretrained, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Weighting factor for class balance (default: 0.25)
        gamma: Focusing parameter for hard examples (default: 2.0)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_function(loss_name='focal', alpha=0.25, gamma=2.0, pos_weight=None):
    """
    Get loss function by name
    
    Args:
        loss_name: Type of loss ('focal', 'ce', 'weighted_ce')
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        pos_weight: Positive class weight for weighted CE
    
    Returns:
        Loss function
    """
    
    if loss_name == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'weighted_ce':
        if pos_weight is not None:
            weight = torch.tensor([1.0, pos_weight])
            return nn.CrossEntropyLoss(weight=weight)
        else:
            return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Test the model
if __name__ == "__main__":
    # Test model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing EfficientNet-B3...")
    model = create_model('efficientnet_b3', pretrained=False, device=device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting ViT...")
    model = create_model('vit_base_patch16_224', pretrained=False, device=device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.tensor([0, 1]).to(device)
    loss = focal_loss(output, targets)
    print(f"Loss value: {loss.item():.4f}")
