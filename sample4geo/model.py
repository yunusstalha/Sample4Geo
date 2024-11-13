import torch
import timm
import numpy as np
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig, CLIPVisionModel, CLIPVisionConfig


class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        self.model_name = model_name.lower()
        self.output_dim = 768

        if 'segformer' in self.model_name:
            # Initialize SegFormer from Hugging Face
            segformer_model = SegformerModel.from_pretrained(model_name)
            self.model = segformer_model.base_model  # Extract the encoder
            
            segformer_hidden_channels = segformer_model.config.hidden_sizes[-1]  # 512 for segformer-b3
            
            self.projection = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Pool spatial dimensions to 1x1
                nn.Flatten(),                   # Flatten to (batch_size, hidden_channels)
                nn.Linear(segformer_hidden_channels, 1024),  # Project to 1024
                nn.ReLU()                       # Optional non-linearity
            )
        elif "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model_sat = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            self.model_pan = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=(img_size, 2 * img_size))

        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        
        if 'segformer' in self.model_name:
            config = SegformerConfig.from_pretrained(self.model_name)
            return config
        if "vit" in self.model_name:
            data_config = timm.data.resolve_model_data_config(self.model_sat)
            return data_config
            
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        if 'segformer' in self.model_name:
            # Enable or disable gradient checkpointing for SegFormer
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    if enable:
                        module.gradient_checkpointing_enable()
                    else:
                        module.gradient_checkpointing_disable()
        elif "vit" in self.model_name:
            self.model_sat.set_grad_checkpointing(enable)
            self.model_pan.set_grad_checkpointing(enable)               
        else:
            # Enable gradient checkpointing for timm models
            self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if 'segformer' in self.model_name:
            # Forward pass through SegFormer encoder
            outputs1 = self.model(img1)  # Assuming img1 is preprocessed appropriately
            if img2 is not None:
                outputs2 = self.model(img2)
                # Extract last_hidden_state
                features1 = outputs1.last_hidden_state  # Shape: [B, 512, 16, 32]
                features2 = outputs2.last_hidden_state
                # Apply projection
                features1 = self.projection(features1)  # Shape: [B, output_dim]
                features2 = self.projection(features2)
                return features1, features2
            else:
                # Extract last_hidden_state
                features = outputs1.last_hidden_state  # Shape: [B, 512, 16, 32]
                # Apply projection
                features = self.projection(features)    # Shape: [B, output_dim]
                return features
        elif "vit" in self.model_name:
            # Forward pass through tim
            if img1 is not None and img2 is not None:
                image_features1 = self.model_pan(img1)     
                image_features2 = self.model_sat(img2)
                
                return image_features1, image_features2
            elif img1 is not None: 
                image_features = self.model_pan(img1)
                return image_features
            elif img2 is not None:
                image_features = self.model_sat(img2)
                return image_features 
                
        else:
            # Forward pass through timm model
            if img2 is not None:
                image_features1 = self.model(img1)     
                image_features2 = self.model(img2)
                
        
                return image_features1, image_features2            
                  
            else:
                image_features = self.model(img1)
                 
                return image_features