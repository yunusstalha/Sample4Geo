import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

# class InfoNCE(nn.Module):

#     def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         super().__init__()
        
#         self.loss_function = loss_function
#         self.device = device

#     def forward(self, image_features1, image_features2, logit_scale):
#         image_features1 = F.normalize(image_features1, dim=-1)
#         image_features2 = F.normalize(image_features2, dim=-1)
        
#         logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
#         logits_per_image2 = logits_per_image1.T
        
#         labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
#         loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

#         return loss  
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, text_features1, text_features2, logit_scale):
        # Normalize all features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        text_features1 = F.normalize(text_features1, dim=-1)
        text_features2 = F.normalize(text_features2, dim=-1)
        
        # Compute logits
        # Image-Image similarities
        logits_per_image = logit_scale * image_features1 @ image_features2.T  # [batch_size, batch_size]
        logits_per_image_T = logits_per_image.T  # Transpose for symmetry

        # Image-Text similarities
        logits_image1_text1 = logit_scale * image_features1 @ text_features1.T  # [batch_size, batch_size]
        logits_image2_text2 = logit_scale * image_features2 @ text_features2.T

        # # Cross-modal similarities (optional)
        # logits_image1_text2 = logit_scale * image_features1 @ text_features2.T
        # logits_image2_text1 = logit_scale * image_features2 @ text_features1.T

        # Labels
        labels = torch.arange(len(logits_per_image), dtype=torch.long, device=self.device)
        
        # Compute losses
        # Image-Image loss
        loss_image_image = (self.loss_function(logits_per_image, labels) + self.loss_function(logits_per_image_T, labels)) / 2

        # Image-Text losses
        loss_image_text1 = (self.loss_function(logits_image1_text1, labels) + self.loss_function(logits_image1_text1.T, labels)) / 2
        loss_image_text2 = (self.loss_function(logits_image2_text2, labels) + self.loss_function(logits_image2_text2.T, labels)) / 2

        # # Cross-modal losses (optional)
        # loss_cross_image1_text2 = (self.loss_function(logits_image1_text2, labels) + self.loss_function(logits_image1_text2.T, labels)) / 2
        # loss_cross_image2_text1 = (self.loss_function(logits_image2_text1, labels) + self.loss_function(logits_image2_text1.T, labels)) / 2

        # Total loss
        loss = (
            loss_image_image +
            loss_image_text1 +
            loss_image_text2 
        ) / 3

        return loss  

