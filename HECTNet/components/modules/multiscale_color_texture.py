import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GaborFilter(nn.Module):
    """
    Implementation of Gabor Filters as a convolutional layer.
    Multiple orientations and scales can be specified to create a bank of filters.
    """
    def __init__(self, in_channels=3, out_channels=16, kernel_size=15, 
                 num_orientations=4, num_scales=2):
        super(GaborFilter, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # Each input channel will have multiple Gabor filters
        # Total filters = in_channels * num_orientations * num_scales
        gabor_filters = self._create_gabor_filters()
        
        # Register the filters as a parameter of the module
        self.register_parameter('weight', nn.Parameter(gabor_filters, requires_grad=False))
        self.padding = kernel_size // 2  # Same padding
        
    def _create_gabor_filters(self):
        """
        Creates a bank of Gabor filters with various orientations and scales.
        """
        # Initialize tensor for all filters
        filters = torch.zeros(self.out_channels, self.in_channels, 
                             self.kernel_size, self.kernel_size)
        
        # We'll distribute the filters across output channels
        filters_per_input = self.out_channels // self.in_channels
        orientations_per_scale = filters_per_input // self.num_scales
        
        # Create meshgrid for kernel
        center = self.kernel_size // 2
        y, x = torch.meshgrid(torch.arange(self.kernel_size) - center, 
                             torch.arange(self.kernel_size) - center, indexing='ij')
        
        # Parameters for Gabor filters
        sigmas = [3.0 + i*2.0 for i in range(self.num_scales)]  # Standard deviation
        lambdas = [4.0 + i*4.0 for i in range(self.num_scales)]  # Wavelength
        gamma = 0.5  # Aspect ratio
        psi = 0  # Phase offset
        
        filter_idx = 0
        for in_channel in range(self.in_channels):
            for scale_idx in range(self.num_scales):
                sigma = sigmas[scale_idx]
                lam = lambdas[scale_idx]
                
                for ori_idx in range(orientations_per_scale):
                    theta = ori_idx * math.pi / orientations_per_scale
                    
                    # Apply rotation to coordinates
                    x_theta = x * math.cos(theta) + y * math.sin(theta)
                    y_theta = -x * math.sin(theta) + y * math.cos(theta)
                    
                    # Gabor function
                    gb = torch.exp(
                        -0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2
                    ) * torch.cos(2 * math.pi * x_theta / lam + psi)
                    
                    # Normalize the filter
                    gb = gb / gb.abs().sum()
                    
                    # Set the filter
                    out_channel = in_channel * filters_per_input + scale_idx * orientations_per_scale + ori_idx
                    if out_channel < self.out_channels:
                        filters[out_channel, in_channel] = gb
                        filter_idx += 1
        
        return filters
        
    def forward(self, x):
        """
        Apply the Gabor filter bank to the input.
        """
        return F.conv2d(x, self.weight, padding=self.padding)

class MultiScaleColorTextureGabor(nn.Module):
    """
    An enhanced branch that captures color and texture information across multiple scales
    using Gabor filters for texture extraction.
    - Processes the image at scales: 1.0, 0.5, 0.25.
    - Applies Gabor filters to each scaled image.
    - Concatenates the filtered features and passes through MLP to produce the auxiliary embedding.
    """
    def __init__(self, in_channels=3, out_dim=32, gabor_channels=16):
        super(MultiScaleColorTextureGabor, self).__init__()
        self.out_dim = out_dim
        self.gabor_channels = gabor_channels
        
        # Gabor filter banks for each scale
        self.gabor_filter = GaborFilter(in_channels=in_channels, out_channels=gabor_channels)
        
        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to process concatenated features
        self.mlp = nn.Sequential(
            nn.Linear(3 * gabor_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) - Original RGB image
        Returns: (B, out_dim) - Auxiliary embedding vector
        """
        B, C, H, W = x.shape
        
        # Generate multi-scale images
        x1 = x  # Scale 1.0
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)  # Scale 0.5
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False) # Scale 0.25
        
        # Apply Gabor filters to each scale
        x1_gabor = self.gabor_filter(x1)  # (B, gabor_channels, H, W)
        x2_gabor = self.gabor_filter(x2)  # (B, gabor_channels, H/2, W/2)
        x3_gabor = self.gabor_filter(x3)  # (B, gabor_channels, H/4, W/4)
        
        # Global average pooling for each scale
        x1_pool = self.global_pool(x1_gabor).view(B, -1)  # (B, gabor_channels)
        x2_pool = self.global_pool(x2_gabor).view(B, -1)  # (B, gabor_channels)
        x3_pool = self.global_pool(x3_gabor).view(B, -1)  # (B, gabor_channels)
        
        # Concatenate pooled features
        multi_scale_vec = torch.cat([x1_pool, x2_pool, x3_pool], dim=1)  # (B, 3*gabor_channels)
        
        # Apply MLP
        out = self.mlp(multi_scale_vec)  # (B, out_dim)
        return out