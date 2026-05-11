import torch
from torch import nn

class conv_block(nn.Module):
  """Convolutional block with Conv1d, BatchNorm and ReLU."""
  def __init__(self, in_channels, out_channels, **kwargs):
    super(conv_block, self).__init__()
    
    # Convert to Python int to avoid numpy type issues
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    
    self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
    self.bathcnorm = nn.BatchNorm1d(out_channels)
    self.relu = nn.ReLU()


  def forward(self, x):
    x = self.conv(x)
    x = self.bathcnorm(x)
    x = self.relu(x)
    
    return x


class FCN(nn.Module):
    """
    Fully Convolutional Network for time series classifcation.
    
    Supports configurable depth (number of conv layers) and filter sizes.
    """
    def __init__(self, input_shape, nb_classes, filters=[128, 256, 128], 
                 kernel_sizes=[8, 5, 3]):
      """
      Args:
          input_shape: Tuple (sequence_length, n_features)
          nb_classes: Number of output classes
          filters: List of filter sizes for each conv block
                  Length determines depth (number of layers)
                  Examples:
                  - [128, 256, 128] -> 3 layers (standard)
                  - [128, 256, 128, 64] -> 4 layers (deeper)
                  - [128, 128] -> 2 layers (shllower)
          kernel_sizes: List of kernel sizes for each conv block
                      If shorter than filters, last value is repeated
                      Default: [8, 5, 3]
      
      """
      super(FCN, self).__init__()
      # Convert all parameters to Python types
      if isinstance(input_shape, (list, tuple)):
        self.input_shape = tuple(int(x) for x in input_shape)
      else:
        self.input_shape = (int(input_shape), 1)
        
      self.nb_classes = int(nb_classes)
      
      # Convert filter sizes to Python ints
      self.filters = [int(f) for f in filters]
      self.depth = len(self.filters)
      
      # Handle kernel sizes
      if isinstance(kernel_sizes, (list, tuple)):
        kernel_sizes = [int(k) for k in kernel_sizes]
        # Extende kernel_sizes if shorter than filters
        while len(kernel_sizes) < self.depth:
          kernel_sizes.append(kernel_sizes[-1])            
        self.kernel_sizes = kernel_sizes[:self.depth]
      else:
        # Single value, use it for all layers
        self.kernel_sizes = [int(kernel_sizes)] * self.depth  
        
      
      # Get input channels 
      in_channels = int(self.input_shape[1]) if len(self.input_shape) > 1 else 1
      
      # Build convolutional blocks dynamically
      self.conv_blocks = nn.ModuleList()
      
      current_channels = in_channels
      for i in range(self.depth):
        self.conv_blocks.append(
          conv_block(
            in_channels=current_channels,
            out_channels=self.filters[i],
            kernel_size=self.kernel_sizes[i],
            stride=1,
            padding='same'
          )
        )          
        current_channels = self.filters[i]
        
                
      # Global Average Pooling
      self.avgpool = nn.AdaptiveAvgPool1d(1)
      
      # Output layer - returns logits (no softmax)
      self.fc = nn.Linear(self.filters[-1], self.nb_classes)

    def forward(self, x):
      """
      Forward pass - returns logits (no softmax).
      
      Args:
          x: Input tensor of shape (batch, n_features, seq_length)
          
      Returns:
          logits: Raw output logits (no softmax applied)
      """
      # Pass through all conv blocks
      for conv_block in self.conv_blocks:
        x = conv_block(x)
        
      x = self.avgpool(x)
      x = x.reshape(x.shape[0], -1)
      
      # Return logits (no softmax for use with CrossEntropyLoss)
      logits = self.fc(x)
      
      return logits