import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        
    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = ks * ks  # Number of sampling points
        
        if self.padding:
            x = self.zero_padding(x)
            
        # Get sampling locations
        p = self._get_p(offset, dtype)
        q = self._get_q(p, x.size(2), x.size(3))
        
        # Sample features
        x_offset = self._bilinear_interpolate(x, q, ks)
        x_offset = self._reshape_x_offset(x_offset, ks)
        
        # Apply convolution
        out = self.conv(x_offset)
        return out

    def _get_p(self, offset, dtype):
        b, _, h, w = offset.size()
        ks = self.kernel_size
        N = ks * ks
        
        # Generate base grid
        p_base = torch.meshgrid(
            torch.arange(0, ks), torch.arange(0, ks), indexing='ij'
        )
        p_base = torch.stack(p_base, dim=-1).view(-1, 2)  # [ks*ks, 2]
        p_base = p_base.unsqueeze(0).unsqueeze(0)  # [1, 1, ks*ks, 2]
        p_base = p_base.repeat(b, h, w, 1, 1)  # [b, h, w, ks*ks, 2]
        
        # Add offsets
        offset = offset.permute(0, 2, 3, 1)  # [b, h, w, 2*ks*ks]
        offset = offset.view(b, h, w, -1, 2)  # [b, h, w, ks*ks, 2]
        
        p = p_base + offset
        p = p.permute(0, 3, 1, 2, 4)  # [b, ks*ks, h, w, 2]
        p = p.reshape(b, -1, h, w)  # [b, 2*ks*ks, h, w]
        
        return p

    def _get_q(self, p, h, w):
        # Split into x and y coordinates
        p_x = p[:, :p.size(1)//2, :, :]
        p_y = p[:, p.size(1)//2:, :, :]
        
        # Clamp to image boundaries
        p_x = torch.clamp(p_x, 0, w-1)
        p_y = torch.clamp(p_y, 0, h-1)
        
        return torch.cat([p_x, p_y], dim=1)

    def _bilinear_interpolate(self, x, q, ks):
        b, c, h, w = x.size()
        N = ks * ks
        
        # Split coordinates
        q_x = q[:, :N, :, :]
        q_y = q[:, N:, :, :]
        
        # Get integer and fractional parts
        q_x0 = torch.floor(q_x).long()
        q_x1 = q_x0 + 1
        q_y0 = torch.floor(q_y).long()
        q_y1 = q_y0 + 1
        
        # Clamp to image boundaries
        q_x0 = torch.clamp(q_x0, 0, w-1)
        q_x1 = torch.clamp(q_x1, 0, w-1)
        q_y0 = torch.clamp(q_y0, 0, h-1)
        q_y1 = torch.clamp(q_y1, 0, h-1)
        
        # Gather pixel values
        x_flat = x.view(b, c, -1)
        
        # Calculate indices
        idx_00 = q_y0 * w + q_x0
        idx_01 = q_y0 * w + q_x1
        idx_10 = q_y1 * w + q_x0
        idx_11 = q_y1 * w + q_x1
        
        # Sample values
        val_00 = torch.gather(x_flat, 2, idx_00.view(b, 1, -1).repeat(1, c, 1))
        val_01 = torch.gather(x_flat, 2, idx_01.view(b, 1, -1).repeat(1, c, 1))
        val_10 = torch.gather(x_flat, 2, idx_10.view(b, 1, -1).repeat(1, c, 1))
        val_11 = torch.gather(x_flat, 2, idx_11.view(b, 1, -1).repeat(1, c, 1))
        
        # Calculate weights
        w_x = (q_x - q_x0.float()).unsqueeze(1)
        w_y = (q_y - q_y0.float()).unsqueeze(1)
        
        # Bilinear interpolation
        val_0 = val_00 * (1 - w_x) + val_01 * w_x
        val_1 = val_10 * (1 - w_x) + val_11 * w_x
        val = val_0 * (1 - w_y) + val_1 * w_y
        
        return val.view(b, c, N, h, w)

    def _reshape_x_offset(self, x_offset, ks):
        b, c, N, h, w = x_offset.size()
        x_offset = x_offset.permute(0, 1, 3, 4, 2).contiguous()
        x_offset = x_offset.view(b, c * ks * ks, h, w)
        return x_offset