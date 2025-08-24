import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss

class MultiComponentLoss(nn.Module):
    def __init__(self, perceptual_model=None, w_ssim=0.3, w_edge=0.05, w_align=0.05, w_perc=0.01):
        super(MultiComponentLoss, self).__init__()
        self.recon_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11, reduction='mean')
        self.perceptual_model = perceptual_model
        # Weights for easy tuning
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_align = w_align
        self.w_perc = w_perc

    def gradient_loss(self, pred, gt):
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_grad_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        gt_grad_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        return F.l1_loss(pred_grad_x, gt_grad_x) + F.l1_loss(pred_grad_y, gt_grad_y)

    def forward(self, sr_output, gt, alignment):
        # Ensure alignment is a single scalar value
        if isinstance(alignment, torch.Tensor):
            alignment = alignment.mean()  # Take mean if it's a tensor
            
        # Reconstruction & edge & SSIM
        loss_recon = self.recon_loss(sr_output, gt)
        loss_edge = self.gradient_loss(sr_output, gt)
        loss_ssim = self.ssim_loss(sr_output, gt)

        # LPIPS perceptual loss
        loss_perc = 0
        if self.perceptual_model:
            # LPIPS expects 3-channel, [-1,1]
            sr_lpips = (sr_output * 2 - 1).clamp(-1, 1)
            gt_lpips = (gt * 2 - 1).clamp(-1, 1)
            if sr_lpips.shape[1] == 1:  # repeat channels for grayscale
                sr_lpips = sr_lpips.repeat(1, 3, 1, 1)
                gt_lpips = gt_lpips.repeat(1, 3, 1, 1)
            loss_perc = self.perceptual_model(sr_lpips, gt_lpips)
            if isinstance(loss_perc, torch.Tensor):
                loss_perc = loss_perc.mean()

        # Combine all
        total_loss = (
            loss_recon
            + self.w_ssim * loss_ssim
            + self.w_edge * loss_edge
            + self.w_align * alignment
            + self.w_perc * loss_perc
        )

        return total_loss, {
            "recon": loss_recon.item(),
            "ssim": loss_ssim.item(),
            "edge": loss_edge.item(),
            "align": alignment.item() if isinstance(alignment, torch.Tensor) else alignment,
            "perc": loss_perc.item() if self.perceptual_model and isinstance(loss_perc, torch.Tensor) else loss_perc
        }