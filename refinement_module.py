from mmseg.models.segmentors.base import DoubleConvPerso,Up,Down,OutConv
import torch.nn as nn
import torch

# taken from mmseg code
def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def apply_random_mask(x, mask_ratio=0.3, patch_size=16):
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w
    num_keep = int(total_patches * (1 - mask_ratio))

    # Randomly select patch indices to keep
    patch_indices = torch.randperm(total_patches, device=x.device)
    keep_indices = patch_indices[:num_keep]

    # Create mask grid
    mask = torch.ones(total_patches, device=x.device)
    mask[keep_indices] = 0
    mask = mask.view(num_patches_h, num_patches_w)

    # Upsample mask to image size
    mask = mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    return x * (1 - mask)  # masked input, binary mask


class UNet(nn.Module):
    def __init__(self,dataset_type, in_channel=2, n_classes=1, norm_layer="bn", bilinear=False):
        super(UNet, self).__init__()
        self.dataset_type = dataset_type
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm_layer = norm_layer

        self.inc = DoubleConvPerso(in_channel, 64, norm_layer)
        self.down1 = Down(64, 128, norm_layer)
        self.down2 = Down(128, 256, norm_layer)
        self.down3 = Down(256, 512, norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_layer)
        self.up1 = Up(1024, 512 // factor, norm_layer, bilinear)
        self.up2 = Up(512, 256 // factor, norm_layer, bilinear)
        self.up3 = Up(256, 128 // factor, norm_layer, bilinear)
        self.up4 = Up(128, 64, norm_layer, bilinear)
        self.outc = OutConv(64, n_classes)

        # Initialize weights safely
        self.apply(self.init_weights)

    def init_weights(self, m):
        #Applies Kaiming normal initialization to all Linear layers.
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, sam,pl, apply_mask=False, mask_ratio=0.3):
        if apply_mask:
            sam = apply_random_mask(sam, mask_ratio=mask_ratio)
            pl = apply_random_mask(pl, mask_ratio=mask_ratio)
        x = torch.cat((sam, pl), dim=1) if self.dataset_type == "cityscapes" else sam
        x1 = self.inc(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return logits