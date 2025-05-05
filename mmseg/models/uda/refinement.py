import torch 
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

class Guidance(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.1,
                 proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_sam = nn.Linear(dim, dim, bias=qkv_bias) 
        self.q_pl_source = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv_sam = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_pl_source = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop_sam = nn.Dropout(attn_drop) 
        self.attn_pl_source = nn.Dropout(attn_drop)

        self.proj_sam = nn.Linear(dim, dim) 
        self.proj_pl_source = nn.Linear(dim, dim)

        self.proj_drop_sam = nn.Dropout(proj_drop)
        self.proj_pl_source = nn.Dropout(proj_drop) 


    def forward(self, sam,pl_source):
        B, N, C = sam.shape
        q_sam = self.q_sam(sam).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3).contiguous()

        q_pl_source = self.q_pl_source(pl_source).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3).contiguous()
        

        kv_sam = self.kv_sam(sam).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k_sam, v_sam = kv_sam[0], kv_sam[1]

        kv_pl_source = self.kv_pl_source(pl_source).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k_pl_source , v_pl_source  = kv_pl_source[0], kv_pl_source[1]

        attn_sam = (q_pl_source @ k_sam.transpose(-2, -1).contiguous()) * self.scale
        attn_sam = attn_sam.softmax(dim=-1)
        attn_sam = self.attn_drop_sam(attn_sam)

        sam_attention = (attn_sam @ v_sam).transpose(1, 2).contiguous().reshape(B, N, C)
        sam_attention = self.proj_sam(sam_attention)
        guidance_sam = self.proj_drop_sam(sam_attention + sam)

        attn_pl_source = (q_sam @ k_pl_source.transpose(-2, -1).contiguous()) * self.scale
        attn_pl_source = attn_pl_source.softmax(dim=-1)
        attn_pl_source = self.attn_pl_source(attn_pl_source)

        attn_pl_source = (attn_pl_source @ v_pl_source).transpose(1, 2).contiguous().reshape(B, N, C)
        attn_pl_source = self.proj_sam(attn_pl_source)
        guidance_pl_source = self.proj_drop_sam(attn_pl_source + pl_source)

        return guidance_sam,guidance_pl_source

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=1024,
                 patch_size=128,
                 stride=64,
                 in_chans=1,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self,device,num_blocks=6,num_seq=289,embed_dim=768):
        super().__init__()
        self.pos_embed = nn.Embedding(num_seq,embed_dim)
        self.register_buffer("positions",torch.arange(num_seq,device=device))

        self.sam_embed = OverlapPatchEmbed()
        self.pl_embed = OverlapPatchEmbed()

        self.encode = nn.ModuleList([Guidance(embed_dim) for _ in range(num_blocks)])

    def forward(self, sam, pl_source):
        positions = self.pos_embed(self.positions)
        sam_embed = self.sam_embed(sam) + positions
        pl_embed = self.pl_embed(pl_source) + positions

        for layer in self.encode:
            sam,pl_source = layer(sam_embed,pl_embed)
        return sam + pl_source

class Decoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=19):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 17 -> 34

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 34 -> 68

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 68 -> 136

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # 136 -> 1088

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: [B, 289, 768] → [B, 768, 17, 17]
        B, N, C = x.shape
        assert N == 289, "Expected 289 tokens (17x17 grid)"
        x = x.transpose(1, 2).contiguous().view(B, C, 17, 17)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        return x

class EncodeDecode(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.encode = Encoder(device)
        self.decode = Decoder()
    def forward(self, sam, pl_source):
        x = self.encode(sam,pl_source)
        return self.decode(x)
