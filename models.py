import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import math
from einops import rearrange

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

class ConvEncoder(nn.Module):
    def __init__(self, in_chans=1, embed_dim=768):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x1 = self.stage1(x)  # [B, 64, H, W]
        x2 = self.stage2(x1) # [B, 128, H/2, W/2]
        x3 = self.stage3(x2) # [B, embed_dim, H/4, W/4]
        x4 = self.stage4(x3) # [B, embed_dim, H/8, W/8]
        return x1, x2, x3,x4 

class TransformerBridge(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_tokens):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
        self.blocks = nn.ModuleList([Guidance(embed_dim) for _ in range(num_blocks)])

    def forward(self, sam,pl):
        B, C, H, W = sam.shape
        sam = rearrange(sam, 'b c h w -> b (h w) c') + self.pos_embed[:, :H*W]
        pl = rearrange(pl, 'b c h w -> b (h w) c') + self.pos_embed[:, :H*W]
        for blk in self.blocks:
            sam, pl = blk(sam, pl)
        sam = rearrange(sam, 'b (h w) c -> b c h w', h=H)
        pl = rearrange(pl, 'b (h w) c -> b c h w', h=H)
        return sam,pl

class TransUNetDecoder(nn.Module):
    def __init__(self, embed_dim, n_cls):
        super().__init__()
        self.up0 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(256+256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(64 + 64, n_cls, kernel_size=1)

    def forward(self,x4, x3, x2, x1):
        x = self.up0(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_final(x)
        return x

class TransUNetLike(nn.Module):
    def __init__(self, n_cls=2, img_size=256, embed_dim=768, num_blocks=7):
        super().__init__()
        self.sam_encoder = ConvEncoder(in_chans=1, embed_dim=embed_dim)
        self.pl_encoder = ConvEncoder(in_chans=1, embed_dim=embed_dim)
        self.bridge = TransformerBridge(embed_dim=embed_dim, num_blocks=num_blocks, num_tokens=(img_size//8)**2)
        self.decoder = TransUNetDecoder(embed_dim=embed_dim, n_cls=n_cls)

    def forward(self, sam,pl):

        sam1, sam2, sam3,sam4 = self.sam_encoder(sam)
        pl1, pl2, pl3,pl4 = self.pl_encoder(pl)
        sam4,pl4 = self.bridge(sam4,pl4 )
        return self.decoder(sam4 + pl4, sam3 + pl3 , sam2 + pl2,sam1 + pl1)
