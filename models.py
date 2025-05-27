import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import math
class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=256,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
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

        token = math.floor((img_size[0] + 2 * (patch_size[0] // 2) - patch_size[0]) / stride) + 1
        self.num_tokens= token ** 2

    def forward(self, x):
        x = self.proj(x)
        _, _, self.H, self.W = x.shape

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 qkv_bias=False,
                 drop_rate=0.1,
                 n_heads=8
                 ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim should be devided by n_heads"
        self.head_embed_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.W_query = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.W_key = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.W_value = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)

        self.sam_att_layer_norm = nn.LayerNorm(embed_dim)
        self.pl_att_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim,2*embed_dim,bias=qkv_bias),
            nn.GELU(),
            nn.Linear(2*embed_dim,embed_dim,bias=qkv_bias)
        )
        self.ff_dropout =  nn.Dropout(drop_rate)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)


    def compute_att(self,query,key,value):
        batch,num_tokens,embed_dim = query.shape
        
        query = query.view(batch,num_tokens,self.n_heads,self.head_embed_dim).transpose(1,2).contiguous() # (batch,n_heads,num_tokens,embed_dim)
        key = key.view(batch,num_tokens,self.n_heads,self.head_embed_dim).transpose(1,2).contiguous() # (batch,n_heads,num_tokens,embed_dim)
        value = value.view(batch,num_tokens,self.n_heads,self.head_embed_dim).transpose(1,2).contiguous() # (batch,n_heads,num_tokens,embed_dim)


        attention_scores:torch.Tensor = (query @ key.transpose(-2,-1)) / self.embed_dim**0.5
        attention_weights = attention_scores.contiguous().softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention = (attention_weights @ value).transpose(1,2).contiguous().view(batch,num_tokens,embed_dim)
        return attention.contiguous()
        
    
    def forward(self,
                pl_source:torch.Tensor,
                sam_source:torch.Tensor
                ):
        sam_source = self.sam_att_layer_norm(sam_source)
        pl_source = self.pl_att_layer_norm(pl_source)
        
        query:torch.Tensor = self.W_query(pl_source)
        key:torch.Tensor = self.W_key(sam_source)
        value:torch.Tensor = self.W_value(sam_source)

        attention = self.compute_att(query,key,value)
        
        attention = self.ff_layer_norm(self.ff_dropout(attention+pl_source))

        out = self.ff(attention)

        return out

class AttentionBlock(nn.Module):
    def __init__(self,embed_dim=768,n_blocks=5,mode="self"):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttention(embed_dim=embed_dim) for _ in range(n_blocks)
        ])
        self.mode = mode
    def forward(self,pl,sam):
        for layer in self.layers:
            pl = layer(pl,pl) if self.mode == "self" else layer(pl,sam)
        return pl

class Encoder(nn.Module):
    def __init__(self,img_size=256,
                 patch_size=32,
                 stride=32,
                 in_chans=1,
                 embed_dim=768):
        super().__init__()

        self.sam_embed = OverlapPatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        self.pl_embed = OverlapPatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=in_chans,embed_dim=embed_dim)
        self.n_tokens = self.sam_embed.num_tokens

        self.pos_embed = nn.Embedding(self.n_tokens, embed_dim)

        self.attention = AttentionBlock()

    def forward(self,pl,sam):
        device = sam.device
        pl = self.pl_embed(pl)
        sam = self.sam_embed(sam)

        positions = torch.arange(self.n_tokens,device=device)

        pos_embed = self.pos_embed(positions)

        sam = sam + pos_embed
        pl = pl + pos_embed

        out = self.attention(pl,sam)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: [B, 289, 768] → [B, 768, 17, 17]
        B, N, C = x.shape
        H,W = int(math.sqrt(N)),int(math.sqrt(N))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

class Refine(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self,pl,sam):
        bottleneck = self.encode(pl,sam)
        return self.decode(bottleneck)


