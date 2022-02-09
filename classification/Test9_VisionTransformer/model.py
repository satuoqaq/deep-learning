import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size  # 224*224
        self.patch_size = patch_size  # 16*16
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])  # 14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 196

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size({H}*{W}) doesn't match model ({self.image_size[0]},{self.image_size[1]})."

        # flatten:[B,C,H,W] -> [B,C,HW]
        # transpose[B,C,HW] ->[B,HW,C]
        x = self.proj(x)
        x = x.flatten(2, -1)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入的token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 根号dim长度
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 其实这个对应[Qw,Qk,Qv]
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches+1,total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads ,num_patches+1 , embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose:-> [batch_size,num_heads,embed_dim_per_head,num_patches+1]
        # @: multiply -> [batch_size, num_heads, num_patches+1, num_patches + 1]
        # 后两维换了得到注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size,num_heads, num_patches+1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches +1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches+1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
