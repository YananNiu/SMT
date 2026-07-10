from typing import Callable, Optional, Tuple


from torch import nn as nn
import torch.nn.functional as F
from timm.layers.format import Format, nchw_to
from timm.layers.trace_utils import _assert

class ColPatchEmbed(nn.Module):
    """ 
    taken from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
    with changes to allow for column/row patches
    """

    def __init__(
            self,
            img_size: int = (224,224), # (H, W)
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            patch_mode: str = 'row',  # 'row' for row patches, 'column' for column patches
    ):
        super().__init__()
        self.patch_mode = patch_mode
        if patch_mode not in ['row', 'column']:
            raise ValueError("patch_mode must be 'row' or 'column'")
            
        self.patch_h = img_size[0] if patch_mode == 'column' else 1 
        self.patch_w = img_size[1] if patch_mode == 'row' else 1
        self.patch_size = tuple([self.patch_h, self.patch_w])

        self.img_size = img_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        # Adjust the kernel size and stride based on patch_mode
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape

        if self.strict_img_size:
            _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
        elif not self.dynamic_img_pad:
            _assert(
                H % self.patch_size[0] == 0,
                f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            )
            _assert(
                W % self.patch_size[1] == 0,
                f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
            )
            
        if self.dynamic_img_pad: 
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h)) # pad the image minimally so H and W are divisible by the patch size
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NC H*W(=L) -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
    
