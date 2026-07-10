# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# source: https://github.com/facebookresearch/deit/blob/main/models_v2.py

import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from smt.models.patch_special import ColPatchEmbed

class Attention_with_gradients(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        
    def save_attn_gradients(self, grad):
        self.attn_gradients = grad
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        attn.requires_grad_(True)
        attn.register_hook(self.save_attn_gradients)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        y, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma_1 * y)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn_weights
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x #shape (B, num_patches, embed_dim)
    


# only for image data
class vit_model_img(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=(224,224),  patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,init_scale=1e-4,attention_visualization=False,**kwargs):
        super().__init__()
        self.attention_visualization = attention_visualization
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_mode = kwargs.get("patch_mode")
        assert self.patch_mode in ["square","column","row"]
        if self.patch_mode == "square":
            Patch_layer = PatchEmbed
            self.patch_embed = Patch_layer(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            Patch_layer = ColPatchEmbed
            self.patch_embed = Patch_layer(img_size=img_size,in_chans=in_chans, embed_dim=embed_dim,patch_mode=self.patch_mode)
        num_patches = self.patch_embed.num_patches
        
        # cls_token: a special learnable embedding for the whole image, prepended to the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # img_pos_embed: positional embedding added to each patch to encode spatial layout.
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    
        # blocks: a stack of transformer encoder layers that process the tokens and extract features.
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        # norm: normalisation applied before reading out the final features.
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        
        trunc_normal_(self.img_pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'img_pos_embed', 'img_cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    
    
    def forward_features(self, x):
        B = x.shape[0]        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embed(x)
        x = x + self.img_pos_embed 
        x = torch.cat((cls_tokens, x), dim=1)
        
        attention_weights = [] 
        for blk in self.blocks:
            x,attn = blk(x)
            attention_weights.append(attn)
        x = self.norm(x)
        if self.attention_visualization:
            return x[:, 0],attention_weights
        else:
            return x[:, 0]
    
    def forward(self, x):
        if self.attention_visualization:
            x,attention_wights = self.forward_features(x)
        else:
            x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        if self.attention_visualization:
            return x,attention_wights
        else:
            return x



class vit_model_ts(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, ts_shape = (1,144), num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers = Block,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,init_scale=1e-4,attention_visualization=False,**kwargs):
        super().__init__()
        self.attention_visualization = attention_visualization
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # cls_token: a special learnable embedding for the whole image, prepended to the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ts_pos_embed = nn.Parameter(torch.zeros(1, ts_shape[0], embed_dim))
        
        # blocks: a stack of transformer encoder layers that process the tokens and extract features.
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        # norm: normalisation applied before reading out the final features.
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.ts_embed = nn.Sequential(
            nn.Linear(ts_shape[1], self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        
        trunc_normal_(self.ts_pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'img_pos_embed', 'img_cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, ts):
        B = ts.shape[0] 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        ts = self.ts_embed(ts)
        # if ts.ndim == 2:
        #     ts = ts.unsqueeze(1)
        ts = ts + self.ts_pos_embed
        ts = torch.cat((cls_tokens, ts), dim=1)
        
        attention_weights = [] 
        for blk in self.blocks:
            ts,attn = blk(ts)
            attention_weights.append(attn)

        ts = self.norm(ts)
        if self.attention_visualization:
            return ts[:, 0],attention_weights
        else:
            return ts[:, 0]
    
    def forward(self, ts):
        if self.attention_visualization:
            x,attention_wights = self.forward_features(ts)
        else:
            x = self.forward_features(ts)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        if self.attention_visualization:
            return x,attention_wights
        else:
            return x
    
    
    
class vit_model_img_ts(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=(224,224),  patch_size=16, ts_shape = (1,144), in_chans=3, num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers = Block,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,init_scale=1e-4,attention_visualization=False,**kwargs):
        super().__init__()
        self.attention_visualization = attention_visualization
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_mode = kwargs.get("patch_mode")
        assert self.patch_mode in ["square","column","row"]
        if self.patch_mode == "square":
            Patch_layer = PatchEmbed
            self.patch_embed = Patch_layer(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            Patch_layer = ColPatchEmbed
            self.patch_embed = Patch_layer(img_size=img_size,in_chans=in_chans, embed_dim=embed_dim,patch_mode=self.patch_mode)
        
        num_patches = self.patch_embed.num_patches

        # cls_token: a special learnable embedding for the whole image, prepended to the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # img_pos_embed: positional embedding added to each patch to encode spatial layout.
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.ts_pos_embed = nn.Parameter(torch.zeros(1, ts_shape[0], embed_dim))
        
        # modality token-type embeddings (not learned).
        # a (2, embed_dim) embedding matrix distinguishing image vs. time-series tokens
        self.token_type_embeddings = nn.Embedding(2, self.embed_dim) #initialization
        
        # blocks: a stack of transformer encoder layers that process the tokens and extract features.
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        # norm: normalisation applied before reading out the final features.
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.ts_embed = nn.Sequential(
            nn.Linear(ts_shape[1], self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        
        trunc_normal_(self.img_pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.ts_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'img_pos_embed', 'img_cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    
    
    def forward_features(self, x, ts):
        assert x.shape[0] == ts.shape[0]
        B = x.shape[0] 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = self.patch_embed(x)
        x = x + self.img_pos_embed + self.token_type_embeddings(torch.ones(B, x.shape[1], dtype=torch.long,device=x.device))# token_type_embeddings index 1 marks image tokens
        x = torch.cat((cls_tokens, x), dim=1)

        ts = self.ts_embed(ts)
        # if ts.ndim == 2:
        #     ts = ts.unsqueeze(1)
        ts = ts + self.ts_pos_embed + self.token_type_embeddings(torch.zeros(B, ts.shape[1], dtype=torch.long,device=x.device))# token_type_embeddings index 0 marks time-series tokens

        # modality fusion
        x_ts = torch.cat((x,ts),dim=1)
        
        attention_weights = [] 
        for blk in self.blocks:
            x_ts,attn = blk(x_ts)
            attention_weights.append(attn)
        x_ts = self.norm(x_ts)
        
        if self.attention_visualization:
            return x_ts[:, 0],attention_weights
        else:
            return x_ts[:, 0]
    
    def forward(self, x,ts):
        if self.attention_visualization:
            x,attention_wights = self.forward_features(x,ts)
        else:
            x = self.forward_features(x,ts)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        if self.attention_visualization:
            return x,attention_wights
        else:
            return x



class vit_model_2img_ts(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, ts_shape = (1,144), in_chans=3, num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,init_scale=1e-4,attention_visualization=False,**kwargs):
        super().__init__()
        self.attention_visualization = attention_visualization
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        #for square patch
        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #for column/row patch
        #self.patch_embed = Patch_layer(img_size=(img_size,img_size), in_chans=in_chans, embed_dim=embed_dim,patch_mode='row')
        num_patches = self.patch_embed.num_patches

        # cls_token: a special learnable embedding for the whole image, prepended to the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # img_pos_embed: positional embedding added to each patch to encode spatial layout.
        self.img_pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.img_pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.ts_pos_embed = nn.Parameter(torch.zeros(1, ts_shape[0], embed_dim))
        
        # modality token-type embeddings (not learned).
        self.token_type_embeddings = nn.Embedding(3, self.embed_dim) #initialization
        
        # blocks: a stack of transformer encoder layers that process the tokens and extract features.
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        # norm: normalisation applied before reading out the final features.
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.ts_embed = nn.Sequential(
            nn.Linear(ts_shape[1], self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        
        trunc_normal_(self.img_pos_embed1, std=.02)
        trunc_normal_(self.img_pos_embed2, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.ts_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'img_pos_embed', 'img_cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    
    
    def forward_features(self, x1,x2, ts):
        assert x1.shape[0] == x2.shape[0] == ts.shape[0]
        B = x1.shape[0] 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x1 = self.patch_embed(x1) #shape (B, num_patches, embed_dim)
        x2 = self.patch_embed(x2)
        x1 = x1 + self.img_pos_embed1 + self.token_type_embeddings(torch.ones(B, x1.shape[1], dtype=torch.long,device=x1.device)) #shape (B, num_patches, embed_dim)
        x2 = x2 + self.img_pos_embed2 + self.token_type_embeddings(2*torch.ones(B, x2.shape[1], dtype=torch.long,device=x2.device))
        x = torch.cat((cls_tokens, x1, x2), dim=1)

        ts = self.ts_embed(ts)
        # if ts.ndim == 2:
        #     ts = ts.unsqueeze(1)
        ts = ts + self.ts_pos_embed + self.token_type_embeddings(torch.zeros(B, ts.shape[1], dtype=torch.long,device=x.device))

        # modality fusion
        x_ts = torch.cat((x,ts),dim=1)
        
        attention_weights = [] 
        for blk in self.blocks:
            x_ts,attn = blk(x_ts)
            attention_weights.append(attn)
        x_ts = self.norm(x_ts)
        
        if self.attention_visualization:
            return x_ts[:, 0],attention_weights
        else:
            return x_ts[:, 0]
    
    def forward(self,x1,x2,ts):
        if self.attention_visualization:
            x,attention_wights = self.forward_features(x1,x2,ts)
        else:
            x = self.forward_features(x1,x2,ts)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        if self.attention_visualization:
            return x,attention_wights
        else:
            return x    
    
    
    
class vit_model_imgs(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, ts_shape = (1,144), in_chans=3, num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 img_num = 3,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,init_scale=1e-4,attention_visualization=False,**kwargs):
        super().__init__()
        self.attention_visualization = attention_visualization
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        #for square patch
        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #for column/row patch
        #self.patch_embed = Patch_layer(img_size=(img_size,img_size), in_chans=in_chans, embed_dim=embed_dim,patch_mode='row')
        num_patches = self.patch_embed.num_patches

        # cls_token: a special learnable embedding for the whole image, prepended to the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.img_num = img_num  # number of images
        # img_pos_embed: positional embedding added to each patch to encode spatial layout.
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.img_num, num_patches, embed_dim))
        
        # modality token-type embeddings (not learned).
        self.token_type_embeddings = nn.Embedding(self.img_num, self.embed_dim) #initialization
        
        # blocks: a stack of transformer encoder layers that process the tokens and extract features.
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        # norm: normalisation applied before reading out the final features.
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.img_pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'img_pos_embed', 'img_cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    
    
    def forward_features(self, xs):
        assert xs.ndim == 5 # [B, img_num, C, H, W]
        B,num_imgs = xs.shape[0],xs.shape[1]
        cls_tokens = self.cls_token.expand(B,-1,-1) # [B, 1, embed_dim]
        xs_reshape = xs.reshape(B*num_imgs,xs.shape[2],xs.shape[3],xs.shape[4])
        xs_reshape = self.patch_embed(xs_reshape) # [B*num_imgs, num_patches, embed_dim]
        xs = xs_reshape.reshape(B,num_imgs,xs_reshape.shape[1],xs_reshape.shape[2]) # [B, num_imgs, num_patches, embed_dim]
        
        token_type_ids = torch.arange(num_imgs).repeat(B,1).to(device=xs.device) #token_type_ids :[batch_size, img_num]
        token_type_embeds = self.token_type_embeddings(token_type_ids)  # [batch_size, img_num, embed_dim]
        token_type_embeds = token_type_embeds[:, :, None, :].expand(-1, -1, xs_reshape.shape[1], -1) # [batch_size, img_num, num_patches, embed_dim]

        xs = xs + self.img_pos_embed + token_type_embeds # [B, num_imgs, num_patches, embed_dim]
        xs = xs.reshape(B,-1,xs.shape[3]) # [B, num_imgs*num_patches, embed_dim]

        xs = torch.cat((cls_tokens, xs), dim=1) # [B, num_imgs*num_patches+1, embed_dim]
        
        attention_weights = [] 
        for blk in self.blocks:
            xs,attn = blk(xs)
            attention_weights.append(attn)
        xs = self.norm(xs)
        
        if self.attention_visualization:
            return xs[:, 0],attention_weights
        else:
            return xs[:, 0]
    
    def forward(self,xs):
        if self.attention_visualization:
            x,attention_wights = self.forward_features(xs)
        else:
            x = self.forward_features(xs)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        if self.attention_visualization:
            return x,attention_wights
        else:
            return x    