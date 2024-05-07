import math
import copy
import os
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import AutoencoderKL
from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from model.attend import Attend
from model.vit import ViT
from model.vit import Vit_recover


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        # nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1),
        DoubleConv(dim, default(dim_out, dim),dim//2),
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        # nn.Conv2d(dim * 4, default(dim_out, dim), 1),
        DoubleConv(dim * 4, default(dim_out, dim)),
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias = True, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        # q: (b d H W)
        # k: (b d h w)
        # v: (b d h w)

        q: (b (H W) d)
        k: (b (H W) d)
        v: (b (H W) d)
        """
        # _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        # q = rearrange(q, 'b d H W -> b (H W) d')
        # k = rearrange(k, 'b d h w -> b (h w) d')
        # v = rearrange(v, 'b d h w -> b (h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
        # dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        # z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z
'''
class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)
    
'''
class Epipolar_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, do_epipolar,do_bidirectional_epipolar, qkv_bias = True, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.do_epipolar = do_epipolar
        self.do_bidirectional_epipolar = do_bidirectional_epipolar

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
    
    def get_epipolar(self,b,h,w,k,src_c2w,target_c2w):
        H = h
        W = H*16/9  #* 原始圖像為 16:9

        k = k.to(dtype=torch.float32)
        src_c2w=src_c2w.to(dtype=torch.float32)
        target_c2w=target_c2w.to(dtype=torch.float32)

        #* unormalize intrinsic 

        k[:,0] = k[:,0]*W
        k[:,1] = k[:,1]*H

        k[:,0,2] = h/2
        k[:,1,2] = h/2

        device = k.device

        #* 創建 h*w 的 uv map
        x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)

        x_coords = x_coords.to(device)
        y_coords = y_coords.to(device)
        coords_tensor = coords_tensor.to(device)

        k_3x3 = k[:,0:3,0:3]
        src_c2w_r = src_c2w[:,0:3,0:3]
        src_c2w_t = src_c2w[:,0:3,3]
        target_c2w_r = target_c2w[:,0:3,0:3]
        target_c2w_t = target_c2w[:,0:3,3]
        target_w2c_r = torch.linalg.inv(target_c2w_r)
        target_w2c_t = -target_c2w_t

        cx = k_3x3[:,0,2].view(b, 1)
        cy = k_3x3[:,1,2].view(b, 1)
        fx = k_3x3[:,0,0].view(b, 1)
        fy = k_3x3[:,1,1].view(b, 1)
        coords_tensor[...,0] = (coords_tensor[...,0]-cx)/fx
        coords_tensor[...,1] = (coords_tensor[...,1]-cy)/fy

        #* 做 H*W 個點的運算
        coords_tensor = rearrange(coords_tensor, 'b hw p -> b p hw') 
        point_3d_world = torch.matmul(src_c2w_r,coords_tensor)              #* 相機坐標系 -> 世界座標
        point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)           #* 相機坐標系 -> 世界座標
        point_2d = torch.matmul(target_w2c_r,point_3d_world)                #* 世界座標 -> 相機座標
        point_2d = point_2d + target_w2c_t.unsqueeze(-1)                    #* 世界座標 -> 相機座標
        pi_to_j = torch.matmul(k_3x3,point_2d)                              #* 相機座標 -> 平面座標

        #* 原點的計算
        oi = torch.zeros(3).to(dtype=torch.float32)
        oi = repeat(oi, 'p -> b p', b=b)
        oi = oi.unsqueeze(-1)
        oi = oi.to(device)
        point_3d_world = torch.matmul(src_c2w_r,oi)
        point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)  
        point_2d = torch.matmul(target_w2c_r,point_3d_world)
        point_2d = point_2d + target_w2c_t.unsqueeze(-1)  
        oi_to_j = torch.matmul(k_3x3,point_2d)
        oi_to_j = rearrange(oi_to_j, 'b c p -> b p c') #* (b,3,1) -> (b,1,3)

        #* 除以深度
        pi_to_j_unnormalize = rearrange(pi_to_j, 'b p hw -> b hw p') 
        pi_to_j = pi_to_j_unnormalize / (pi_to_j_unnormalize[..., -1:] + 1e-6)   #* (b,hw,3)
        # pi_to_j = pi_to_j_unnormalize / pi_to_j_unnormalize[..., -1:]
        oi_to_j = oi_to_j / oi_to_j[..., -1:]   #* (b,1,3)

        # print(f"pi_to_j: {pi_to_j[0,9]}")
        # print(f"oi_to_j: {oi_to_j[0,0]}")

        #* 計算feature map 每個點到每個 epipolar line 的距離
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32) # (4096,3)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)
        coords_tensor = coords_tensor.to(device)

        oi_to_pi = pi_to_j - oi_to_j            #* h*w 個 epipolar line (b,hw,3)
        oi_to_coord = coords_tensor - oi_to_j   #* h*w 個點   (b,hw,3)

        ''''
            #* 這裡做擴展
            oi_to_pi    [0,0,0]     ->      oi_to_pi_repeat     [0,0,0]
                        [1,1,1]                                 [0,0,0]
                        [2,2,2]                                 [1,1,1]
                                                                [1,1,1]
                                                                .
                                                                .
                                                                .

            oi_to_coord     [0,0,0]     ->      oi_to_coord_repeat      [0,0,0]
                            [1,1,1]                                     [1,1,1]
                            [2,2,2]                                     [2,2,2]
                                                                        [0,0,0]
                                                                        .
                                                                        .
                                                                        .
        '''
        oi_to_pi_repeat = repeat(oi_to_pi, 'b i j -> b i (repeat j)',repeat = h*w)
        oi_to_pi_repeat = rearrange(oi_to_pi_repeat,"b i (repeat j) -> b (i repeat) j", repeat = h*w)
        oi_to_coord_repeat = repeat(oi_to_coord, 'b i j -> b (repeat i) j',repeat = h*w)


        area = torch.cross(oi_to_pi_repeat,oi_to_coord_repeat,dim=-1)     #* (b,hw*hw,3)
        area = torch.norm(area,dim=-1 ,p=2)
        vector_len = torch.norm(oi_to_pi_repeat, dim=-1, p=2)
        distance = area/vector_len

        distance_weight = 1 - torch.sigmoid(50*(distance-0.05))

        epipolar_map = rearrange(distance_weight,"b (hw hw2) -> b hw hw2",hw = h*w)

        #* 如果 max(1-sigmoid) < 0.5 
        #* => min(distance) > 0.05 
        #* => 每個點離epipolar line 太遠
        #* => epipolar line 不在圖中
        #* weight map 全設為 1 
        max_values, _ = torch.max(epipolar_map, dim=-1)
        mask = max_values < 0.5
        epipolar_map[mask.unsqueeze(-1).expand_as(epipolar_map)] = 1

        if (torch.any(torch.isnan(epipolar_map)) or
            torch.any(torch.isnan(distance)) or
            torch.any(torch.isnan(distance_weight)) or
            torch.any(torch.isnan(area)) or
            torch.any(torch.isnan(vector_len)) or        
            torch.any(torch.isnan(distance_weight)) or
            torch.any(torch.isnan(oi_to_pi_repeat)) or
            torch.any(torch.isnan(oi_to_coord_repeat))):
            print(f"find nan !!!")
            print(f"epipolar_map: {torch.any(torch.isnan(epipolar_map))}")
            print(f"distance_weight: {torch.any(torch.isnan(distance_weight)) }")
            print(f"distance: {torch.any(torch.isnan(distance)) }")
            print(f"vector_len: {torch.any(torch.isnan(vector_len)) }")
            print(f"area: {torch.any(torch.isnan(area)) }")
            print(f"oi_to_pi_repeat: {torch.any(torch.isnan(oi_to_pi_repeat))}")
            print(f"oi_to_coord_repeat: {torch.any(torch.isnan(oi_to_coord_repeat))}")
            print(f"pi_to_j: {torch.any(torch.isnan(pi_to_j))}")
            print(f"oi_to_j: {torch.any(torch.isnan(oi_to_j))}")
            print(f"pi_to_j_unnormalize has zero: {torch.any(torch.eq(pi_to_j_unnormalize[...,-1:],0))}")
            print(" ")
            print("break !")
            os._exit(0)



        return epipolar_map


    def forward(self, x, src_encode,intrinsic = None,c2w = None):
        b, c, h, w = x.shape

        '''
            #! 這裡要做 cross view attention
            cross_attend = cross_view_attention(x,src_encode,k,w2c)

            #! 這裡要根據 intrinsic 跟 relative extrinsic 得到 epipolar line 的weight map
            epipolar_map = get_epipolar(k,c2w)

            #! epipolar 加權
            weighted_attention = cross_attend*epipolar_map

            #! softmax
            attn = weighted_attentio.softmax(dim = -1)

            #! 乘上 value
            out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        '''

        """
        q: (b d H W)
        k: (b d h w)
        v: (b d h w)
        """
        _, _, H, W = x.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(x, 'b d H W -> b (H W) d')
        k = rearrange(src_encode, 'b d h w -> b (h w) d')
        v = rearrange(src_encode, 'b d h w -> b (h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        #* 一般的 cross attention -> 得到 attention map
        cross_attend = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)

        weight_map = torch.ones_like(cross_attend)

        if self.do_epipolar:
            #* 得到 epipolar weighted map (B,HW,HW)
            epipolar_map = self.get_epipolar(b,h,w,intrinsic.clone(),c2w[1],c2w[0])

            epipolar_map = repeat(epipolar_map,'b hw hw2 -> (b repeat) hw hw2',repeat = self.heads)

            weight_map = weight_map*epipolar_map
        
        if self.do_bidirectional_epipolar:
            #* 做反方向的epipolar
            epipolar_map = self.get_epipolar(b,h,w,intrinsic.clone(),c2w[0],c2w[1])

            epipolar_map_transpose = epipolar_map.permute(0,2,1)

            epipolar_map = repeat(epipolar_map_transpose,'b hw hw2 -> (b repeat) hw hw2',repeat = self.heads)

            weight_map = weight_map*epipolar_map

            # if self.do_blur:
            #     if h == 64:
            #         kernel_size = 7
            #     elif h == 32:
            #         kernel_size = 5
            #     elif h == 16:
            #         kernel_size = 3
            #     else:
            #         kernel_size = 1

            #     weight_map = rearrange(weight_map,'b hw (c height weight) -> (b hw) c height weight',c=1,height=h)

            #     transform1 = T.GaussianBlur(kernel_size,1.5)
            #     blurred_weightmap = transform1(weight_map)
            #     weight_map = rearrange(blurred_weightmap,'(b hw) c h w -> b hw (c h w)',hw=h*w,c=1)

        cross_attend = cross_attend * weight_map
        att = cross_attend.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        do_epipolar = False,
        do_bidirectional_epipolar = False,
        do_mae = False,
        mask_ratio = 0.5
    ):
        super().__init__()

        # self.midas_model = midas_model

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        self.do_epipolar = do_epipolar
        self.do_bidirectional_epipolar = do_bidirectional_epipolar
        self.do_mae = do_mae
        if self.do_mae:
            self.mask_ratio = mask_ratio
        
        self.block_num = len(dim_mults) #* 做多少個block 64,32,16,8 或 32,16,8

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        dims = [*map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        self.conv1d_256_128 = nn.Conv2d(256,128,kernel_size=1)
        self.conv1d_256_512 = nn.Conv2d(256,512,kernel_size=1)
        self.conv1d_512_1024 = nn.Conv2d(512,1024,kernel_size=1)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mae = nn.ModuleList([])
        num_resolutions = len(in_out)

        self.init_res1 = block_klass(dim,dim,time_emb_dim = time_dim)
        self.init_res2 = block_klass(dim,dim,time_emb_dim = time_dim)
        self.init_down = Downsample(dim,dim*2)

        self.imgsize = 32

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            #* (256,512) (512,1024)
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention
            epipolar_klass = Epipolar_Attention
            cross_klass = CrossAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                epipolar_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads,do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar),

                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                epipolar_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads,do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar),

                # Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),
                Downsample(dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                epipolar_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads,do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar),
            ]))

            if self.do_mae:
                self.mae.append(nn.ModuleList([
                    ViT(image_size = (self.imgsize//(2**(ind))),
                        patch_size = (self.imgsize//(2**(ind)))//4,
                        dim = 1024,
                        channels = dim_in,
                        depth = 2,
                        heads=layer_attn_heads,
                        dim_head=layer_attn_dim_head,
                        dropout=0.1,
                        emb_dropout=0.1,
                        num_classes=1024,
                        mlp_dim=1024,
                    ),
                    cross_klass(1024, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    cross_klass(1024, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    cross_klass(1024, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    cross_klass(1024, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    Vit_recover(
                        image_size = (self.imgsize//(2**(ind))),
                        patch_size = (self.imgsize//(2**(ind)))//4,
                        dim = 1024,
                        channels = dim_in,
                    )
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn1 = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        #* 只做三次epipolar，如果mul_dims=(1,2,4,8)，就不做 mid epipolar atten
        if self.block_num == 3:
            self.mid_epipolar_attn1  = epipolar_klass(mid_dim, dim_head = attn_dim_head[-1], heads = attn_heads[-1],do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar)

        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn2 = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        if self.block_num == 3:
            self.mid_epipolar_attn2  = epipolar_klass(mid_dim, dim_head = attn_dim_head[-1], heads = attn_heads[-1],do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar)
        
        if self.do_mae:
            self.midmae_vit = ViT(image_size = (self.imgsize//(2**(ind+1))),
                                    patch_size = (self.imgsize//(2**(ind+1)))//4,
                                    dim = 1024,
                                    channels = mid_dim,
                                    depth = 2,
                                    heads=attn_heads[-1],
                                    dim_head=attn_dim_head[-1],
                                    dropout=0.1,
                                    emb_dropout=0.1,
                                    num_classes=1024,       #* 不重要 沒有要做分類
                                    mlp_dim=1024,
                                )
            self.midmae_cross1  = cross_klass(1024, dim_head = attn_dim_head[-1], heads = attn_heads[-1])
            self.midmae_attn1 = cross_klass(1024, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
            self.midmae_cross2  = cross_klass(1024, dim_head = attn_dim_head[-1], heads = attn_heads[-1])
            self.midmae_attn2 = cross_klass(1024, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
            self.midmae_recover = Vit_recover(
                                image_size = (self.imgsize//(2**(ind+1))),
                                patch_size = (self.imgsize//(2**(ind+1)))//4,
                                dim = 1024,
                                channels = mid_dim,
                            )

        self.mid_block3 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn3 = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        if self.block_num == 3:
            self.mid_epipolar_attn3  = epipolar_klass(mid_dim, dim_head = attn_dim_head[-1], heads = attn_heads[-1],do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention
            epipolar_klass = Epipolar_Attention

            self.ups.append(nn.ModuleList([
                # Upsample(dim_out + dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out + dim_out, dim_in, 3, padding = 1),
                Upsample(dim_out + dim_out, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                epipolar_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads,do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar),

                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                epipolar_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads,do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar),

                block_klass(dim_in,dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                epipolar_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads,do_epipolar=self.do_epipolar,do_bidirectional_epipolar = self.do_bidirectional_epipolar),

            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_up = Upsample(dim*4, dim)
        self.final_res1 = block_klass(dim, dim, time_emb_dim = time_dim)
        self.final_res2 = block_klass(dim, dim, time_emb_dim = time_dim)

        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, src_l2, src_l3,src_l4, K , c2w, x_self_cond = None):

        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        
        #* 因為midas 中間層的dimension 跟diffusion 中間層不同，所以用 1d conv 擴展 (暫時)
        if self.block_num == 3:
            src_l3 = self.conv1d_256_512(src_l3)
            src_l4 = self.conv1d_256_512(src_l4)
            src_l4 = self.conv1d_512_1024(src_l4)
        elif self.block_num == 4:
            src_l2 = self.conv1d_256_128(src_l2)
            src_l4 = self.conv1d_256_512(src_l4)

        src_encode = [src_l2,src_l3,src_l4]   #! 有三個resolution 的source img embedding
                                            #! 256 的 midas: 32x32 16x16 8x8
                                            #! 512 的 midas: 64x64 32x32 16x16

        h = []

        #* 小的Unet，在64x64 的block 做簡單的initial
        if self.block_num == 3:
            x = self.init_res1(x,t)
            x = self.init_res2(x,t)
            x = self.init_down(x)
            h.append(x)

        iter = 0
        for block1, attn1, epi1, block2, attn2, epi2, downsample, attn3, epi3 in self.downs:
            
            x = block1(x, t)
            x = attn1(x) + x
            x = epi1(x,src_encode[iter],K,c2w) + x     #!根據 src img 和 camera 參數做 epipolar attention  
            # h.append(x)

            x = block2(x, t)
            x = attn2(x) + x
            x = epi2(x,src_encode[iter],K,c2w) + x
            # h.append(x)

            if self.do_mae:
                vit1, maecross1, maeattn1,maecross2,maeattn2,vitrecover = self.mae[iter]
                src_latent, target_latent = vit1(x,src_encode[iter],self.mask_ratio)
                x = maecross1(target_latent,src_latent,src_latent)
                x = maeattn1(x,x,x) + x
                x = maecross2(x,src_latent,src_latent) + x
                x = maeattn2(x,x,x) + x
                x = vitrecover(x)

            iter += 1
            x = downsample(x)
            x = attn3(x) + x
            if self.block_num == 3:
                x = epi3(x,src_encode[iter],K,c2w) + x

            h.append(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn1(x) + x
        if self.block_num == 3:
            x = self.mid_epipolar_attn1(x,src_encode[iter],K,c2w) + x

        x = self.mid_block2(x, t)
        x = self.mid_attn2(x) + x
        if self.block_num == 3:
            x = self.mid_epipolar_attn2(x,src_encode[iter],K,c2w) + x
        if self.do_mae:
            src_latent, target_latent = self.midmae_vit(x,src_encode[iter],self.mask_ratio)
            x = self.midmae_cross1(target_latent,src_latent,src_latent)
            x = self.midmae_attn1(x,x,x) + x
            x = self.midmae_cross2(x,src_latent,src_latent) + x
            x = self.midmae_attn2(x,x,x) + x
            x = self.midmae_recover(x)

        x = self.mid_block3(x, t)
        x = self.mid_attn3(x) + x
        if self.block_num == 3:
            x = self.mid_epipolar_attn3(x,src_encode[iter],K,c2w) + x


        for upsample, attn1, epi1, block1, attn2, epi2, block2, attn3, epi3 in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = upsample(x)
            iter -= 1

            x = attn1(x) + x
            x = epi1(x,src_encode[iter],K,c2w) + x
            x = block1(x)

            x = attn2(x) + x
            x = epi2(x,src_encode[iter],K,c2w) + x
            x = block2(x)

            x = attn3(x) + x
            x = epi3(x,src_encode[iter],K,c2w) + x

        #* 小的Unet，在64x64 的block 做簡單的conv
        if self.block_num == 3:
            x = torch.cat((x, h.pop()), dim = 1)
            x = self.final_up(x)
            x = self.final_res1(x, t)
            x = self.final_res2(x, t)

        x = self.final_conv(x)

        return x

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size = None,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        # midas_model = None,
        do_latent = True,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        # self.midas_model = midas_model
        self.do_latent = do_latent
        if self.do_latent:
            model_key = "stabilityai/stable-diffusion-2-1-base"
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, src_l2, src_l3,src_l4, K, c2w, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, 
                                src_l2 = src_l2, 
                                src_l3 = src_l3,
                                src_l4 = src_l4,
                                K = K, 
                                c2w = c2w,
                                x_self_cond = x_self_cond,
                                )
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t,src_l2, src_l3,src_l4, K, c2w, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t,src_l2, src_l3,src_l4, K, c2w, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int,src_l2, src_l3,src_l4, K, c2w, x_self_cond = None,noise = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times,src_l2, src_l3,src_l4, K, c2w, x_self_cond = x_self_cond, clip_denoised = True)
        
        if noise == None:
            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0

        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape,img, src_l2, src_l3,src_l4, K, c2w,step_noise, return_all_timesteps = False):
        batch, device = shape[0], self.device

        c2w = c2w.transpose(0,1)
        # img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # self.num_timesteps -> self.sampling_timesteps
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None

            # noise = step_noise[:,t]
            # if t < 100:
            #     noise = None
            # step_t = arr[t]
            noise = None

            img, x_start = self.p_sample(img, t,src_l2, src_l3,src_l4, K, c2w, self_cond,noise = noise)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape,img, src_l2, src_l3,src_l4, K, c2w, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        c2w = c2w.transpose(0,1)

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, src_l2, src_l3,src_l4, K, c2w, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self,H=144, W=176, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, H, W), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, src_l2, src_l3,src_l4, K, c2w, noise = None, offset_noise_strength = None):

        '''
            x_start: 要預測的 target image
            K: 共同的intrinsic (4x4)
            c2w: 兩張圖的 world to camera matrix (4x4)
        '''

        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t,
                                src_l2 = src_l2, 
                                src_l3 = src_l3,
                                src_l4 = src_l4,
                                K = K, 
                                c2w = c2w,
                                x_self_cond = x_self_cond,
                                )

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img_seq, src_l2, src_l3,src_l4, K, c2w):
        b, time, c, h, w, device = *img_seq.shape, img_seq.device
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img_seq = img_seq.transpose(0,1)
        c2w = c2w.transpose(0,1)

        prev_img = img_seq[0]
        now_img = img_seq[1]

        #* 512x512 -> 64x64
        if self.do_latent:
            with torch.no_grad():
                latents = self.vae.encode(now_img).latent_dist.sample()
                now_img = 0.18215 * latents

        # img = self.normalize(now_img)
        loss = self.p_losses(now_img, t, src_l2, src_l3,src_l4,K,c2w)

        return loss




