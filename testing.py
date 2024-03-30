import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
import torch

a = torch.arange(8).reshape(2,4)

print(a)

a = rearrange(a,'b (m d) -> (b m) d',m=2)

print(a)

a = rearrange(a,'(b m) d -> b (m d)',m=2)

print(a)

b = torch.arange(8).reshape(2,2,2)

print(b)

b = repeat(b,'i j k -> (i repeat) j k',repeat = 2)

print(b)
