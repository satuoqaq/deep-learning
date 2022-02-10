# from typing import Optional, Callable
#
# import torch
# from torch import Tensor
# import torch.nn as nn
# from torch.nn import functional as F
#
#
# def _make_divisible(ch, divisor=8, min_ch=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_ch is None:
#         min_ch = divisor
#     new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_ch < 0.9 * ch:
#         new_ch += divisor
#     return new_ch
#
#
# class ConvBNActivation(nn.Sequential):
#     def __init__(self,
#                  in_planes: int,
#                  out_planes: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  groups: int = 1,
#                  norm_layer: Optional[Callable[..., nn.Module]] = None,
#                  activation_layer: Optional[Callable[..., nn.Module]] = None):
#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if activation_layer is None:
#             activation_layer = nn.SiLU
#
#         super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
#                                                          out_channels=out_planes,
#                                                          kernel_size=kernel_size,
#                                                          stride=stride,
#                                                          groups=groups,
#                                                          padding=padding,
#                                                          bias=False),
#                                                norm_layer(out_planes),
#                                                activation_layer())
#
#
# class SqueezeExcitation(nn.Module):
#     def __init__(self,
#                  input_c: int,
#                  expand_c, int,
#                  squeeze_factor: int = 4):
#         super(SqueezeExcitation, self).__init__()
#         squeeze_c = input_c // squeeze_factor
#         self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
#         self.ac1 = nn.SiLU()
#         self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
#         self.ac2 = nn.Sigmoid()
#
#     def forward(self, x: Tensor) -> Tensor:
#         scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
#         scale = self.fc1(scale)
#         scale = self.ac1(scale)
#         scale = self.fc2(scale)
#         scale = self.ac2(scale)
#         return scale * x
#
#
# class InvertedResidualConfig:
#     def __init__(self,
#                  kernel_size: int,
#                  input_c: int,
#                  out_c: int,
#                  expanded_ratio: int,
#                  stride: int,
#                  use_se: bool,
#                  drop_out:float,
#                  index:str,
#                  with_coefficent
#                  ):
