"""D4-equivariant scalar-feature UNet: kernel tying, not prediction ensembling.

All feature channels transform as scalar image fields. This is a stronger
restriction than a regular-representation group CNN, and must be reported.
Strided convolutions become stride-1 convolution followed by 2x2 average
pooling so even-grid downsampling commutes with D4. Widths stay [16,32,64].
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from simdiff_eval.torch_compat import install_torch_backend_compat

install_torch_backend_compat(entry_point="d4_unet64")

from diffusers import UNet2DModel
from diffusers.configuration_utils import register_to_config


def d4(x, element):
    out = torch.rot90(x, element % 4, (-2,-1))
    return out.flip(-1) if element >= 4 else out


class InvariantKernel(nn.Module):
    def forward(self, weight):
        return torch.stack([d4(weight,g) for g in range(8)]).mean(0)


def pool_after_conv(module, inputs, output):
    return F.avg_pool2d(output, 2, 2)


class D4ScalarUNet2DModel(UNet2DModel):
    @register_to_config
    def __init__(self, sample_size=128, in_channels=1, out_channels=1,
                 layers_per_block=2, block_out_channels=(16,32,64),
                 down_block_types=('DownBlock2D','DownBlock2D','AttnDownBlock2D'),
                 up_block_types=('AttnUpBlock2D','UpBlock2D','UpBlock2D'),
                 norm_num_groups=16):
        super().__init__(sample_size=sample_size,in_channels=in_channels,
            out_channels=out_channels,layers_per_block=layers_per_block,
            block_out_channels=tuple(block_out_channels),
            down_block_types=tuple(down_block_types),up_block_types=tuple(up_block_types),
            norm_num_groups=norm_num_groups)
        for name, module in list(self.named_modules()):
            if isinstance(module,nn.Conv2d):
                assert module.kernel_size[0] == module.kernel_size[1], name
                assert module.padding[0] == module.padding[1], name
                if module.stride == (2,2):
                    module.stride=(1,1)
                    module.register_forward_hook(pool_after_conv)
                elif module.stride != (1,1):
                    raise ValueError('Unaudited convolution stride: '+name)
                parametrize.register_parametrization(module,'weight',InvariantKernel())


def register_model():
    """Process-local registration for cosmodiff's getattr(diffusers, ...) loader."""
    import diffusers
    diffusers.D4ScalarUNet2DModel = D4ScalarUNet2DModel
