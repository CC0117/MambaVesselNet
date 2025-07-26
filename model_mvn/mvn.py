# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    # normalize input features to have mean 0 and standard deviation 1
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        # normalize the input before passing it to the Mamba layer
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # specify the type of Mamba layer to use
            bimamba_type="v2",
        )

    def forward(self, x):
        B, C = x.shape[:2]
        # print(f"B: {B}, C: {C}")
        # print("dim: ", self.dim)

        assert C == self.dim
        # compute the number of tokens
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # flatten the spatial dimensions, features are treated as sequence of tokens
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # apply layer normalization
        x_norm = self.norm(x_flat)
        # process the input with Mamba layer
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class MlpChannel(nn.Module):
    """Used within the MambaEncoder class, to process and transform the input features."""

    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # down-sampling layers
        """initialize a series of down-sampling layers, 
        each layer consists of a 3D convolution and a layer normalization"""
        """GSC Block"""
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # create 3 down-sampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # encoding stages, each stage consists of a sequence of Mamba layers
        """Tri-orientated Mamba Tom"""
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                # depth[i] determines the number of Mamba layers in the i-th stage
                *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # the indices of the output features
        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            """norm the layer then pass to MLP"""
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            # create a multi-layer perceptron (MLP) for each stage
            self.mlps.append(MlpChannel(dims[i_layer], 4 * dims[i_layer]))

    """This structure allows the MambaEncoder to apply a series of down-sampling and Mamba transformations, 
    followed by normalization and MLP processing, to extract and refine features from the input data. 
    The use of out_indices enables selective inclusion of intermediate outputs, 
    which can be useful for multi-scale feature extraction commonly used in segmentation tasks."""

    def forward_features(self, x):
        outs = []
        for i in range(4):
            # GSC Block
            x = self.downsample_layers[i](x)
            # Mamba Block
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # apply layer normalization
                x_out = norm_layer(x)
                # apply MLP
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, downsample=True):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
            )

        self.mamba_layers = nn.Sequential(
            *[MambaLayer(dim=out_channels) for _ in range(depth)]
        )

        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(out_channels, out_channels * 4)

    def forward(self, x):
        if self.downsample:
            x = self.downsample_layer(x)

        x = self.mamba_layers(x)
        x = self.norm(x)
        x = self.mlp(x)

        return x


def downsample_layer(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
    )


class mvnNet (nn.Module):
    def __init__(self, in_chans=1, out_chans=13, feature_dims=None, norm_name="instance", spatial_dims=3,
                 res_block=True):
        super().__init__()

        if feature_dims is None:
            feature_dims = [48, 96, 192, 384, 768]
        #
        # self.stem = nn.Sequential(
        #     nn.Sequential(
        #         nn.Conv3d(in_chans, feature_dims[0], kernel_size=7, stride=2, padding=3),
        #         LayerNorm(feature_dims[0], eps=1e-6, data_format="channels_first"),
        #     )
        # )

        # Encoder
        self.enc_conv1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feature_dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample1 = downsample_layer(feature_dims[0], feature_dims[0])

        self.enc_conv2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[0],
            out_channels=feature_dims[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample2 = downsample_layer(feature_dims[1], feature_dims[1])

        self.enc_conv3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[1],
            out_channels=feature_dims[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample3 = downsample_layer(feature_dims[2], feature_dims[2])

        self.enc_conv4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[2],
            out_channels=feature_dims[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample4 = downsample_layer(feature_dims[3], feature_dims[3])

        self.enc_conv5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[3],
            out_channels=feature_dims[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Encoder Mamba Blocks
        self.enc_mamba1 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.enc_mamba2 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.enc_mamba3 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.enc_mamba4 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)

        # Decoder Mamba Blocks
        self.dec_mamba1 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.dec_mamba2 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.dec_mamba3 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.dec_mamba4 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)

        # Decoder Conv Layers
        self.dec_conv5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[4],
            out_channels=feature_dims[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[3],
            out_channels=feature_dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[2],
            out_channels=feature_dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[1],
            out_channels=feature_dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[0],
            out_channels=feature_dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Final output layer
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_dims[0], out_channels=out_chans)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc1_down = self.downsample1(enc1)

        enc2 = self.enc_conv2(enc1_down)
        enc2_down = self.downsample2(enc2)

        enc3 = self.enc_conv3(enc2_down)
        enc3_down = self.downsample3(enc3)

        enc4 = self.enc_conv4(enc3_down)
        enc4_down = self.downsample4(enc4)

        enc5 = self.enc_conv5(enc4_down)

        x = self.enc_mamba1(enc5)
        x = self.enc_mamba2(x)
        x = self.enc_mamba3(x)
        x = self.enc_mamba4(x)

        x = self.dec_mamba1(x)
        x = self.dec_mamba2(x)
        x = self.dec_mamba3(x)
        x = self.dec_mamba4(x)

        dec5 = self.dec_conv5(x, enc4)
        dec4 = self.dec_conv4(dec5, enc3)
        dec3 = self.dec_conv3(dec4, enc2)
        dec2 = self.dec_conv2(dec3, enc1)

        x = self.dec_conv1(dec2)
        out = self.out(x)

        return out
