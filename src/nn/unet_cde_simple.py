from .base import BaseNetwork
from .nn_utils import ConvBlock, ResConvBlock, ODEfunc, CDEBlock
import torch


class CDEUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 5,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu',
                 bilinear: bool = True,
                 use_bn: bool = True):
        '''
        A UNet model with Controlled Differential Equations.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        depth: int
            Depth of the model (encoding or decoding)
        use_residual: bool
            Whether to use residual connection within the same conv block
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        non_linearity : string
            One of 'relu' and 'softplus'
        '''
        super().__init__()

        self.device = device
        self.depth = depth
        self.use_residual = use_residual
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = torch.nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = torch.nn.Softplus()
        self.use_bn = use_bn

        n_f = num_filters  # shorthand

        if self.use_residual:
            conv_block = ResConvBlock
            upconv_block = ResConvBlock
        else:
            conv_block = ConvBlock
            upconv_block = ConvBlock

        # This is for the encoder.
        self.conv1x1 = torch.nn.Conv2d(in_channels, n_f, 1, 1)
        self.down_list = torch.nn.ModuleList([])
        self.down_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            if self.use_bn:
                self.down_conn_list.append(torch.nn.Sequential(
                    torch.nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1),
                    torch.nn.BatchNorm2d(n_f * 2 ** (d + 1)),
                ))
            else:
                self.down_conn_list.append(torch.nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))

        self.bottleneck = conv_block(n_f * 2 ** self.depth)

        if not self.bilinear:
            self.pooling = torch.nn.MaxPool2d(2)

        # This is for the decoder.
        self.cde_list = torch.nn.ModuleList([])
        self.up_list = torch.nn.ModuleList([])
        self.up_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.cde_list.append(CDEBlock(ODEfunc(dim=n_f * 2 ** d)))
            self.up_list.append(upconv_block(n_f * 2 ** d))
            if self.use_bn:
                self.up_conn_list.append(torch.nn.Sequential(
                    torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1),
                    torch.nn.BatchNorm2d(n_f * 2 ** d),
                ))
            else:
                self.up_conn_list.append(torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1))
        self.cde_list = self.cde_list[::-1]
        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.cde_bottleneck = CDEBlock(ODEfunc(dim=n_f * 2 ** self.depth))
        self.out_layer = torch.nn.Conv2d(n_f, out_channels, 1)

    def time_independent_parameters(self):
        '''
        Parameters related to ODE.
        '''
        return set(self.parameters()) - set(self.cde_list.parameters()) - set(self.cde_bottleneck.parameters())

    def freeze_time_independent(self):
        '''
        Freeze paramters that are time-independent.
        '''
        for p in self.time_independent_parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        Time embedding through CDE.
        '''

        x = self.non_linearity(self.conv1x1(x))

        residual_list = []
        for d in range(self.depth):
            x = self.down_list[d](x)
            residual_list.append(x.clone())
            x = self.non_linearity(self.down_conn_list[d](x))
            if self.bilinear:
                x = torch.nn.functional.interpolate(x,
                                                    scale_factor=0.5,
                                                    mode='bilinear',
                                                    align_corners=False)
            else:
                x = self.pooling(x)

        x = self.bottleneck(x)

        # Skip CDE if no time difference.
        use_cde = not (len(t) == 1 and t.item() == 0)
        if use_cde:
            integration_time = t.float()

        if use_cde:
            x = self.cde_bottleneck(x, integration_time)

        for d in range(self.depth):
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=2,
                                                mode='bilinear',
                                                align_corners=False)
            if use_cde:
                res = self.cde_list[d](residual_list.pop(-1), integration_time)
            else:
                res = residual_list.pop(-1)
            x = torch.cat([x, res], dim=1)
            x = self.non_linearity(self.up_conn_list[d](x))
            x = self.up_list[d](x)

        output = self.out_layer(x)

        return output
