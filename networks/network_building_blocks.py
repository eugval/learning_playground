from torch import nn
import torch
import numpy as np


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)  # Currently, convolutional layers do not have a bias

    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    if type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['Relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['Leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['Selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()],
        ['None', nn.Identity()],
        ["sigmoid", nn.Sigmoid()],
        ["Sigmoid", nn.Sigmoid()],
        ["tanh", nn.Tanh()],
        ["Tanh", nn.Tanh()],
        ["spatialsoftargmax", SpatialSoftArgmax(normalise=True)],
        ["Spatialsoftargmax", SpatialSoftArgmax(normalise=True)],
        ["spatialsoftargmax_with_map", SpatialSoftArgmax(normalise=True, return_map=True)],
        ["Spatialsoftargmax_with_map", SpatialSoftArgmax(normalise=True, return_map=True)],
        ["softmax", nn.Softmax(dim=1)],
        ["Softmax", nn.Softmax(dim=1)]
    ])[activation]



def select_normalisation(dims, norm_type = 'none', dimension=1, arguments = {}):
    if(norm_type == 'none'):
        return nn.Identity()
    elif(norm_type == 'batchnorm'):
        if(dimension ==1):
            return nn.BatchNorm1d(dims, **arguments)
        elif(dimension ==2):
            return nn.BatchNorm2d(dims, **arguments)
        elif(dimension ==3):
            return nn.BatchNorm3d(dims,**arguments)
        else:
            raise ValueError('Batchnorm dimensions can only be 1,2,3')
    elif(norm_type == 'layernorm'):
        return nn.LayerNorm(dims, **arguments)
    elif(norm_type == 'instancenorm'):
        if(dimension ==1):
            #TODO: Instancenorm1d seems buggy, so replacing it with batcnorm that does not tracks stats
            return nn.InstanceNorm1d(dims, **arguments)
        elif (dimension == 2):
            return nn.InstanceNorm2d(dims, **arguments)
        elif (dimension == 3):
            return nn.InstanceNorm3d(dims, **arguments)
        else:
            raise ValueError('InstanceNorm dimensions can only be 1,2,3')
    elif (norm_type == 'groupnorm'):
        raise NotImplementedError()

    else:
        raise NotImplementedError()



class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y





def create_gaussian_map(mu, dim_X, dim_Y, std, alphas=None):
    # y = torch.from_numpy(np.linspace(-dim_Y / dim_X, dim_Y / dim_X, dim_Y)).to(mu.device).double()
    y = torch.from_numpy(np.linspace(-1, 1, dim_Y)).to(mu.device).float()
    x = torch.from_numpy(np.linspace(-1, 1, dim_X)).to(mu.device).float()

    y = y.view((1, 1, dim_Y))
    x = x.view((1, 1, dim_X))

    gauss_y = torch.exp(-torch.square(torch.abs((mu[:, :, 1].unsqueeze(2) - y) * (1 / (std + 1e-4)))))
    gauss_x = torch.exp(-torch.square(torch.abs((mu[:, :, 0].unsqueeze(2) - x) * (1 / (std + 1e-4)))))

    gauss_y = gauss_y.unsqueeze(2)
    gauss_x = gauss_x.unsqueeze(3)
    gauss_xy = gauss_x * gauss_y

    if (alphas is not None):
        gauss_xy = alphas.unsqueeze(2).unsqueeze(3) * gauss_xy

    return gauss_xy





class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False, return_map=False):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True) if temperature is None else nn.Parameter(
            torch.ones(1) * temperature.cpu().numpy()[0], requires_grad=False).to(temperature.device)
        self.normalise = normalise
        self.return_map = return_map

    def forward(self, x):
        n, c, h, w = x.size()

        spatial_softmax_per_map = nn.functional.softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        # (N, C, 2)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])

        if (self.return_map):
            # TODO: deubg here
            out = create_gaussian_map(out, h, w, 0.05, None)

        return out







class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride,
                 norm_type='batchnorm', dropout=0., bias=True,
                 activation='relu', residual=False, norm_layer_arguments = {}):
        super(ConvBlock, self).__init__()
        assert kernel_size % 2. == 1.

        if (residual):
            assert ch_in == ch_out
            # self.pooling = nn.AvgPool2d(kernel_size=stride, stride=stride)  # TODO
            self.pooling = nn.MaxPool2d(kernel_size=stride, stride=stride)
            stride = 1

        self.residual = residual

        modules = []
        modules.append(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias))

        if (norm_type != 'none'):
            #nn.BatchNorm2d(ch_out, track_running_stats=track_running_stats_batchnorm)
            # TODO: This currently does not support proper, multi-dimensional layer normalisation
            modules.append(select_normalisation(ch_out, norm_type = norm_type,dimension =2, arguments=norm_layer_arguments))

        self.conv = nn.Sequential(*modules)

        if (dropout > 0):
            self.dropout = nn.Dropout2d(dropout, False)
        else:
            self.dropout = None

        self.activation = activation_func(activation)

        self.apply(init_weights)

    def forward(self, x):
        if (self.residual):
            residual = x
            x = self.conv(x)
            x += residual
            x = self.activation(x)
            x = self.pooling(x)
        else:
            x = self.conv(x)
            x = self.activation(x)

        if (self.dropout is not None):
            x = self.dropout(x)
        return x




class DoubleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, norm_type='batchnorm', dropout=0., bias=True, activation="relu",
                 residual=False, final_activation = None, final_norm = True, norm_layer_arguments = {}):
        super(DoubleConvBlock, self).__init__()
        assert kernel_size % 2. == 1.

        if (residual):
            assert ch_in == ch_out
            # self.pooling = nn.AvgPool2d(kernel_size=stride, stride=stride)  # TODO
            self.pooling = nn.MaxPool2d(kernel_size=stride, stride=stride)  # TODO
            stride = 1

        self.residual = residual

        convs = []

        convs.append(
            ConvBlock(ch_in, ch_out, kernel_size=kernel_size, stride=stride, norm_type = norm_type, dropout=dropout,
                      bias=bias, activation=activation, residual=False, norm_layer_arguments = norm_layer_arguments ))


        if(not final_norm):
            norm_type = 'none'
            norm_layer_arguments = {}

        convs.append(
            ConvBlock(ch_out, ch_out, kernel_size=kernel_size, stride=1, norm_type=norm_type, dropout=False, bias=bias,
                      activation="none", residual=False, norm_layer_arguments = norm_layer_arguments))

        if (dropout > 0):
            self.dropout = nn.Dropout2d(dropout, False)
        else:
            self.dropout = None

        if(final_activation is not None):
            self.activation = activation_func(final_activation)
        else:
            self.activation = activation_func(activation)

        self.conv = nn.Sequential(*convs)

        self.apply(init_weights)

    def forward(self, x):
        if (self.residual):
            residual = x
            x = self.conv(x)
            x += residual
            x = self.activation(x)
            x = self.pooling(x)
        else:
            x = self.conv(x)
            x = self.activation(x)

        if (self.dropout is not None):
            x = self.dropout(x)

        return x






#
# %  ch_in= unet_decoder_channels[i] , ch_out = unet_decoder_channels[i],
#                     kernel_size =3 , stride=1, norm_type=norm_type, bias=bias, activation="relu",
#                     norm_layer_arguments={}
class FilmBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, norm_type='batchnorm', bias=True, activation="relu", norm_layer_arguments = {}):
        super(FilmBlock, self).__init__()
        assert kernel_size % 2. == 1.


        self.conv1 =  ConvBlock(ch_in, ch_out, kernel_size=kernel_size, stride=stride, norm_type = norm_type, dropout=0.,
                      bias=bias, activation=activation, residual=False, norm_layer_arguments = norm_layer_arguments )


        self.conv2 =  ConvBlock(ch_out, ch_out, kernel_size=kernel_size, stride=1, norm_type=norm_type, dropout=0., bias=bias,
                      activation="none", residual=False, norm_layer_arguments = norm_layer_arguments)

        self.activation = activation_func('relu')



        self.apply(init_weights)

    def forward(self, x, beta, gamma, neutralise_film = False):
        if(neutralise_film):
            x2 = x
        else:
            x = self.conv1(x)

            x2 = self.conv2(x)
            x2 = x2*beta.unsqueeze(2).unsqueeze(3) + gamma.unsqueeze(2).unsqueeze(3)

            x2 = self.activation(x2) + x

        return x2









class UpConvBlockNonParametric(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, norm_type='batchnorm', dropout=0., bias=True,
                 activation='relu', residual=False, double_conv=False, mode='bilinear',
                 double_conv_final_activation=None,
                 norm_layer_arguments={},
                 double_conv_final_norm = True,

                 ):
        super(UpConvBlockNonParametric, self).__init__()

        modules = []
        if mode in ['linear', 'bilinear', 'trilinear']:
            modules.append(nn.Upsample(scale_factor=2, mode=mode, align_corners=True))
        else:
            modules.append(nn.Upsample(scale_factor=2, mode=mode))

        if (double_conv):
            modules.append(
                DoubleConvBlock(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size, stride=1, norm_type=norm_type,
                                dropout=dropout, bias=bias, activation=activation, residual=residual,
                                final_activation=double_conv_final_activation,
                                norm_layer_arguments = norm_layer_arguments,
                                final_norm = double_conv_final_norm))
        else:
            modules.append(
                ConvBlock(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size, stride=1, norm_type =norm_type,
                          dropout=dropout, bias=bias, activation=activation, residual=residual,
                          norm_layer_arguments = norm_layer_arguments))

        self.up_conv = nn.Sequential(*modules)

        self.apply(init_weights)

    def forward(self, x):
        x = self.up_conv(x)
        return x


class UpConvBlockParametric(nn.Module):
    def __init__(self,
                 ch_in, ch_out, kernel_size, norm_type= 'batchnorm', dropout=0., bias=True,
                 activation ='relu', residual=False, double_conv=False, double_conv_final_activation = None,
                 double_conv_final_norm = True,
                 norm_layer_arguments = {},
                 ):
        super(UpConvBlockParametric, self).__init__()

        modules = []

        modules.append(nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=2,
                                          padding=kernel_size // 2, output_padding=1, bias=True, dilation=1))

        if (double_conv):
            modules.append(
                DoubleConvBlock(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size, stride=1,
                                norm_type=norm_type, dropout=dropout, bias=bias, activation=activation,
                                residual=residual, final_activation = double_conv_final_activation,
                                norm_layer_arguments = norm_layer_arguments,
                                final_norm = double_conv_final_norm))
        else:
            modules.append(
                ConvBlock(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size, stride=1, norm_type=norm_type,
                          dropout=dropout, bias=bias, activation=activation, residual=residual,
                          norm_layer_arguments = norm_layer_arguments))

        self.up_conv = nn.Sequential(*modules)

        self.apply(init_weights)

    def forward(self, x):
        x = self.up_conv(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, norm_type='batchnorm', dropout=0., bias=True,
                 activation='relu', norm_layer_arguments = {}):

        super(LinearBlock, self).__init__()



        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        if (norm_type != 'none'):
            # nn.BatchNorm1d(out_features)
            # TODO: This currently does not support proper, multi-dimensional layer normalisation
            self.norm = select_normalisation(out_features, norm_type=norm_type, dimension =1, arguments=norm_layer_arguments)
        else:
            self.norm = nn.Identity()


        if (dropout > 0):
            self.dropout = nn.Dropout(dropout, False)
        else:
            self.dropout = None

        self.activation = activation_func(activation)

        self.apply(init_weights)

    def forward(self, x):

        x = self.linear(x)

        if(isinstance(self.norm, nn.InstanceNorm1d)):
            #TODO: bug with instance norm pytorch requires by hand implementaiton
            mean = torch.mean(x, dim=1, keepdim=True)
            var = torch.var(x, dim =1 ,unbiased=False, keepdim = True)
            std = torch.sqrt(var +1.e-5)
            x = (x-mean)/std
        else:
            x = self.norm(x)

        x = self.activation(x)
        if (self.dropout is not None):
            x = self.dropout(x)
        return x

# class LinearBlock(nn.Module):
#     def __init__(self, in_features, out_features, norm_type='batchnorm', dropout=0., bias=True,
#                  activation='relu', norm_layer_arguments={}):
#
#         super(LinearBlock, self).__init__()
#
#         modules = []
#
#         modules.append(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
#
#         if (norm_type != 'none'):
#             # nn.BatchNorm1d(out_features)
#             # TODO: This currently does not support proper, multi-dimensional layer normalisation
#             modules.append(select_normalisation(out_features, norm_type=norm_type, dimension=1,
#                                                 arguments=norm_layer_arguments))
#
#         self.linear = nn.Sequential(*modules)
#
#         if (dropout > 0):
#             self.dropout = nn.Dropout(dropout, False)
#         else:
#             self.dropout = None
#
#         self.activation = activation_func(activation)
#
#         self.apply(init_weights)
#
#     def forward(self, x):
#
#         x = self.linear(x)
#         x = self.activation(x)
#         if (self.dropout is not None):
#             x = self.dropout(x)
#         return x


class EncoderFlexible(nn.Module):
    def __init__(self, channels, kernels, strides, activation='relu', final_activation = 'none', bias=True,
                 norm_type='batchnorm', final_norm = False, dropout=0., final_dropout=0., residuals=False,
                 double_conv=False, double_convs_second_activation = None, norm_layer_arguments = {},
                 extra_channels_on_conv_output = None):

        super(EncoderFlexible, self).__init__()


        if(extra_channels_on_conv_output is  None):
            extra_channels_on_conv_output = 0

        convs = []

        for i in range(len(channels) - 2):
            _residual = False
            if residuals:
                if channels[i]+ extra_channels_on_conv_output == channels[i + 1]:
                    _residual = True
                else:
                    print(
                        'Layer {} is not be a residual layer (in_channels={} != out_channels={})'.format(i, channels[i],
                                                                                                         channels[
                                                                                                             i + 1]))

            if (double_conv):
                convs.append(
                    DoubleConvBlock(ch_in=channels[i]+extra_channels_on_conv_output, ch_out=channels[i + 1],
                                    kernel_size=kernels[i], stride=strides[i],
                                    norm_type=norm_type, dropout=dropout, bias=bias,
                                    activation=activation, residual=_residual, final_activation = double_convs_second_activation ,
                                    norm_layer_arguments = norm_layer_arguments))
            else:
                convs.append(
                    ConvBlock(ch_in=channels[i]+extra_channels_on_conv_output, ch_out=channels[i + 1],
                              kernel_size=kernels[i], stride=strides[i],
                              norm_type=norm_type, dropout=dropout, bias=bias,
                              activation=activation, residual=_residual,
                              norm_layer_arguments = norm_layer_arguments))

        ### Final Layer
        _residual = False
        if residuals:
            if channels[-2] == channels[-1]:
                _residual = True
            else:
                print(
                    'Last layer is not a residual layer (in_channels={} != out_channels={})'.format(channels[-2],
                                                                                                    channels[-1]))

        if (not final_norm):
            norm_type = 'none'
            norm_layer_arguments = {}

        if (double_conv):
            convs.append(DoubleConvBlock(ch_in=channels[-2]+extra_channels_on_conv_output, ch_out=channels[-1],
                                         kernel_size=kernels[-1], stride=strides[-1],
                                         norm_type=norm_type, dropout=final_dropout, bias=bias,
                                         activation=activation, final_activation=final_activation, residual=_residual,
                                         norm_layer_arguments = norm_layer_arguments))
        else:
            convs.append(ConvBlock(ch_in=channels[-2]+extra_channels_on_conv_output, ch_out=channels[-1],
                                   kernel_size=kernels[-1], stride=strides[-1],
                                   norm_type=norm_type, dropout=final_dropout, bias=bias,
                                   activation=final_activation, residual=_residual,
                                   norm_layer_arguments = norm_layer_arguments))

        self.modulelist = nn.ModuleList(convs)

        self.apply(init_weights)

    def forward(self, x):

        for i in range(len(self.modulelist)):
            x = self.modulelist[i](x)

        return x


class DecoderFlexible(nn.Module):
    def __init__(self, channels, kernels, activation='relu', final_activation='none',  bias=True, parametric=True,
                 norm_type = 'batchnorm', final_norm=False, dropout=0., final_dropout=0., residuals=False,
                 double_conv=False, norm_layer_arguments = {},
                 ):

        super(DecoderFlexible, self).__init__()
        upconvs = []

        for i in range(len(channels) - 2):
            _residual = False
            if residuals:
                if channels[i] == channels[i + 1]:
                    _residual = True
                else:
                    print(
                        'Layer {} is not be a residual layer (in_channels={} != out_channels={})'.format(i, channels[i],
                                                                                                         channels[
                                                                                                             i + 1]))
            if parametric:
                upconvs.append(UpConvBlockParametric(ch_in=channels[i], ch_out=channels[i + 1],
                                                     kernel_size=kernels[i], norm_type=norm_type,
                                                     dropout=dropout,
                                                     bias=bias, activation=activation, residual=_residual,
                                                     double_conv=double_conv,
                                                     norm_layer_arguments = norm_layer_arguments))
            else:
                upconvs.append(UpConvBlockNonParametric(ch_in=channels[i], ch_out=channels[i + 1],
                                                        kernel_size=kernels[i], norm_type=norm_type,
                                                        dropout=dropout, bias=bias, activation=activation,
                                                        residual=_residual, double_conv=double_conv,
                                                        norm_layer_arguments = norm_layer_arguments))

        ### Final Layer
        _residual = False
        if residuals:
            if channels[-2] == channels[-1]:
                _residual = True
            else:
                print(
                    'Last layer is not a residual layer (in_channels={} != out_channels={})'.format(channels[-2],  channels[-1]))


        if (not final_norm):
            norm_type = 'none'
            norm_layer_arguments = {}

        if parametric:
            if(double_conv):
                upconvs.append(UpConvBlockParametric(ch_in=channels[-2], ch_out=channels[-1],
                                                     kernel_size=kernels[-1], norm_type=norm_type,
                                                     dropout=final_dropout, bias=bias, activation=activation, double_conv_final_activation=final_activation,
                                                     double_conv_final_norm = final_norm,
                                                     residual=_residual, double_conv=double_conv,
                                                     norm_layer_arguments = norm_layer_arguments))
            else:
                upconvs.append(UpConvBlockParametric(ch_in=channels[-2], ch_out=channels[-1],
                                                     kernel_size=kernels[-1], norm_type=norm_type,
                                                     dropout=final_dropout, bias=bias, activation=final_activation,
                                                     residual=_residual, double_conv=double_conv,
                                                     norm_layer_arguments = norm_layer_arguments))

        else:
            if (double_conv):
                upconvs.append(UpConvBlockNonParametric(ch_in=channels[-2], ch_out=channels[-1],
                                                        kernel_size=kernels[-1], norm_type = norm_type, double_conv_final_norm = final_norm,
                                                        dropout=final_dropout, bias=bias,
                                                        activation=activation, double_conv_final_activation=final_activation,
                                                        residual=_residual, double_conv=double_conv,
                                                        norm_layer_arguments = norm_layer_arguments))
            else:
                upconvs.append(UpConvBlockNonParametric(ch_in=channels[-2], ch_out=channels[-1],
                                                        kernel_size=kernels[-1], norm_type = norm_type,
                                                        dropout=final_dropout, bias=bias,
                                                        activation=final_activation,
                                                        residual=_residual, double_conv=double_conv,
                                                        norm_layer_arguments = norm_layer_arguments))


        self.modulelist = nn.ModuleList(upconvs)

        self.apply(init_weights)

    def forward(self, x):

        for i in range(len(self.modulelist)):
            x = self.modulelist[i](x)

        return x


class MLPFlexible(nn.Module):
    def __init__(self, neurons, activation="relu",
                 final_activation="none", dropout=0., bias=True, norm_type='none', final_norm=False,
                 final_dropout=0., norm_layer_arguments = {}):
        super().__init__()

        self.layer_dims = neurons

        modules = []

        for i in range(len(neurons) - 2):
            modules.append(LinearBlock(neurons[i], neurons[i + 1], norm_type=norm_type, dropout=dropout, bias=bias,
                                       activation=activation, norm_layer_arguments = norm_layer_arguments))

        ## Final Layer ##

        if(not final_norm):
            norm_type = 'none'
            norm_layer_arguments =  {}

        modules.append(
            LinearBlock(neurons[-2], neurons[-1], norm_type=norm_type, dropout=final_dropout, bias=bias,
                        activation=final_activation, norm_layer_arguments = norm_layer_arguments))

        self.modulelist = nn.ModuleList(modules)

        self.apply(init_weights)

    def forward(self, x):

        for i in range(len(self.modulelist)):
            x = self.modulelist[i](x)

        return x
