from torch import nn

import numpy as np
import dense_correspondence_control.learning.networks.network_building_blocks as blocks
import torch

class CnnFlexible(nn.Module):

    def __init__(self, input_dimensions,  encoder_channels,regressor_layers,  kernels = None, strides=None,  dropout=0.,
                 residuals=False, double_conv=True, norm_type = 'batchnorm', final_activation = 'None', activations = 'relu',
                 bias = True, return_logits = False):
        super().__init__()


        encoder_final_norm = regressor_layers is not None
        encoder_final_dropout = dropout if (regressor_layers is not None) else 0.
        encoder_final_activation = final_activation if (regressor_layers is None) else activations


        if(kernels is None):
            kernels = [3] * (len(encoder_channels)-1)
        else:
            assert len(kernels) == len(encoder_channels)-1

        if(strides is None):
            strides = [2] * (len(encoder_channels)-1)
        else:
            assert len(strides) == len(kernels)

        self.encoder = blocks.EncoderFlexible(channels=encoder_channels, kernels=kernels,
                                              strides=strides, activation=activations,
                                              final_activation=encoder_final_activation, bias=bias,
                                              norm_type=norm_type, final_norm=encoder_final_norm, dropout=dropout,
                                              final_dropout=encoder_final_dropout, residuals=residuals,
                                              double_conv=double_conv)
        input_size = input_dimensions
        encoding_size = [encoder_channels[-1],
                         int(np.ceil(input_size[0] / (2 ** (len(encoder_channels) - 1)))),
                         int(np.ceil(input_size[1] / (2 ** (len(encoder_channels) - 1))))]




        self.regressor_neurons = [int(np.prod(encoding_size))] + regressor_layers

        self.return_logits = return_logits
        if(return_logits):
            self.final_activation = blocks.activation_func(final_activation)
            final_activation = 'None'

        self.regressor = blocks.MLPFlexible(self.regressor_neurons, activation=activations, final_activation=final_activation,
                                            dropout=dropout, bias=True, norm_type=norm_type, final_norm=False,
                                             final_dropout=False)


        self.apply(blocks.init_weights)

    def forward(self, x):
        out = self.encoder(x)


        out = out.contiguous().view(-1, self.regressor_neurons[0])
        out = self.regressor(out)

        if(self.return_logits):
            return out, self.final_activation(out)

        return out






class FullyConvolutionalCnnFlexible(nn.Module):

    def __init__(self,  channels,  kernels = None, strides=None,  dropout=0., final_regularisations = False,
                 residuals=False, double_conv=True, norm_type = 'batchnorm', final_activation = 'none', activations = 'relu',
                 bias = True):
        super().__init__()

        if(kernels is None):
            kernels = [3] * (len(channels)-1)
        else:
            assert len(kernels) == len(channels)-1

        if(strides is None):
            strides = [1] * (len(channels)-1)
        else:
            assert len(strides) == len(kernels)

        self.encoder = blocks.EncoderFlexible(channels=channels, kernels=kernels,
                                              strides=strides, activation=activations,
                                              final_activation=final_activation, bias=bias,
                                              norm_type=norm_type, final_norm=final_regularisations, dropout=dropout,
                                              final_dropout=final_regularisations, residuals=residuals,
                                              double_conv=double_conv)
        self.apply(blocks.init_weights)

    def forward(self, x):
        out = self.encoder(x)

        return out





class SiameseCnnFlexible(nn.Module):

    def __init__(self, input_dimensions,  encoder_channels,regressor_layers,  kernels = None, strides=None,  dropout=0.,
                 residuals=False, double_conv=True, norm_type = 'batchnorm', final_activation = 'None', activations = 'relu',
                 bias = True):
        super().__init__()


        encoder_final_norm = regressor_layers is not None
        encoder_final_dropout = dropout if (regressor_layers is not None) else 0.
        encoder_final_activation = final_activation if (regressor_layers is None) else activations


        if(kernels is None):
            kernels = [3] * (len(encoder_channels)-1)
        else:
            assert len(kernels) == len(encoder_channels)-1

        if(strides is None):
            strides = [2] * (len(encoder_channels)-1)
        else:
            assert len(strides) == len(kernels)

        self.encoder = blocks.EncoderFlexible(channels=encoder_channels, kernels=kernels,
                                              strides=strides, activation=activations,
                                              final_activation=encoder_final_activation, bias=bias,
                                              norm_type=norm_type, final_norm=encoder_final_norm, dropout=dropout,
                                              final_dropout=encoder_final_dropout, residuals=residuals,
                                              double_conv=double_conv)
        input_size = input_dimensions
        encoding_size = [encoder_channels[-1],
                         int(np.ceil(input_size[0] / (2 ** (len(encoder_channels) - 1)))),
                         int(np.ceil(input_size[1] / (2 ** (len(encoder_channels) - 1))))]



        self.encoding_len = int(np.prod(encoding_size))
        self.regressor_neurons = [2*self.encoding_len] + regressor_layers

        self.regressor = blocks.MLPFlexible(self.regressor_neurons, activation=activations, final_activation=final_activation,
                                            dropout=dropout, bias=True, norm_type=norm_type, final_norm=False,
                                        final_dropout=False)


        self.apply(blocks.init_weights)

    def forward(self, x1,x2):
        encoding1 = self.encoder(x1)
        encoding1 = encoding1.contiguous().view(-1, self.encoding_len)
        encoding2 = self.encoder(x2)
        encoding2 = encoding2.contiguous().view(-1, self.encoding_len)

        encoding = torch.cat([encoding1,encoding2], dim =1)

        out = self.regressor(encoding)

        return out



