from torch import nn
import dense_correspondence_control.learning.networks.network_building_blocks as blocks
import torch
from dense_correspondence_control.learning.networks.CNN import FullyConvolutionalCnnFlexible

class UnetDecoderFlexible(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, kernels, activation='relu',
                 bias=True, parametric=True, norm_type = 'batchnorm', dropout=0.,
                 residuals=False, double_conv=False,
                 norm_layer_arguments = {}):

        try:
            assert len(encoder_channels) == len(decoder_channels) + 1 == len(kernels) + 1
        except:
            raise Exception(
                'In order to upsample to the original image resolution, the number of decoder layers must be one more than the number of encoder layers')

        super(UnetDecoderFlexible, self).__init__()
        upconvs = []

        decoder_input_channels = [encoder_channels[-1]] + \
                                 [list(reversed(encoder_channels))[i + 1] + decoder_channels[i] for i in
                                  range(len(decoder_channels) - 1)]

        decoder_output_channels = decoder_channels

        for i in range(len(decoder_input_channels)):
            _residual = False
            if residuals:
                if decoder_input_channels[i] == decoder_output_channels[i]:
                    _residual = True
                else:
                    print(
                        'Layer {} is not be a residual layer (in_channels={} != out_channels={})'.format(
                            i, decoder_input_channels[i], decoder_output_channels[i]))

            if parametric:
                upconvs.append(
                    blocks.UpConvBlockParametric(ch_in=decoder_input_channels[i], ch_out=decoder_output_channels[i],
                                                 kernel_size=kernels[i], norm_type=norm_type,
                                                 dropout=dropout,
                                                 bias=bias, activation=activation, residual=_residual,
                                                 double_conv=double_conv, norm_layer_arguments = norm_layer_arguments))
            else:
                upconvs.append(blocks.UpConvBlockNonParametric(ch_in=decoder_input_channels[i],
                                                               ch_out=decoder_output_channels[i],
                                                               kernel_size=kernels[i], norm_type=norm_type,
                                                               dropout=dropout, bias=bias, activation=activation,
                                                               residual=_residual, double_conv=double_conv,
                                                               norm_layer_arguments = norm_layer_arguments))

        self.modulelist = nn.ModuleList(upconvs)

    def forward(self, x):

        for i in range(len(self.modulelist)):
            x = self.modulelist[i](x)

        return x


class Unet(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, parametric_decoder=False, activations='relu', bias=False,
                 norm_type='batchnorm', dropout=0., residuals=False, double_conv=False, use_1x1_conv=True, concat_rgb=True,
                 final_activation='sigmoid', final_encoder_activation='relu',
                 double_convs_second_activation=None, num_output_channels=1, post_upsampling_convs = None,
                 return_features_pre_final_activation = False, norm_layer_arguments = {}, post_decoder_norm = True):

        super(Unet, self).__init__()


        encoder_kernels = [3] * (len(encoder_channels) - 1)
        encoder_strides = [2] * (len(encoder_channels) - 1)
        decoder_kernels = [3] * (len(decoder_channels))


        self.encoder = blocks.EncoderFlexible(channels=encoder_channels, kernels=encoder_kernels,
                                              strides=encoder_strides, activation=activations,
                                              final_activation=final_encoder_activation, bias=bias,
                                              norm_type=norm_type, final_norm = True, dropout=dropout,
                                              final_dropout=dropout, residuals=residuals,
                                              double_conv=double_conv,
                                              double_convs_second_activation=double_convs_second_activation,
                                              norm_layer_arguments = norm_layer_arguments)

        self.decoder = UnetDecoderFlexible(encoder_channels=encoder_channels, decoder_channels=decoder_channels,
                                           kernels=decoder_kernels, activation=activations,
                                           bias=bias, parametric=parametric_decoder, norm_type=norm_type, dropout=0.,
                                           residuals=residuals, double_conv=double_conv,
                                           norm_layer_arguments = norm_layer_arguments )


        ### Post decoder layers ###
        if (not post_decoder_norm):
            norm_type = 'none'
            norm_layer_arguments = {}

        self.concat_rbg = concat_rgb
        if self.concat_rbg:
            self.rgb_conv = blocks.DoubleConvBlock(ch_in=encoder_channels[0], ch_out=decoder_channels[-1], kernel_size=3, stride=1,
                                                   norm_type=norm_type, dropout=dropout, bias=bias,
                                                   activation=activations,
                                                   residual=residuals,
                                                   norm_layer_arguments = norm_layer_arguments)

            self.final_rgb_conv = blocks.DoubleConvBlock(ch_in=2 * decoder_channels[-1], ch_out=decoder_channels[-1],
                                                         kernel_size=3, stride=1, norm_type=norm_type, dropout=dropout,
                                                         bias=bias, activation=activations, residual=residuals,
                                                         norm_layer_arguments = norm_layer_arguments)

        self.post_upsampling_convs = None
        if(post_upsampling_convs is not None):
            post_upsampling_convs_list = []
            prev_channels = decoder_channels[-1]
            for post_upsampling_conv in post_upsampling_convs:
                channels, kernel_size, stride = post_upsampling_conv
                post_upsampling_convs_list.append(blocks.ConvBlock(prev_channels,channels,kernel_size=kernel_size, stride=stride, norm_type=norm_type,
                                                                        dropout = dropout, activation=activations, bias=bias, residual=residuals,
                                                                   norm_layer_arguments = norm_layer_arguments))
                prev_channels = channels
                decoder_channels.append(channels)

            self.post_upsampling_convs = nn.ModuleList(post_upsampling_convs_list)





        self.use_1x1_conv = use_1x1_conv
        if self.use_1x1_conv:
            self.conv_1x1 = nn.Conv2d(decoder_channels[-1], num_output_channels, kernel_size=1, padding=0, stride=1, bias=bias)

        self.final_activation = blocks.activation_func(final_activation)

        self.return_features_pre_final_activation = return_features_pre_final_activation

        self.apply(blocks.init_weights)

    def forward(self, x):

        forward_features = [x]

        for i in range(len(self.encoder.modulelist)):
            x = self.encoder.modulelist[i](x)
            forward_features.append(x)

        forward_features = list(reversed(forward_features))

        # bottleneck of unet
        x = self.decoder.modulelist[0](x)

        for i in range(1, len(self.decoder.modulelist)):
            ff = forward_features[i]
            x = torch.cat([x, ff], dim=1)
            x = self.decoder.modulelist[i](x)

        if (self.concat_rbg):
            rgb = forward_features[-1]
            rgb_features = self.rgb_conv(rgb)
            x = torch.cat([x, rgb_features], dim=1)
            x = self.final_rgb_conv(x)

        if(self.post_upsampling_convs):
            for i in range(len(self.post_upsampling_convs)):
                x = self.post_upsampling_convs[i](x)

        if (self.use_1x1_conv):
            x = self.conv_1x1(x)

        output = self.final_activation(x)

        if(self.return_features_pre_final_activation):
            return output, x
        else:
            return output








class UnetWithAuxHead(nn.Module):

    def __init__(self, encoder_channels, decoder_channels,
                 aux_channels, aux_kernels, aux_strides, aux_double_convs= False, aux_final_activation='none', aux_bias = False,

                 parametric_decoder=False, activations='relu', bias=False,
                 norm_type='batchnorm', dropout=0., residuals=False, double_conv=False, use_1x1_conv=True,
                 concat_rgb=True,
                 final_activation='sigmoid', final_encoder_activation='relu',
                 double_convs_second_activation=None, num_output_channels=1, post_upsampling_convs=None,
                 return_features_pre_final_activation=False, norm_layer_arguments={}, post_decoder_norm=True):
        super(UnetWithAuxHead, self).__init__()

        self.Unet = Unet(encoder_channels, decoder_channels, parametric_decoder, activations, bias,
                 norm_type, dropout, residuals, double_conv, use_1x1_conv, concat_rgb,
                 final_activation, final_encoder_activation,
                 double_convs_second_activation, num_output_channels, post_upsampling_convs,
                 return_features_pre_final_activation, norm_layer_arguments, post_decoder_norm)

        aux_dropout = dropout if len(aux_channels)>1 else 0.
        aux_residuals = residuals if len(aux_channels)>1 else False
        aux_norm_type = norm_type if len(aux_channels)>1 else 'none'

        self.AuxNet = FullyConvolutionalCnnFlexible( channels=aux_channels,  kernels = aux_kernels, strides=aux_strides,
                                  final_regularisations = False, dropout=aux_dropout, residuals=aux_residuals, double_conv=aux_double_convs,
                                  norm_type = aux_norm_type, final_activation = 'none', activations = activations, bias = aux_bias)

        self.final_activation = blocks.activation_func(aux_final_activation)


    def forward(self,x):
        unet_out = self.Unet(x)
        aux_out_logits = self.AuxNet(unet_out)
        aux_out = self.final_activation(aux_out_logits)

        return unet_out, aux_out, aux_out_logits