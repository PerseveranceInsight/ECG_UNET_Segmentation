import troch
import torch.nn as nn
import torch.nn.functional as nn_func

class unet_1d_conv_block(nn.Module):
    def __init__(self, kernel_parameters):
        super(unet_1d_conv_block, self).__init__()
        kernel_input_channel = kernel_parameters['input_channel']
        kernel_output_channel = kernel_parameters['output_channel']
        kernel_stride = kernel_parameters['stride']
        kernel_padding = kernel_parameters['padding']
        kernel_padding_mode = kernel_parameters['padding_mode']

        self.cov1d_block = nn.Sequential(nn.Conv1d(in_channels = kernel_input_channel,
                                                   out_channels = kernel_output_channel,
                                                   stride = kernel_stride,
                                                   padding = kernel_padding,
                                                   bias = True,
                                                   padding_mode = kernel_padding_mode),
                                         nn.BatchNorm1d(num_features = kernel_output_channel),
                                         nn.ReLU(inplace=False))

                                               

