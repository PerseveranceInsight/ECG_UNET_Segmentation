import torch
import torch.nn as nn
import torch.nn.functional as nn_func

class unet_1d_conv_block(nn.Module):
    def __init__(self, kernel_parameters):
        super(unet_1d_conv_block, self).__init__()
        kernel_input_channel = kernel_parameters['input_channel']
        kernel_output_channel = kernel_parameters['output_channel']
        kernel_size = kernel_parameters['kernel_size']
        kernel_stride = kernel_parameters['stride']
        kernel_padding = kernel_parameters['padding']
        kernel_padding_mode = kernel_parameters['padding_mode']

        self.conv1d_block = nn.Sequential(nn.Conv1d(in_channels = kernel_input_channel,
                                                    out_channels = kernel_output_channel,
                                                    kernel_size = kernel_size,
                                                    stride = kernel_stride,
                                                    padding = kernel_padding,
                                                    bias = True,
                                                    padding_mode = kernel_padding_mode),
                                          nn.BatchNorm1d(num_features = kernel_output_channel),
                                          nn.ReLU(inplace=False))


    def forward(self, input):
        return self.conv1d_block(input)

class unet_1d_conv_group(nn.Module):
    def __init__(self, group_parameters, group_idx):
        super(unet_1d_conv_group, self).__init__()
        kernel_parameters_1 = {'input_channel': group_parameters['input_channel'][group_idx],
                               'output_channel': group_parameters['middle_channel'][group_idx],
                               'kernel_size': group_parameters['kernel_size'],
                               'padding': group_parameters['padding'],
                               'stride': group_parameters['stride'],
                               'padding_mode': group_parameters['padding_mode']}
        kernel_parameters_2 =  {'input_channel': group_parameters['middle_channel'][group_idx],
                                'output_channel': group_parameters['output_channel'][group_idx],
                                'kernel_size': group_parameters['kernel_size'],
                                'padding': group_parameters['padding'],
                                'stride': group_parameters['stride'],
                                'padding_mode': group_parameters['padding_mode']}
        self.conv1d_block1 = unet_1d_conv_block(kernel_parameters_1)
        self.conv1d_block2 = unet_1d_conv_block(kernel_parameters_2)

    def forward(self, input):
        middle = self.conv1d_block1(input)
        return self.conv1d_block2(middle)

class unet_1d_model(nn.Module):
    def __init__(self, conv_group_parameters, pool_parameters, deconv_group_parameters):
        super(unet_1d_model, self).__init__()
        self.unet_1d_conv_group1 = unet_1d_conv_group(group_parameters = conv_group_parameters, 
                                                      group_idx = 0)
        self.unet_1d_conv_group2 = unet_1d_conv_group(group_parameters = conv_group_parameters,
                                                      group_idx = 1)
        self.unet_1d_conv_group3 = unet_1d_conv_group(group_parameters = conv_group_parameters,
                                                      group_idx = 2)
        self.unet_1d_conv_group4 = unet_1d_conv_group(group_parameters = conv_group_parameters,
                                                      group_idx = 3)
        self.unet_1d_conv_group5 = unet_1d_conv_group(group_parameters = conv_group_parameters,
                                                      group_idx = 4)
        
        self.max_pool1d_1 = nn.MaxPool1d(kernel_size = pool_parameters['kernel_size'],
                                         stride = pool_parameters['stride'],
                                         padding = pool_parameters['padding'])
        self.max_pool1d_2 = nn.MaxPool1d(kernel_size = pool_parameters['kernel_size'],
                                         stride = pool_parameters['stride'],
                                         padding = pool_parameters['padding'])
        self.max_pool1d_3 = nn.MaxPool1d(kernel_size = pool_parameters['kernel_size'],
                                         stride = pool_parameters['stride'],
                                         padding = pool_parameters['padding'])
        self.max_pool1d_4 = nn.MaxPool1d(kernel_size = pool_parameters['kernel_size'],
                                         stride = pool_parameters['stride'],
                                         padding = pool_parameters['padding'])

    
    def forward(self, input):
        group_1_out = self.unet_1d_conv_group1(input)
        pool_1_out = self.max_pool1d_1(group_1_out)
        group_2_out = self.unet_1d_conv_group2(pool_1_out)
        pool_2_out = self.max_pool1d_2(group_2_out)
        group_3_out = self.unet_1d_conv_group3(pool_2_out)
        pool_3_out = self.max_pool1d_3(group_3_out)
        group_4_out = self.unet_1d_conv_group4(pool_3_out)
        pool_4_out = self.max_pool1d_4(group_4_out)
        group_5_out = self.unet_1d_conv_group5(pool_4_out)
        return group_5_out
                                    


if __name__ == '__main__':
    conv_group_parameters = {'input_channel': [1, 4, 8, 16, 32],
                             'middle_channel': [4, 8, 16, 32, 64],
                             'output_channel': [4, 8, 16, 32, 64],
                             'kernel_size': 9,
                             'padding': 4,
                             'stride': 1,
                             'padding_mode': 'zeros'}
    pool_parameters = {'kernel_size': 8,
                       'stride': 2,
                       'padding': 3}

    test_input = torch.randn(1, 1, 2000)        
    test_model = unet_1d_model(conv_group_parameters,
                               pool_parameters,
                               None)
    test_output = test_model.forward(test_input)
    print('size of test_output : {0}'.format(test_output.size()))

