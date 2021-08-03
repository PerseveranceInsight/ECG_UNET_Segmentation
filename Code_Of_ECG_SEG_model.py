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
    def __init__(self, encoder_parameters, 
                       pool_parameters, 
                       decoder_parameters,
                       tran_conv_parameters):
        super(unet_1d_model, self).__init__()
        self.unet_encoder1 = unet_1d_conv_group(group_parameters = encoder_parameters,
                                                group_idx = 0)
        self.unet_encoder2 = unet_1d_conv_group(group_parameters = encoder_parameters,
                                                group_idx = 1)
        self.unet_encoder3 = unet_1d_conv_group(group_parameters = encoder_parameters,
                                                group_idx = 2)
        self.unet_encoder4 = unet_1d_conv_group(group_parameters = encoder_parameters,
                                                group_idx = 3)
        self.unet_encoder5 = unet_1d_conv_group(group_parameters = encoder_parameters,
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
        self.tran_conv1 = nn.ConvTranspose1d(in_channels = tran_conv_parameters['input_channel'][0],
                                             out_channels = tran_conv_parameters['output_channel'][0],
                                             kernel_size = tran_conv_parameters['kernel_size'],
                                             stride = tran_conv_parameters['stride'],
                                             padding = tran_conv_parameters['padding'])
        self.tran_conv2 = nn.ConvTranspose1d(in_channels = tran_conv_parameters['input_channel'][1],
                                             out_channels = tran_conv_parameters['output_channel'][1],
                                             kernel_size = tran_conv_parameters['kernel_size'],
                                             stride = tran_conv_parameters['stride'],
                                             padding = tran_conv_parameters['padding'])
        self.tran_conv3 = nn.ConvTranspose1d(in_channels = tran_conv_parameters['input_channel'][2],
                                             out_channels = tran_conv_parameters['output_channel'][2],
                                             kernel_size = tran_conv_parameters['kernel_size'],
                                             stride = tran_conv_parameters['stride'],
                                             padding = tran_conv_parameters['padding'])
        self.tran_conv4 = nn.ConvTranspose1d(in_channels = tran_conv_parameters['input_channel'][3],
                                             out_channels = tran_conv_parameters['output_channel'][3],
                                             kernel_size = tran_conv_parameters['kernel_size'],
                                             stride = tran_conv_parameters['stride'],
                                             padding = tran_conv_parameters['padding'])

        self.unet_decoder1 = unet_1d_conv_group(group_parameters = decoder_parameters,
                                                group_idx = 0)
        self.unet_decoder2 = unet_1d_conv_group(group_parameters = decoder_parameters,
                                                group_idx = 1)
        self.unet_decoder3 = unet_1d_conv_group(group_parameters = decoder_parameters,
                                                group_idx = 2)
        self.unet_decoder4 = unet_1d_conv_group(group_parameters = decoder_parameters,
                                                group_idx = 3)

    
    def forward(self, input):
        num_batches = input.size()[0]
        encode_1_out = self.unet_encoder1(input)
        pool_1_out = self.max_pool1d_1(encode_1_out)
        encode_2_out = self.unet_encoder2(pool_1_out)
        pool_2_out = self.max_pool1d_2(encode_2_out)
        encode_3_out = self.unet_encoder3(pool_2_out)
        pool_3_out = self.max_pool1d_3(encode_3_out)
        encode_4_out = self.unet_encoder4(pool_3_out)
        pool_4_out = self.max_pool1d_4(encode_4_out)
        encode_5_out = self.unet_encoder5(pool_4_out)

        up_conv_1_out = self.tran_conv1(encode_5_out)

        decoder1_in = torch.cat((encode_4_out, up_conv_1_out), dim = 1) 
        decoder1_out = self.unet_decoder1(decoder1_in) 

        up_conv_2_out = self.tran_conv2(decoder1_out)
        
        decoder2_in = torch.cat((encode_3_out, up_conv_2_out), dim = 1)
        decoder2_out = self.unet_decoder2(decoder2_in)

        up_conv_3_out = self.tran_conv3(decoder2_out)

        decoder3_in = torch.cat((encode_2_out, up_conv_3_out), dim = 1)
        decoder3_out = self.unet_decoder3(decoder3_in)
        
        up_conv_4_out = self.tran_conv4(decoder3_out)

        decoder4_in = torch.cat((encode_1_out, up_conv_4_out), dim = 1)
        decoder4_out = self.unet_decoder4(decoder4_in)

        pred_seg_out = torch.argmax(decoder4_out, dim=1)
        pred_seg_out = torch.reshape(pred_seg_out,
                                     (num_batches, 1, 2000))

        return pred_seg_out
                                    
if __name__ == '__main__':
    encoder_parameters = {'input_channel': [1, 4, 8, 16, 32],
                          'middle_channel': [4, 8, 16, 32, 64],
                          'output_channel': [4, 8, 16, 32, 64],
                          'kernel_size': 9,
                          'padding': 4,
                          'stride': 1,
                          'padding_mode': 'zeros'}
    pool_parameters = {'kernel_size': 8,
                       'stride': 2,
                       'padding': 3}
    tran_conv_parameters = {'input_channel': [64, 32, 16, 8],
                            'output_channel': [64, 32, 16, 8],
                            'kernel_size': 8,
                            'stride': 2,
                            'padding': 3}
    decoder_parameters = {'input_channel': [96, 48, 24, 12],
                          'middle_channel': [32, 16, 8, 4],
                          'output_channel': [32, 16, 8, 4],
                          'kernel_size': 3,
                          'stride': 1,
                          'padding': 1,
                          'padding_mode': 'zeros'}


    test_input = torch.randn(32, 1, 2000)        
    test_model = unet_1d_model(encoder_parameters,
                               pool_parameters,
                               decoder_parameters,
                               tran_conv_parameters)
    test_output = test_model.forward(test_input)
    print('size of test_output : {0}'.format(test_output.size()))

