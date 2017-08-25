import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math
from vgg import *
from resnet import *

class CharModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=0, rnntype='RNN'):
        super(CharModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnntype = rnntype
        if rnntype == 'RNN':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        elif rnntype == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        elif rnntype == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        else:
            raise ValueError('Wrong RNN type, {} is not supported'.format(rnntype))

        if(output_size > 0 ):
            self.output = nn.Linear(hidden_size, output_size)
            n = hidden_size *  output_size
            self.output.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input, hidden=None):
        outputs, hidden_t = self.rnn(input, hidden)
        if hasattr(self, 'output'):
            outputs = self.output(outputs)
        return outputs, hidden_t

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))



class VisionModel(nn.Module):
    def __init__(self, preTrained='True'):
        super(VisionModel, self).__init__()
        # Vision Model 
        #self.vision = vgg16(pretrained=vgg_pretrained, num_classes=128)
        self.vision = resnet34(pretrained=preTrained, num_classes=128)
        # LSTM Model(temporal)
        self.rnn  = nn.LSTM(128, 128, 2, batch_first=True)

        # Language Model
        #self.lang = nn.LSTM(100, 128, 2) 
                 
        # Output 
        self.output = nn.Linear(128, 2)
        n = self.output.in_features * self.output.out_features
        self.output.weight.data.normal_(0, math.sqrt(2. / n))
        self.output.bias.data.zero_() 
                   
    def forward(self, img):
        img.cuda()
        dims = img.size()
        img_feature = self.vision(img.view(-1, dims[2], dims[3], dims[4]))
        if (dims[1] != 1):
            img_feature = img_feature.view(dims[0], dims[1], -1)

            h0 = ( Variable(torch.zeros(2, dims[0], 128)).cuda(),  Variable(torch.zeros(2, dims[0], 128)).cuda()) 
            img_feature, hn = self.rnn(img_feature, h0)
            img_feature = img_feature[:,-1, :]

        #h0 = ( Variable(torch.zeros(2, text.size(1), 128)).cuda(),  Variable(torch.zeros(2, text.size(1), 128)).cuda()) 


        #rnn_feature, hn = self.lang(text, h0 )
        #rnn_feature = rnn_feature[-1]

        pred = self.output(img_feature)

        return pred

class LangModel(nn.Module):
    def __init__(self, preTrained='True', input=100):
        super(LangModel, self).__init__()
        # Language Model
        self.lang = nn.LSTM(input, 128, 3, batch_first=True) 
                 
        # Output 
        self.output = nn.Linear(128, 2)
        n = self.output.in_features * self.output.out_features
        self.output.weight.data.normal_(0, math.sqrt(2. / n))
        self.output.bias.data.zero_() 
                   
    def forward(self, text):
        text.cuda()
        h0 = ( Variable(torch.zeros(3, text.size(0), 128)).cuda(),  Variable(torch.zeros(3, text.size(0), 128)).cuda()) 

        lang_feature, hn = self.lang(text, h0 )
        lang_feature = lang_feature[:,-1,:]

        pred = self.output(lang_feature)
        return pred

class MultiModel(nn.Module):
    def __init__(self, preTrained='True'):
        super(MultiModel, self).__init__()

        # Output 
        self.output = nn.Sequential(nn.Linear(256, 256), 
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, 256), 
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(256, 2))

        self._initialize_weights()

        # Vision Model 
        #self.vision = vgg16(pretrained=vgg_pretrained, num_classes=128)
        self.vision = resnet34(pretrained=preTrained, num_classes=128)
        # LSTM Model(temporal)
        self.rnn  = nn.LSTM(128, 128, 2, batch_first=True)

        # Language Model
        self.lang = nn.LSTM(100, 128, 3, batch_first=True) 
                 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

            
    def forward(self, img, text):
        img.cuda()
        text.cuda() 

        dims = img.size()
        img_feature = self.vision(img.view(-1, dims[2], dims[3], dims[4]))
        if (dims[1] != 1):
            img_feature = img_feature.view(dims[0], dims[1], -1)

            h0 = ( Variable(torch.zeros(2, dims[0], 128)).cuda(),  Variable(torch.zeros(2, dims[0], 128)).cuda()) 
            img_feature, hn = self.rnn(img_feature, h0)
            img_feature = img_feature[:,-1, :]

        h0 = ( Variable(torch.zeros(3, text.size(0), 128)).cuda(),  Variable(torch.zeros(3, text.size(0), 128)).cuda()) 

        lang_feature, hn = self.lang(text, h0 )
        lang_feature = lang_feature[:,-1,:]

        multi_feature = torch.cat((img_feature,lang_feature), 1)

        pred = self.output(multi_feature)

        return pred


