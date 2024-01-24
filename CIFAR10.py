# Code for custom quantizer from here: 
# https://github.com/Xilinx/brevitas/blob/master/notebooks/03_anatomy_of_a_quantizer.ipynb
from brevitas.core.scaling import ParameterScaling
from brevitas.inject import ExtendedInjector
import torch

from brevitas.quant import SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst, Int8WeightPerTensorFixedPoint,Int8ActPerTensorFixedPoint

from torch import nn
from torch.nn import Module
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn import BatchNorm1d

import brevitas.nn as qnn

from operator import mul
from functools import reduce
from common import CommonActQuant
from common import CommonWeightQuant
from tensor_norm import TensorNorm


DROPOUT = 0.2

class Courbariaux_Binary_Net_MNIST(Module):
    # https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/Train-time/mnist.py
    # Adapted referencing:
    # https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/bnn_pynq/models/FC.py
    # For structural information. For like most of the code now
    def __init__(self):
        super(Courbariaux_Binary_Net_MNIST, self).__init__()

        # Hard coding the values
        in_bit_width = 1
        weight_bit_width = 1
        activation_bit_width = 1
        in_features = (32,32)
        out_features = [1024,1024,1024]
        num_classes = 10
        
        self.Features = ModuleList()
        # The first layer will quantise the network
        self.Features.append(
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = in_bit_width)
        )
        # Your first Drop out layer. This is similar to courbrax
        self.Features.append(
            nn.Dropout(p=DROPOUT)
        )
        in_features = reduce(mul,in_features)
        # print("In features:")
        # print(in_features)

        for out_features in out_features:
            self.Features.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant
                )    
            )
            in_features = out_features
            self.Features.append(
                nn.BatchNorm1d(num_features=in_features) # if only passing in one value network must be in eval mode
            )
            self.Features.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
            self.Features.append(
                qnn.QuantDropout(p=DROPOUT)
            )
        self.Features.append(
            qnn.QuantLinear(
                in_features=in_features,
                out_features=num_classes,
                bias = False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant,
                bit_width=weight_bit_width
            )
        )
        self.Features.append(
            TensorNorm()
        )

        for m in self.Features:
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data,-1,1)

    def clip_weights(self, min_val, max_val):
        for mod in self.Features:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device) # unsure how this effects the network
        for module in self.Features:
            x = module(x)
        return x


class Lenet5v1(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self):
        super(Lenet5v1, self).__init__()
       
        weight_bit_width = 8
        activation_bit_width = 8
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        self.Conv1 = qnn.QuantConv2d(3,6,5,weight_quant=CommonWeightQuant,weight_bit_width=weight_bit_width)
        self.Conv2 = qnn.QuantConv2d(6,16,5,weight_quant=CommonWeightQuant,weight_bit_width=weight_bit_width)
        #self.Pool = F.max_pool2d(4)
        self.Relu = qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width)
        # FC Layers
        in_features = 144
        out_features = [120,84,10]
        self.FCLayers = ModuleList()
        
        for out_features in out_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )

            in_features = out_features

            # Needed for FINN
            # self.FCLayers.append(
            #     qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width) # if only passing in one value network must be in eval mode
            # )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=in_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data,-1,1)

    def clip_weights(self, min_val, max_val):
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self,x):
        x = self.Quant(x) # Pool layers not compatable?
        x = F.max_pool2d(self.Relu(self.Conv1(x)),4)
        x = self.Relu(self.Conv2(x))
        x = x.reshape(x.shape[0], -1)
        # FC layers
        for module in self.FCLayers:
            x = self.Relu(module(x))
        # no soft max. Not compatable
        return x


class Lenet5v2(Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        activation_bit_width = 8
        weight_bit_width = 8
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)
        
        self.conv1 = qnn.QuantConv2d(3, 6, 5,weight_quant=CommonWeightQuant,weight_bit_width=weight_bit_width,bias = True)
        self.relu1 = qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width)
        self.conv2 = qnn.QuantConv2d(6, 16, 5,weight_quant=CommonWeightQuant,weight_bit_width=weight_bit_width)
        self.relu2 = qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width)
        self.fc1   = qnn.QuantLinear(144, 120, bias = True) # Since maxpool has been cut increase the input size for this layer
        self.relu3 = qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width)
        self.fc2   = qnn.QuantLinear(120, 84,bias = True)
        self.relu4 = qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width)
        self.relu5 = qnn.QuantReLU(act_quant=CommonActQuant,bit_width=activation_bit_width)
        self.fc3   = qnn.QuantLinear(84, 10, bias = True)
        self.batchNorm = nn.BatchNorm2d(num_features=6)

    def forward(self, x):
        x = self.Quant(x)
        x = self.relu1(self.batchNorm(self.conv1(x)))
        x = F.max_pool2d(x, 4)
        x = self.relu2(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.relu5(self.fc3(x))
        return x     


class Lenet5v3_7bit(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self):
        super(Lenet5v3_7bit, self).__init__()
       
        weight_bit_width = 7 # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = 7
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
            
        
        
        # FC Layers
        FC_features = [(144,120),(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 84
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x
class Lenet5v3_2bit(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self):
        super(Lenet5v3_2bit, self).__init__()
       
        weight_bit_width = 2 # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = 2
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
            
        
        
        # FC Layers
        FC_features = [(144,120),(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 84
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x

class Lenet5v4(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v4,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        CNV_layers = [(6,True),(16,True)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
        self.CNVLayers.append(qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
        
        
        # FC Layers
        FC_features = [(400,120),(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 84
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x

class Quant8BitFCMNISTOld(Module):
    # https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
    def __init__(self):
        super(Quant8BitFCMNISTOld, self).__init__()
        # Removed the Dropouts
        # Have attempted to get rid of a bug with synthesising by swapping common** with predefined Int8*** tensors. It should be noted that 8 bits did not work but a 3 bit system did when troubleshooting the problem
        # Hard coding the values
        # in_bit_width = 3
        weight_bit_width = 8  
        # activation_bit_width = 3
        in_features = (3,32,32)
        out_features = [32]
        num_classes = 10
        
        self.Features = ModuleList()
        in_features = reduce(mul,in_features)
        # print("In features:")
        # print(in_features)
        # Changed the network topology to be more inline with the original CIFAR-10 one which worked. Need to confirm that the network actually has 8 bit quantisations.
        for out_features in out_features:
            self.Features.append(
                qnn.QuantLinear(in_features,out_features,bias=True, weight_bit_width=8, act_bit_width=8)   
            )
            in_features = out_features
            self.Features.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            self.Features.append(
            qnn.QuantLinear(in_features,num_classes, bias=True, weight_bit_width=8, act_bit_width=8)
            )
            self.Features.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )

            # self.Features.append(
            #     nn.BatchNorm1d(num_features=in_features) # if only passing in one value network must be in eval mode
            # )
            # self.Features.append(
            #     qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)
            # )
        # self.Features.append(
        #     TensorNorm()
        # )



    def forward(self,x):
        x = x.view(x.shape[0], -1)
        #x = 2.0 * x - torch.tensor([1.0], device=x.device) # unsure how this effects the network
        for module in self.Features:
            x = module(x)
        return x



class Lenet5v9(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v9,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        # self.CNVLayers.append(
        #     qnn.QuantIdentity()
        # )
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=True))
            in_channels = out_channels
            # self.CNVLayers.append(
            #     nn.BatchNorm2d(num_features=out_channels)
            # )
            self.CNVLayers.append(
                qnn.QuantIdentity()
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
        
        
        # FC Layers
        FC_features = [(144,10)]#,(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(in_features,out_features,bias=True)   
            )
            # self.FCLayers.append(
            #     nn.BatchNorm1d(num_features=out_features)
            # )
            # self.FCLayers.append(
            #     qnn.QuantIdentity() # Needed to enable all the network to be loaded onto the FPGA
            # )
        # in_channels = 84
        # out_channels = 10
        # self.FCLayers.append(
        #        qnn.QuantLinear(in_channels,out_channels,bias=True, weight_bit_width=8, act_bit_width=8)   
        # )
        # self.FCLayers.append(
        #         qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        #     )
        # for m in self.FCLayers:
        #     if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
        #         torch.nn.init.uniform_(m.weight.data, -1, 1)
        # self.FCLayers.append(TensorNorm())

    # def clip_weights(self, min_val, max_val):
    #     for mod in self.CNVLayers:
    #         if isinstance(mod, qnn.QuantConv2d):
    #             mod.weight.data.clamp_(min_val, max_val)
    #     for mod in self.FCLayers:
    #         if isinstance(mod, qnn.QuantLinear):
    #             mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x

# from brevitas.quant import Sign
from brevitas.quant import SignedTernaryWeightPerTensorConst
from brevitas.quant import SignedTernaryActPerTensorConst
class Lenet5v8Tri(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v8Tri,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        self.CNVLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
        )
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant= SignedTernaryWeightPerTensorConst
                    ))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=out_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
        
        
        # FC Layers
        FC_features = [(144,120),(120,10)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(in_features,out_features,bias=False,weight_quant=SignedTernaryWeightPerTensorConst)   
            )
            self.FCLayers.append(
                nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width) # Needed to enable all the network to be loaded onto the FPGA
            )
        self.FCLayers.append(
            TensorNorm()
        )
        # in_channels = 84
        # out_channels = 10
        # self.FCLayers.append(
        #        qnn.QuantLinear(in_channels,out_channels,bias=True, weight_bit_width=8, act_bit_width=8)   
        # )
        # self.FCLayers.append(
        #         qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        #     )
        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device) # unsure how this effects the network
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x
class Lenet5v8Triv1(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v8Triv1,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        self.CNVLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=2)
        )
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_bit_width = 2,
                    weight_quant= CommonWeightQuant
                    ))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=out_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=2)
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
        
        
        # FC Layers
        FC_features = [(144,120),(120,10)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(in_features,out_features,bias=False,weight_bit_width = 2,
                    weight_quant= CommonWeightQuant)   
            )
            self.FCLayers.append(
                nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=2) # Needed to enable all the network to be loaded onto the FPGA
            )
        self.FCLayers.append(
            TensorNorm()
        )
        # in_channels = 84
        # out_channels = 10
        # self.FCLayers.append(
        #        qnn.QuantLinear(in_channels,out_channels,bias=True, weight_bit_width=8, act_bit_width=8)   
        # )
        # self.FCLayers.append(
        #         qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        #     )
        # for m in self.FCLayers:
        #     if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
        #         torch.nn.init.uniform_(m.weight.data, -1, 1)

    # def clip_weights(self, min_val, max_val):
    #     for mod in self.CNVLayers:
    #         if isinstance(mod, qnn.QuantConv2d):
    #             mod.weight.data.clamp_(min_val, max_val)
    #     for mod in self.FCLayers:
    #         if isinstance(mod, qnn.QuantLinear):
    #             mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x


class Lenet5v4withBias(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v4withBias,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits

        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=True,
                    weight_bit_width=weight_bit_width,
                    activation_bit_width = weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=out_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
        
        
        # FC Layers
        FC_features = [(400,120),(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=True,
                weight_bit_width=weight_bit_width, # might need to be bit_width instead of the bit_width stuff
                activation_bit_width = activation_bit_width
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
            )
            
        in_channels = 84
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=True,
                weight_bit_width=weight_bit_width,
                activation_bit_width = weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
        )

        self.FCLayers.append(
            TensorNorm()
        )
        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x
    
class Lenet5v6withoutBias(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v6withoutBias,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits

        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_bit_width =bits,
                    weight_quant=CommonWeightQuant,
                    ))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=out_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
        
        
        # FC Layers
        FC_features = [(144,120),(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_bit_width=weight_bit_width, # might need to be bit_width instead of the bit_width stuff
                weight_quant=CommonWeightQuant
                )    
            )
            self.FCLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            self.FCLayers.append(
                nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
            )
            
        in_channels = 84
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant
                )    
        )

        self.FCLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
        )

        self.FCLayers.append(
            TensorNorm()
        )
        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x
class Lenet5v7withBias(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v7withBias,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=True,
                    ))
            in_channels = out_channels
            # self.CNVLayers.append(
            #     qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            # )
            # self.CNVLayers.append(
            #     nn.BatchNorm2d(num_features=out_channels)
            # )
            self.CNVLayers.append(
                qnn.QuantIdentity()
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
        
        
        # FC Layers
        FC_features = [(144,10)]#,(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(in_features,out_features,bias=True, weight_bit_width=8, act_bit_width=8)   
            )
            self.FCLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
        # in_channels = 84
        # out_channels = 10
        # self.FCLayers.append(
        #        qnn.QuantLinear(in_channels,out_channels,bias=True, weight_bit_width=8, act_bit_width=8)   
        # )
        # self.FCLayers.append(
        #         qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        #     )
        # for m in self.FCLayers:
        #     if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
        #         torch.nn.init.uniform_(m.weight.data, -1, 1)

    # def clip_weights(self, min_val, max_val):
    #     for mod in self.CNVLayers:
    #         if isinstance(mod, qnn.QuantConv2d):
    #             mod.weight.data.clamp_(min_val, max_val)
    #     for mod in self.FCLayers:
    #         if isinstance(mod, qnn.QuantLinear):
    #             mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x

class Lenet5v6withBias(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v6withoutBias,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits

        # Conv. Layers
        CNV_layers = [(6,True),(16,False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    bit_width = bits
                    ))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=out_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
            )
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=4)
                )
        
        
        # FC Layers
        FC_features = [(144,120),(120,84)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_bit_width=weight_bit_width, # might need to be bit_width instead of the bit_width stuff
                activation_bit_width = activation_bit_width
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
            )
            
        in_channels = 84
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_bit_width=weight_bit_width,
                activation_bit_width = weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=weight_bit_width)
        )

        self.FCLayers.append(
            TensorNorm()
        )
        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x

class Lenet5v5(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v5,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        in_channels = 3
        CNV_layers = [(6*in_channels,True),(16*in_channels,True)]
        self.CNVLayers = ModuleList()
        
        
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
        self.CNVLayers.append(qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
        
        
        # FC Layers
        FC_features = [(16*5*5*3,120*3),(120*3,84*3)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 84*3
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        )
        self.FCLayers.append(
            nn.BatchNorm1d(num_features=out_channels)
        )
        self.FCLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x
    
class Lenet5v5withBias(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self,bits=2):
        super(Lenet5v5withBias,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        in_channels = 3
        CNV_layers = [(6*in_channels,True),(16*in_channels,True)]
        self.CNVLayers = ModuleList()
        
        
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=True,
                    weight_bit_width=weight_bit_width,
                    activation_bit_width = activation_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            # self.CNVLayers.append(
            #     nn.BatchNorm2d(num_features=in_channels)
            # )
            # self.CNVLayers.append(
            #     qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
        self.CNVLayers.append(qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
        
        
        # FC Layers
        FC_features = [(16*5*5*3,120*3),(120*3,84*3)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=True,
                weight_bit_width=weight_bit_width,
                activation_bit_width=activation_bit_width                
                )    
            )
            self.FCLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            # self.FCLayers.append(
            #      nn.BatchNorm1d(num_features=out_features)
            # )
            # self.FCLayers.append(
            #     qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            # )
        in_channels = 84*3
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=True,
                weight_bit_width=weight_bit_width,
                activation_bit_width = activation_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        )
        # self.FCLayers.append(
        #     nn.BatchNorm1d(num_features=out_channels)
        # )
        # self.FCLayers.append(
        #     qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
        # )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x
from brevitas.core.restrict_val import RestrictValueType

class FINNv2(Module):
    def __init__(self,bits = 2):
        super(FINNv2, self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=8,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO)

        # Conv. Layers
        CNV_layers = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels,eps=1e-4)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
            
        
        
        # FC Layers
        FC_features = [(256, 512), (512, 512)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 512
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x


class FINNv1_binary(Module):
    # https://www.kaggle.com/code/xcwang21/cifar10-with-mlp-lenet-resnet-for-beginners
    def __init__(self):
        super(FINNv1_binary, self).__init__()
       
        weight_bit_width = 1 # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = 1
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        CNV_layers = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels,eps=1e-4)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
            
        
        
        # FC Layers
        FC_features = [(256, 512), (512, 512)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 512
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x




class BNN_CONV(Module):
    # Sourced from: https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/vgg_cifar10_binary.py
    # Modified for FINN by: Julien
    # Modifications:
    # 1. Removed HardTanh not compatiable with FINN
    # 2. Removed logsoftmax, finn has its own K selector
    # 4. Replaced IO between conv end and start FC layer  (TO DO) - fIXED
    # 5. Added in the Quantisation controls for each of the layers
    # 6. Added inital Quantisation layer at start of the network
    # 7. set number of classes to 10
    # 8. moved max pool to be after batch norm and quant identity
    # 9. removed padding
    # NOTE: the network here is unable to be synthesised due to resource utilsation error. I am unable to spend the time to debug such an error.
    def __init__(self, num_classes=10):
        super(BNN_CONV, self).__init__()
        self.infl_ratio=1 # changed from 3
        activation_bit_width = 1
        weight_bit_width = 1
        self.features = nn.Sequential(
                   
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width),
            qnn.QuantConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1,
                      bias=False, weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            nn.BatchNorm2d(128*self.infl_ratio),
            #nn.Hardtanh(inplace=True),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            qnn.QuantConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, bias=False, 
                           weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            
            nn.BatchNorm2d(128*self.infl_ratio),
            #nn.Hardtanh(inplace=True),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            qnn.QuantConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, bias=False, 
                           weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            
            #nn.Hardtanh(inplace=True),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            nn.BatchNorm2d(256*self.infl_ratio),
            
            qnn.QuantConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3,bias=False,
                           weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            
            nn.BatchNorm2d(256*self.infl_ratio),
            #nn.Hardtanh(inplace=True),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            nn.MaxPool2d(kernel_size=2, stride=2),
            qnn.QuantConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3,bias=False, 
                           weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            
            #nn.Hardtanh(inplace=True),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            nn.BatchNorm2d(512*self.infl_ratio),
            qnn.QuantConv2d(512*self.infl_ratio, 512, kernel_size=3, bias=False,
                           weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            
            nn.BatchNorm2d(512),
            
            #nn.Hardtanh(inplace=True)
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(512),
            # REQUIRED TO PROVED MULTTHRESOLD AFTER MAXPOOL
            #qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
        )
        self.classifier = nn.Sequential(
            qnn.QuantLinear(512, 1024, bias=False
                           , weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            nn.BatchNorm1d(1024),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            #nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            qnn.QuantLinear(1024, 1024, bias=False
                           , weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            nn.BatchNorm1d(1024),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            #nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            qnn.QuantLinear(1024, num_classes, bias=False
                           , weight_quant=CommonWeightQuant, weight_bit_width=weight_bit_width),
            nn.BatchNorm1d(num_classes),
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width),
            #nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

class VGG11(Module):
    def __init__(self,bits):
        super(VGG11, self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)
        

        # Conv. Layers
        CNV_layers = [(64, True), (128, True), (256, False), (256, True), (512, False), (512, True),(512,False),(512,True)] # Original
        CNV_layers = [(64, True), (64, True), (64, False), (64, True), (128, False), (128, True),(128,False),(128,True)]
        self.CNVLayers = ModuleList()
        
        in_channels = 3
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                    padding=(1,1)))
            in_channels = out_channels
            self.CNVLayers.append(
                qnn.QuantReLU(inplace=True)
            )
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels,eps=1e-4)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
            
        if pooling:
            # if pooling was true for the last layer you need to requantise it for finn to work
            # self.CNVLayers.append(
            #     nn.BatchNorm2d(num_features=in_channels)
            # )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width = activation_bit_width)
            )
        
        
        # FC Layers
        #FC_features = [(512, 4096), (4096, 4096),(4096,10)] # Original system
        # From Jerret's work
        FC_features = [(32,10)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
       
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x


class Calik(Module):
 
    def __init__(self,bits=2):
        super(Calik,self).__init__()
       
        weight_bit_width = bits # note that bigger values like 8 will not fit on the fpga 283 mem cells used (280 max aval.)
        activation_bit_width = bits
        # Quantise the inputs
        self.Quant = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = activation_bit_width)

        # Conv. Layers
        in_channels = 3
        CNV_layers = [(24*3,True),(12*3,True),(6*3,True),(3*3,True)]
        self.CNVLayers = ModuleList()
        
        
        
        for out_channels, pooling in CNV_layers:
            self.CNVLayers.append(
                qnn.QuantConv2d(
                    kernel_size=5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                    padding_type='same'))
            in_channels = out_channels
            self.CNVLayers.append(
                nn.BatchNorm2d(num_features=in_channels)
            )
            self.CNVLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
            if pooling:
                self.CNVLayers.append(
                    nn.MaxPool2d(kernel_size=2)
                )
        self.CNVLayers.append(qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=activation_bit_width))
        
        
        # # FC Layers
        FC_features = [(36,36)]
        self.FCLayers = ModuleList()
        
        for in_features, out_features in FC_features:
            self.FCLayers.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
            )
            self.FCLayers.append(
                qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
            )
            self.FCLayers.append(
                 nn.BatchNorm1d(num_features=out_features)
            )
            self.FCLayers.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
            )
        in_channels = 36
        out_channels = 10
        self.FCLayers.append(
              qnn.QuantLinear(
                in_features=in_channels,
                out_features=out_channels,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width # might need to be bit_width instead of the bit_width stuff
                )    
        )
        self.FCLayers.append(
            qnn.QuantReLU() # Needed to enable all the network to be loaded onto the FPGA
        )
        self.FCLayers.append(
                nn.BatchNorm1d(num_features=out_channels)
        )
        self.FCLayers.append(
            qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
        )
        self.FCLayers.append(
            TensorNorm()
        )

        for m in self.FCLayers:
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.CNVLayers:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.FCLayers:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)



    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.Quant(x)
        for mod in self.CNVLayers:
            x = mod(x)
            # print(type(mod))
            # print(x.shape)
        x = x.view(x.shape[0], -1)
        for mod in self.FCLayers:
            x = mod(x)
        return x