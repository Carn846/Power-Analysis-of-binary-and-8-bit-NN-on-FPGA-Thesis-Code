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

from brevitas.quant import SignedTernaryActPerTensorConst,SignedTernaryWeightPerTensorConst

class QuantTrinaryFCMNIST(Module):
    # https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/Train-time/mnist.py
    # Adapted referencing:
    # https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/bnn_pynq/models/FC.py
    # For structural information. For like most of the code now
    def __init__(self):
        super(QuantTrinaryFCMNIST, self).__init__()
        # Removed the Dropouts
        # Hard coding the values
        in_bit_width = 2
        weight_bit_width = 2
        activation_bit_width = 2
        in_features = (28,28)
        out_features = [1024]
        num_classes = 10
        
        self.Features = ModuleList()
        # The first layer will quantise the network
        self.Features.append(
            qnn.QuantIdentity(act_quant=CommonActQuant, bit_width = in_bit_width)
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
                weight_bit_width=weight_bit_width, # might be wrong order here
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
                nn.BatchNorm1d(num_features=10) # if only passing in one value network must be in eval mode
            )
        self.Features.append(
                qnn.QuantIdentity(act_quant=CommonActQuant,bit_width=activation_bit_width)
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
from brevitas.quant import Int8WeightPerTensorFixedPoint

class FC8bit(Module):
    # https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
    def __init__(self):
        super(FC8bit, self).__init__()
        # Removed the Dropouts
        # Have attempted to get rid of a bug with synthesising by swapping common** with predefined Int8*** tensors. It should be noted that 8 bits did not work but a 3 bit system did when troubleshooting the problem
        # Hard coding the values
        # in_bit_width = 3
        weight_bit_width = 8  
        # activation_bit_width = 3
        in_features = (28,28)
        out_features = [32]
        num_classes = 10
        
        self.Features = ModuleList()
        in_features = reduce(mul,in_features)
        # self.quant = qnn.QuantIdentity()

        for out_features in out_features:
            self.Features.append(
                qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=True
                )   
            )
            in_features = out_features
            
            # self.Features.append(
            #     nn.BatchNorm1d(num_features=in_features) # if only passing in one value network must be in eval mode
            # )
            self.Features.append(
                qnn.QuantIdentity() # Needed to enable all the network to be loaded onto the FPGA
            )
        self.Features.append(
            qnn.QuantLinear(
                in_features=in_features,
                out_features=10,
                bias=True
                )   
        )
        self.Features.append(
                qnn.QuantIdentity() # Needed to enable all the network to be loaded onto the FPGA
            )

        # self.Features.append(
        #     TensorNorm()
        # )




    def forward(self,x):
        x = x.view(x.shape[0], -1)
        # x = self.quant(x)
        #x = 2.0 * x - torch.tensor([1.0], device=x.device) # unsure how this effects the network
        for module in self.Features:
            x = module(x)
        return x


