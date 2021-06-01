import paddle
import math
from paddleseg.utils import logger
import paddle.nn as nn

def get_optimizer_lr(model):
    backbone_params = nn.ParameterList()
    other_params = nn.ParameterList()
    for name, param in model.named_parameters():
        if param.stop_gradient:
            other_params.append(param)
            continue

        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            backbone_params.append(param)
            
    return backbone_params, other_params
   

    