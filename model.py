import torch
#import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url
#from torch.nn.utils.rnn import pack_padded_sequence

class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().detach()
    def close(self):
        self.hook.remove()

# This is a simple model that returns that last fully-connected layer of a Resnet 18 CNN      
class EncoderCNN(resnet.ResNet):
    def __init__(self):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        state_dict = load_state_dict_from_url(resnet.model_urls['resnet18'], progress=True)
        self.load_state_dict(state_dict)
        self.activation = SaveFeatures(list(self.children())[-1])
     
    def __call__(self, inputs):
        super().__call__(inputs)
        return self.activation.features