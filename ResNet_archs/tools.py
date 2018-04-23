###################################################################################################################
#
#  The scripts is employed to convert the original network  to only containing conv layer(from res1 to res5) 
#  which is removing the last fully connected layer
#  Author: Rosun
#  Date: 2018/04/21
#
##################################################################################################################

from collections import OrderedDict
import os
import numpy as np
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152, netParams
import torch

def convert_state_dict_remove_fc(state_dict):
    """
    Remove state_dict 'fc.weight' and 'fc.bias' of the ResNets,
    Args:   
        state_dict is the loaded DataParallel model_state
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        if k.find("fc")!= -1:
            print(k)
        else:
            name = k
            #name = k[7:] # remove the prefix module.
            # My heart is borken, the pytorch have no ability to do with the problem.
            state_dict_new[name] = v
    return state_dict_new

if __name__=="__main__":
    model_name='resnet152'
    
    if model_name== 'resnet18':
        model= resnet18(pretrained=True, fc_flag=True)
        torch.save(model.state_dict(), 'pretrainedfile/'+'resnet18.pth')
    elif model_name== 'resnet34':
        model= resnet34(pretrained=True, fc_flag=True)
        torch.save(model.state_dict(), 'pretrainedfile/'+'resnet34.pth')

    elif model_name== 'resnet50':
        model= resnet50(pretrained=True, fc_flag=True)
        torch.save(model.state_dict(), 'pretrainedfile/'+'resnet50.pth')

    elif model_name== 'resnet101':
        model= resnet101(pretrained=True, fc_flag=True)
        torch.save(model.state_dict(), 'pretrainedfile/'+'resnet101.pth')

    elif model_name== 'resnet152':
        model= resnet152(pretrained=True, fc_flag=True)
        torch.save(model.state_dict(), 'pretrainedfile/'+'resnet152.pth')
    else:
        print("model name is error")

    checkpoint = torch.load('pretrainedfile/'+model_name+'.pth')
    model.load_state_dict(checkpoint)
    print("The original {} model parameter: ".format(model_name), netParams(model))




    if model_name== 'resnet18':
        model_nofc= resnet18(pretrained=None, fc_flag=False)
    elif model_name== 'resnet34':
        model_nofc= resnet34(pretrained=None, fc_flag=False)

    elif model_name== 'resnet50':
        model_nofc= resnet50(pretrained=None, fc_flag=False)

    elif model_name== 'resnet101':
        model_nofc= resnet101(pretrained=None, fc_flag=False)
   
    elif model_name== 'resnet152':
        model_nofc= resnet152(pretrained=None, fc_flag=False)
    else:
        print("model name is error")


    checkpoint_nofc = convert_state_dict_remove_fc(checkpoint)
    print("{} model without fc layer, model parameter: ".format(model_name), netParams(model_nofc))
    
    model_nofc.load_state_dict(checkpoint_nofc)
    torch.save(model_nofc.state_dict(), 'pretrainedfile/'+model_name+'_nofc.pth')

    print('the parameters diff: ', netParams(model)-netParams(model_nofc))

