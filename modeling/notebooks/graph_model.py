#!/usr/bin/env python
# coding: utf-8

# # Model Graphing
# Graph CNN's layout using [waleedka/hiddenlayer](https://github.com/waleedka/hiddenlayer)  
# 
# To use `hiddenlayer`, you must first install [Graphviz](https://graphviz.gitlab.io/download/), and then the [graphviz](https://github.com/xflr6/graphviz) python wrapper package. If you have conda available it's easiest to install both together with `conda install graphviz python-graphviz`. Also, unless you compile from source, `hiddenlayer` will only work with recent versions of pytorch (>1.3) and [does not work for efficientnet models](https://github.com/waleedka/hiddenlayer/issues/49) due to a limitation with `torch.onnx`  

# ***
# # Setup

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys, os
sys.path.append(os.path.expanduser('~/mount_sinai_health_hackathon_ekg_img/'))
from common_code import *
get_ipython().run_line_magic('matplotlib', 'inline')

import timm # pretrained models from rwightman/pytorch-image-models
import torchvision.models as models # pretrained models from pytorch

import hiddenlayer as hl # for graphing the model


# In[ ]:


# Models to choose from ['tf_efficientnet_b7_ns', resnet, alexnet, vgg, squeezenet, densenet] # inception
# model_name = 'tf_efficientnet_b7_ns'
model_name = 'resnet'

# Number of classes in the dataset
n_classes = 9

feature_extract = True
use_pretrained=False


# In[ ]:


output_path = f'../output/{model_name}'
models_path = f'../models/{model_name}'


# In[ ]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# ***
# # Create the Model

# In[ ]:


def initialize_model(model_name, n_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'tf_efficientnet_b7_ns':
        ''' EfficientNet-B7 NoisyStudent. Tensorflow compatible variant
            Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
        '''
        model_ft = timm.create_model('tf_efficientnet_b7_ns', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, n_classes)
        input_size = model_ft.default_cfg['input_size'][1]

    elif model_name == 'resnet':
        ''' Resnet101
        '''
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
        input_size = 224

    elif model_name == 'alexnet':
        ''' Alexnet
        '''
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,n_classes)
        input_size = 224

    elif model_name == 'vgg':
        ''' VGG11_bn
        '''
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,n_classes)
        input_size = 224

    elif model_name == 'squeezenet':
        ''' Squeezenet
        '''
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.n_classes = n_classes
        input_size = 224

    elif model_name == 'densenet':
        ''' Densenet
        '''
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, n_classes)
        input_size = 224

#    elif model_name == 'inception':
#        ''' Inception v3
#        Be careful, expects (299,299) sized images and has auxiliary output
#        '''
#        model_ft = models.inception_v3(pretrained=use_pretrained)
#        set_parameter_requires_grad(model_ft, feature_extract)
#        # Handle the auxilary net
#        num_ftrs = model_ft.AuxLogits.fc.in_features
#        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, n_classes)
#        # Handle the primary net
#        num_ftrs = model_ft.fc.in_features
#        model_ft.fc = nn.Linear(num_ftrs,n_classes)
#        input_size = 299

    else:
        raise ValueError(f'Invalid model_name = {model_name}')
        exit()

    return model_ft, input_size


# In[ ]:


model, input_size = initialize_model(model_name, n_classes, feature_extract, use_pretrained)


# In[ ]:


print(f'input_size = {input_size}')


# ***
# # Load model from disk

# In[ ]:


dfp_train_results = load_dfp(models_path, 'train_results', tag='', cols_bool=['saved_model'],
                             cols_float=['train_loss','val_loss','best_val_loss','delta_per_best','elapsed_time','epoch_time'])

best_epoch = dfp_train_results.iloc[dfp_train_results['val_loss'].idxmin()]['epoch']
load_model(model, 'cpu', best_epoch, model_name, models_path)


# ***
# # Plot the Model

# In[ ]:


# model.eval()


# In[ ]:


hl_graph = hl.build_graph(model, torch.zeros([1, 3, 600, 600]))


# In[ ]:


hl_graph


# ***
# # Dev

# In[ ]:


from common_code import *


# In[ ]:





# In[ ]:





# In[ ]:




