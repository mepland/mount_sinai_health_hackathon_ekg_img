#!/usr/bin/env python
# coding: utf-8

# # Model Training
# For reference see [Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)  
# For additional pretrained models see [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models), in particular the [README model list](https://github.com/rwightman/pytorch-image-models#ported-weights), [EfficientNet generator](https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py), and [pretrained weights](https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-weights)  
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip');
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt');
# ***
# # Setup

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys, os
sys.path.append(os.path.expanduser('~/mount_sinai_health_hackathon_ekg_img/'))
from common_code import *
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import confusion_matrix

import timm # pretrained models from rwightman/pytorch-image-models
import torchvision.models as models
from torchvision import datasets, transforms
# from sklearn.metrics import confusion_matrix


# In[2]:


Dx_classes = {
'Normal': 'Normal sinus rhythm',
'AF': 'Atrial fibrillation',
'I-AVB': 'Airst-degree atrioventricular block',
'LBBB': 'Left bundle branch block',
'PAC': 'Premature atrial complex',
'PVC': 'Premature ventricular complex',
'RBBB': 'Right bundle branch block',
'STD': 'ST-segment depression',
'STE': 'ST-segment elevation',
}


# In[3]:


# Check if gpu support is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')


# In[4]:


# Models to choose from ['tf_efficientnet_b7_ns', resnet, alexnet, vgg, squeezenet, densenet] # inception
# model_name = 'resnet'
model_name = 'tf_efficientnet_b7_ns'

# Number of classes in the dataset
n_classes = len(Dx_classes.keys())

# Batch size for training (change depending on how much memory you have, and how large the model is)
batch_size = 40 # TODO 32 was working with 2.8 GB memory left, 40 works with around 1 GB. 64 didn't work. These images are only a few kb so I'm not sure what's driving that scaling...

# Flag for feature extraction. When True only update the reshaped layer params, when False train the whole model from scratch.
# Should probably remain True.
feature_extract = True

# use pretrained model, should probably remain True.
use_pretrained=True

# path to data dir
data_path = os.path.expanduser('~/mount_sinai_health_hackathon_ekg_img/data')

# resolution of preprocessed images
# im_res = 800
im_res = 600
# im_res=224


# In[5]:


output_path = f'../output/{model_name}'
models_path = f'../models/{model_name}'


# In[6]:


# test_mem()


# ***
# ### Helper Functions

# In[7]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# ***
# # Create the Model

# In[8]:


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


# In[9]:


model, input_size = initialize_model(model_name, n_classes, feature_extract, use_pretrained)


# In[10]:


print(f'input_size = {input_size}')


# In[11]:


if im_res < input_size:
    raise ValueError(f'Warning, trying to run a model with an input size of {input_size}x{input_size} on images that are only {im_res}x{im_res}! You can procede at your own risk, ie upscaling, better to fix one or the other size though!')


# In[12]:


loss_fn = nn.CrossEntropyLoss()

model.to(device);


# In[13]:


# Gather the parameters to be optimized/updated in this run.
# If we are finetuning we will be updating all parameters
# However, if we are using the feature extract method, we will only update the parameters that we have just initialized,
# i.e. the parameters with requires_grad is True.

params_to_update = model.parameters()
print('Params to learn:')
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print(name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print(name)


# In[14]:


# Setup optimizer
# optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(params_to_update, weight_decay=1e-5)


# ***
# # Load Data

# ### Compute Normalization Factors
dl_unnormalized = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/train',
                            transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize(input_size), transforms.ToTensor()])),
    batch_size=batch_size, shuffle=False, num_workers=8
)

pop_mean, pop_std0 = compute_channel_norms(dl_unnormalized)

print(f"pop_mean = [{', '.join([f'{v:.8f}' for v in pop_mean])}]")
print(f"pop_std0 = [{', '.join([f'{v:.8f}' for v in pop_std0])}]")
# In[15]:


# use normalization results computed earlier
if input_size == 224:
    pop_mean = np.array([])
    pop_std0 = np.array([])
elif input_size == 600:
    pop_mean = np.array([0.95724732, 0.95724732, 0.95724732])
    pop_std0 = np.array([0.08290727, 0.08290727, 0.08290727])
elif input_size == 800:
    pop_mean = np.array([])
    pop_std0 = np.array([])
else:
    raise ValueError(f'No precomputed mean, std avalaible for input_size = {input_size}')

# use normalization results used when training the model, only works for timm models. Should probably only use for color images
pop_mean = np.array(model.default_cfg['mean'])
pop_std0 = np.array(model.default_cfg['std'])
# In[16]:


print(f"pop_mean = [{', '.join([f'{v:.8f}' for v in pop_mean])}]")
print(f"pop_std0 = [{', '.join([f'{v:.8f}' for v in pop_std0])}]")


# ### Actually Load Data

# In[17]:


# need to fake 3 channels r = b = g with Grayscale to use pretrained networks
transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize(pop_mean, pop_std0)])

ds_train = tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/train', transform=transform)
ds_val = tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/val', transform=transform)


# In[18]:


class_to_idx = {}
for k,v in ds_train.class_to_idx.items():
    class_to_idx[k] = v
class_to_idx = OrderedDict(sorted(class_to_idx.items(), key=lambda x: x))
idx_to_class = OrderedDict([[v,k] for k,v in class_to_idx.items()])


# In[19]:


dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)


# In[20]:


# ds_test = tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/test', transform=transform)
# dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=8)


# In[21]:


# test_mem()


# ***
# # Train

# In[22]:


# test_mem()


# In[ ]:


dfp_train_results = train_model(dl_train, dl_val,
model, optimizer, loss_fn, device,
model_name=model_name, models_path=models_path,
max_epochs=200, max_time_min=500,
do_es=True, es_min_val_per_improvement=0.0005, es_epochs=10,
do_decay_lr=True, initial_lr=0.001, lr_epoch_period=25, lr_n_period_cap=4,
save_model_inhibit=20, # don't save anything out for the first save_model_inhibit = 10 epochs, set to -1 to start saving immediately
n_models_on_disk=3, # keep the last n_models_on_disk models on disk, set to -1 to keep all
)


# ***
# # Eval

# In[ ]:


dfp_train_results = load_dfp(models_path, 'train_results', tag='', cols_bool=['saved_model'],
                             cols_float=['train_loss','val_loss','best_val_loss','delta_per_best','elapsed_time','epoch_time'])


# In[ ]:


dfp_train_results

plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='', inline=True,
                   ann_text_std_add=None,
                   y_axis_params={'log': False},
                   loss_cols=['val_loss'],
                  )plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='', inline=True,
                   ann_text_std_add=None,
                   y_axis_params={'log': False},
                   loss_cols=['train_loss'],
                  )plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='', inline=True,
                   ann_text_std_add=None,
                   y_axis_params={'log': False},
                   loss_cols=['train_loss', 'val_loss'],
                  )
# ### Load model from disk

# In[ ]:


best_epoch = dfp_train_results.iloc[dfp_train_results['val_loss'].idxmin()]['epoch']
load_model(model, device, best_epoch, model_name, models_path)


# ### Make Predictions

# In[ ]:


labels, preds = get_preds(dl_val, model, device)


# In[ ]:


# labels


# In[ ]:


# preds


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(y_true=labels, y_pred=preds, labels=[class_to_idx[k] for k in Dx_classes.keys()])


# In[ ]:


plot_confusion_matrix(conf_matrix, label_names=Dx_classes.keys(),
                      m_path=output_path, tag='_val', inline=False,
                      ann_text_std_add=None,
                      normalize=True,
                     )


# In[ ]:


plot_confusion_matrix(conf_matrix, label_names=Dx_classes.keys(),
                      m_path=output_path, tag='_raw_val', inline=False,
                      ann_text_std_add=None,
                      normalize=False,
                     )


# ***
# # Dev

# In[ ]:


from common_code import *


# In[ ]:


# test_mem()


# In[ ]:





# In[ ]:




