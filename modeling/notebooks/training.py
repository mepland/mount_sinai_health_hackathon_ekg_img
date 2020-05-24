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

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys, os
sys.path.append(os.path.expanduser('~/mount_sinai_health_hackathon_ekg_img/'))
sys.path.append(os.path.expanduser('~/mount_sinai_health_hackathon_ekg_img/modeling'))
from common_code import *
get_ipython().run_line_magic('matplotlib', 'inline')

import timm # pretrained models from rwightman/pytorch-image-models
import torchvision.models as models # pretrained models from pytorch
from mobilenetv3 import MobileNetV3 # mobilenetv3 definition

from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix


# In[ ]:


# Check if gpu support is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')


# In[ ]:


Dx_classes = {
'Normal': 'Normal sinus rhythm',
'AF': 'Atrial fibrillation',
'I-AVB': 'First-degree atrioventricular block',
'LBBB': 'Left bundle branch block',
'PAC': 'Premature atrial complex',
'PVC': 'Premature ventricular complex',
'RBBB': 'Right bundle branch block',
'STD': 'ST-segment depression',
'STE': 'ST-segment elevation',
}


# In[ ]:


# Models to choose from [mobilenetv3_small_dev, tf_efficientnet_b7_ns, tf_efficientnet_b6_ns, resnet, alexnet, vgg, squeezenet, densenet]

model_name = 'mobilenetv3_small_dev' # Any dimension, tested at 600
# model_name = 'tf_efficientnet_b7_ns' # 600
# model_name = 'tf_efficientnet_b6_ns' # 528
# model_name = 'resnet' # 224

# resume training on a prior model
resume_training = False

# Batch size for training (change depending on how much memory you have, and how large the model is)
batch_size = 32 # 40

# balance classes by reweighting in loss function
balance_class_weights = True

# use pretrained model, should probably remain True.
use_pretrained=False # True

# Flag for feature extraction. When True only update the reshaped layer params, when False train the whole model from scratch. Should probably remain True.
feature_extract = False

# Number of classes in the dataset
n_classes = len(Dx_classes.keys())

# path to data dir
data_path = os.path.expanduser('~/mount_sinai_health_hackathon_ekg_img/data')

# channels of preprocessed images to use, 1 or 3
im_channels=1

# resolution of preprocessed images
im_res = 800
# im_res = 600


# In[ ]:


if '_dev' in model_name:
    if use_pretrained:
        raise ValueError('Can not use a pretrained dev model!')
    if feature_extract and not use_pretrained:
        raise ValueError('Can not update last feature layer for a non pretrained model!')
else:
    if im_channels != 3:
        raise ValueError('Must have 3 initial color channels for most standard models!')


# In[ ]:


output_path = f'../output/{model_name}'
models_path = f'../models/{model_name}'


# In[ ]:


# test_mem()


# ***
# ### Make Training Deterministic
# See [Pytorch's Docs on Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)  

# In[ ]:


rnd_seed=42
np.random.seed(rnd_seed)
torch.manual_seed(rnd_seed+1)
if str(device) == 'cuda':
    torch.cuda.manual_seed(rnd_seed+2)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ***
# ### Helper Functions

# In[ ]:


# Gathers the parameters to be optimized/updated in training. If we are finetuning we will be updating all parameters
# However, if we are using the feature extract method, we will only update the parameters that we have just initialized,
# i.e. the parameters with requires_grad is True.

def get_parameter_requires_grad(model, feature_extracting, print_not_feature_extracting=False):
    params_to_update = model.parameters()
    if feature_extracting:
        print('Params to learn:')
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(name)
    else:
        if print_not_feature_extracting:
            print('Params to learn:')
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                if print_not_feature_extracting:
                    print(name)
    return params_to_update


# In[ ]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# In[ ]:


def mobilenetv3_small_dev(cfgs=None, **kwargs):
    if cfgs is None:
        # use original cfgs for mobilentv3 small
         cfgs = [
            # k, t, c, SE, HS, s
            [3,    1,  16, 1, 0, 2],
            [3,  4.5,  24, 0, 0, 2],
            [3, 3.67,  24, 0, 0, 1],
            [5,    4,  40, 1, 1, 2],
            [5,    6,  40, 1, 1, 1],
            [5,    6,  40, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    6,  96, 1, 1, 2],
            [5,    6,  96, 1, 1, 1],
            [5,    6,  96, 1, 1, 1],
        ]
    return MobileNetV3(cfgs, mode='small', **kwargs)


# ***
# # Create the Model

# In[ ]:


def initialize_model(model_name, n_classes, feature_extract, use_pretrained=True):
    model = None
    input_size = 0

    if model_name == 'mobilenetv3_small_dev':
        ''' (Modified) MobileNetV3 Small
            Paper: Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)
        '''
        cfgs = [
            # k, t, c, SE, HS, s
            [3,    1,  4, 1, 0, 2],
            [3,  4.5,  8, 0, 0, 2],
            [3, 3.67,  8, 0, 0, 1],
            [5,    4,  16, 1, 1, 2],
            [5,    6,  16, 1, 1, 1],
            [5,    6,  30, 1, 1, 1],
            [5,    3,  30, 1, 1, 1],
        ]
        model = mobilenetv3_small_dev(cfgs, num_classes=n_classes, im_channels=im_channels)
        input_size = im_res
    elif model_name == 'tf_efficientnet_b7_ns':
        ''' EfficientNet-B7 NoisyStudent. Tensorflow compatible variant
            Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
        '''
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)
        input_size = model.default_cfg['input_size'][1]
    elif model_name == 'tf_efficientnet_b6_ns':
        ''' EfficientNet-B6 NoisyStudent. Tensorflow compatible variant
            Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
        '''
        model = timm.create_model('tf_efficientnet_b6_ns', pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)
        input_size = model.default_cfg['input_size'][1]
    elif model_name == 'resnet':
        ''' Resnet101
        '''
        model = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        input_size = 224
    elif model_name == 'alexnet':
        ''' Alexnet
        '''
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,n_classes)
        input_size = 224
    elif model_name == 'vgg':
        ''' VGG11_bn
        '''
        model = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,n_classes)
        input_size = 224
    elif model_name == 'squeezenet':
        ''' Squeezenet
        '''
        model = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))
        model.n_classes = n_classes
        input_size = 224
    elif model_name == 'densenet':
        ''' Densenet
        '''
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)
        input_size = 224
    else:
        raise ValueError(f'Invalid model_name = {model_name}')

    return model, input_size


# In[ ]:


model, input_size = initialize_model(model_name, n_classes, feature_extract, use_pretrained)


# In[ ]:


print(f'input_size = {input_size}')


# In[ ]:


if im_res < input_size:
    raise ValueError(f'Warning, trying to run a model with an input size of {input_size}x{input_size} on images that are only {im_res}x{im_res}! You can proceed at your own risk, ie upscaling, better to fix one or the other size though!')


# In[ ]:


params_to_update = get_parameter_requires_grad(model, feature_extract, print_not_feature_extracting=False)


# In[ ]:


# Setup optimizer
optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(params_to_update, weight_decay=1e-5)


# ***
# # Load Previously Trained Model
# To continue the training in another session  

# In[ ]:


if resume_training:
    print('Resuming Training!')
    dfp_train_results_prior = load_dfp(models_path, 'train_results', tag='', cols_bool=['saved_model'],
                                       cols_float=['train_loss','val_loss','best_val_loss','delta_per_best','elapsed_time','epoch_time'])

    best_epoch = dfp_train_results_prior.iloc[dfp_train_results_prior['val_loss'].idxmin()]['epoch']
    load_model(model, device, best_epoch, model_name, models_path)
else:
    dfp_train_results_prior = None
    model.to(device);


# ***
# # Load Data

# ### Compute Normalization Factors
dl_unnormalized = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/train',
                            transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize(input_size), transforms.ToTensor()])),
    batch_size=batch_size, shuffle=False, num_workers=8
)

norm_mean, norm_std0 = compute_channel_norms(dl_unnormalized)

print(f"norm_mean = [{', '.join([f'{v:.8f}' for v in norm_mean])}]")
print(f"norm_std0 = [{', '.join([f'{v:.8f}' for v in norm_std0])}]")
# In[ ]:


# use normalization results computed earlier
if input_size == 224:
    norm_mean = np.array([])
    norm_std0 = np.array([])
elif input_size == 600:
    norm_mean = np.array([0.95740360, 0.95740360, 0.95740360])
    norm_std0 = np.array([0.08236791, 0.08236791, 0.08236791])
elif input_size == 800:
    norm_mean = np.array([0.95808870, 0.95808870, 0.95808870])
    norm_std0 = np.array([0.09361055, 0.09361055, 0.09361055])
else:
    raise ValueError(f'No precomputed mean, std available for input_size = {input_size}')

if im_channels == 1:
    norm_mean = np.array([norm_mean[0]])
    norm_std0 = np.array([norm_std0[0]])

# use normalization results used when training the model, only works for timm models. Should probably only use for color images
norm_mean = np.array(model.default_cfg['mean'])
norm_std0 = np.array(model.default_cfg['std'])
# In[ ]:


print(f"norm_mean = [{', '.join([f'{v:.8f}' for v in norm_mean])}]")
print(f"norm_std0 = [{', '.join([f'{v:.8f}' for v in norm_std0])}]")


# ### Actually Load Data

# In[ ]:


# need to fake 3 channels r = b = g with Grayscale to use pretrained networks
transform = transforms.Compose([transforms.Grayscale(num_output_channels=im_channels), transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std0)])

ds_train = tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/train', transform=transform)
ds_val = tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/val', transform=transform)


# In[ ]:


class_to_idx = {}
for k,v in ds_train.class_to_idx.items():
    class_to_idx[k] = v
class_to_idx = dict(sorted(class_to_idx.items(), key=lambda x: x))
idx_to_class = dict([[v,k] for k,v in class_to_idx.items()])


# In[ ]:


pin_memory=True

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=8)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=8)


# In[ ]:


# ds_test = tv.datasets.ImageFolder(root=f'{data_path}/preprocessed/im_res_{im_res}/test', transform=transform)
# dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=8)


# In[ ]:


# test_mem()


# ***
# # Setup Loss Function
# Balance class weights or leave with None, ie uniform, weights  

# In[ ]:


class_counts = torch.zeros(n_classes)
for idx in range(n_classes):
    idx_class_tensor = torch.tensor(ds_train.targets) == idx
    class_counts[idx] = idx_class_tensor.sum().item()
print(f'Class Counts: {class_counts}')

if balance_class_weights:
    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights / class_weights.max()
    class_weights = class_weights.to(device)
    print(f'Class Weights: {class_weights}')

    class_counts_weighted = class_counts
    for i in range(len(class_counts)):
        class_counts_weighted[i] = class_weights[i]*class_counts_weighted[i]
    print(f'Class Counts Weighted: {class_counts_weighted}')
else:
    class_weights=None
    print('Using default, ie uniform, weights')


# In[ ]:


# https://pytorch.org/docs/stable/nn.html#crossentropyloss
loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
# reduction='mean', return mean CrossEntropyLoss over batches


# ***
# # Train

# In[ ]:


# test_mem()


# In[ ]:


model_info = {
    'start_time': str(dt.datetime.now()),
    'model_name': model_name,
    'optimizer': str(optimizer).replace('\n   ', ',').replace('\n', ''),
    'loss_fn': str(loss_fn),
    'loss_fn.reduction': loss_fn.reduction,
    'resume_training': resume_training,
    'batch_size': batch_size,
    'feature_extract': feature_extract,
    'use_pretrained': use_pretrained,
    'balance_class_weights': balance_class_weights,
    'class_counts': ', '.join([f'{c:.0f}' for c in class_counts]),
    'class_weights': ', '.join([f'{c:f}' for c in class_weights]),
    'data_path': data_path,
    'input_size': input_size,
    'im_res': im_res,
    'im_channels': im_channels,
    'rnd_seed': rnd_seed,
    'norm_mean': ', '.join([f'{c:f}' for c in norm_mean]),
    'norm_std0': ', '.join([f'{c:f}' for c in norm_std0]),
    'starting_memory': f'CUDA memory allocated: {humanize.naturalsize(torch.cuda.memory_allocated())}, cached: {humanize.naturalsize(torch.cuda.memory_cached())}',
    'pin_memory': pin_memory,
    'n_classes': n_classes,
    'idx_to_class': idx_to_class,
    'Dx_classes': Dx_classes,
}

os.makedirs(models_path, exist_ok=True)
with open(os.path.join(models_path, 'model_info.json'), 'w') as f_json:
    json.dump(model_info, f_json, sort_keys=False, indent=4)


# In[ ]:


dfp_train_results = train_model(dl_train, dl_val,
model, optimizer, loss_fn, device,
model_name=model_name, models_path=models_path,
max_epochs=300, max_time_min=180,
do_es=True, es_min_val_per_improvement=0.0005, es_epochs=10,
do_decay_lr=False, # initial_lr=0.001, lr_epoch_period=25, lr_n_period_cap=4,
# save_model_inhibit=10, # don't save anything out for the first save_model_inhibit epochs, set to -1 to start saving immediately
n_models_on_disk=3, # keep the last n_models_on_disk models on disk, set to -1 to keep all
dfp_train_results_prior=dfp_train_results_prior # dfp_train_results from prior training session, use to resume
)


# ***
# # Eval

# In[ ]:


dfp_train_results = load_dfp(models_path, 'train_results', tag='', cols_bool=['saved_model'],
                             cols_float=['train_loss','val_loss','best_val_loss','delta_per_best','elapsed_time','epoch_time'])


# In[ ]:


dfp_train_results


# In[ ]:


plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='_val', inline=False,
                   ann_text_std_add=None,
                   y_axis_params={'log': False},
                   loss_cols=['val_loss'],
                  )


# In[ ]:


plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='_train', inline=False,
                   ann_text_std_add=None,
                   y_axis_params={'log': False},
                   loss_cols=['train_loss'],
                  )


# In[ ]:


plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='', inline=False,
                   ann_text_std_add=None,
                   y_axis_params={'log': False},
                   loss_cols=['train_loss', 'val_loss'],
                  )


# ### Load model from disk

# In[ ]:


best_epoch = dfp_train_results.iloc[dfp_train_results['val_loss'].idxmin()]['epoch']
load_model(model, device, best_epoch, model_name, models_path)


# In[ ]:


best_epoch


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


cms = {'_val': True, '_raw_val': False}
for tag,norm in cms.items():
    plot_confusion_matrix(conf_matrix, label_names=Dx_classes.keys(),
                          m_path=output_path, tag=tag, inline=False,
                          ann_text_std_add=None,
                          normalize=norm,
                         )


# ***
# # Dev

# In[ ]:


from common_code import *


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


test_mem(print_objects=False)


# In[ ]:





# In[ ]:





# In[ ]:




