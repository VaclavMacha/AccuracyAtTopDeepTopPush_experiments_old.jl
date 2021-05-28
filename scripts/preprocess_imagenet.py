import os
import numpy as np
from tqdm import tqdm
import scipy
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

os.getcwd()
os.chdir('/disk/macha/data_deeptoppush/datasets/ImageNet')

def eval_dataset(model, loader, set, saveat):
    s_pred_list = []
    s_list = []
    y_list = []
    i = 0
    k = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            i += 1
            x, y = batch
            outputs = model(x.cuda())
            s_gpu = model.extract_features(x.cuda())
            s_cpu = s_gpu.detach().cpu().numpy()

            s_pred_list.append(outputs.detach().cpu().numpy())
            s_list.append(s_cpu)
            y_list.append(y)

            del s_gpu, outputs
            torch.cuda.empty_cache()

            if i % saveat == 0:
                k += 1
                s = np.concatenate(s_list, axis=0)
                y = np.concatenate(y_list, axis=0)
                s_pred = np.concatenate(s_pred_list, axis=0)

                np.save(f'{set}_samples_{k}.npy', s.reshape(s.shape[0], -1))
                np.save(f'{set}_labels_{k}.npy', y)
                np.save(f'{set}_scores_{k}.npy', s_pred)

                s_pred_list = []
                s_list = []
                y_list = []
    k += 1
    s = np.concatenate(s_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    s_pred = np.concatenate(s_pred_list, axis=0)

    np.save(f'{set}_samples_{k}.npy', s.reshape(s.shape[0], -1))
    np.save(f'{set}_labels_{k}.npy', y)
    np.save(f'{set}_scores_{k}.npy', s_pred)
    return

# preprocess
model = EfficientNet.from_pretrained('efficientnet-b0')
model.cuda()
model.eval()

tfms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = 'train'

imagenet_data = torchvision.datasets.ImageNet(
    '/disk/macha/data_deeptoppush/datasets/ImageNet_jpeg',
    dataset,
    transform = tfms
)

loader = DataLoader(
    imagenet_data,
    batch_size=25,
    shuffle=False,

)

eval_dataset(model, loader, dataset, saveat = 2000)
