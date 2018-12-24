# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from model_triplet import ft_net
import json
from shutil import copyfile
from triplet_loss import TripletFolder

version = torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50_VIPeR_triplet', type=str, help='output model name')
parser.add_argument('--data_dir',default='/media/data2/songzr/mydata/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--margin', default=0.3, type=float, help='margin')
parser.add_argument('--poolsize', default=128, type=int, help='poolsize')
parser.add_argument('--alpha', default=0.0, type=float, help='regularization, push to -1')


opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

image_datasets = TripletFolder(os.path.join(data_dir, 'train_all'),data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,shuffle=True, num_workers=8)           

since = time.time()
inputs, classes, pos, pos_classes = next(iter(dataloaders))
print(time.time()-since)

# Training the model
# illustrate:
# -  Scheduling the learning rate
# -  Saving the best model
# In the following, parameter ``scheduler`` is an LR scheduler object from ``torch.optim.lr_scheduler``.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0

    for epoch in range(num_epochs):
        scheduler.step()
        model.train(True)  # Set model to training mode
         
        running_loss = 0.0
        running_corrects = 0.0
        running_margin = 0.0
        running_reg = 0.0

        for data in dataloaders:
            inputs, labels, pos, pos_labels = data
            now_batch_size,c,h,w = inputs.shape
            if now_batch_size<opt.batchsize: # skip the last batch
                continue
            pos = pos.view(4*opt.batchsize,c,h,w)
            pos_labels = pos_labels.repeat(4).reshape(4,opt.batchsize)
            pos_labels = pos_labels.transpose(0,1).reshape(4*opt.batchsize)

            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                pos = Variable(pos.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, pos, labels = Variable(inputs), Variable(pos), Variable(labels)

            optimizer.zero_grad()
            outputs, f = model(inputs)
            _, pf = model(pos)
            neg_labels = pos_labels
            # hard-neg
            nf_data = pf # 128*512
            rand = np.random.permutation(4*opt.batchsize)[0:opt.poolsize]
            nf_data = nf_data[rand,:]
            neg_labels = neg_labels[rand]
            nf_t = nf_data.transpose(0,1) # 512*128
            score = torch.mm(f.data, nf_t) # cosine 32*128 
            score, rank = score.sort(dim=1, descending = True) # score high == hard
            labels_cpu = labels.cpu()
            nf_hard = torch.zeros(f.shape).cuda()
            for k in range(now_batch_size):
                hard = rank[k,:]
                for kk in hard:
                    now_label = neg_labels[kk] 
                    anchor_label = labels_cpu[k]
                    if now_label != anchor_label:
                        nf_hard[k,:] = nf_data[kk,:]
                        break
            #hard-pos            
            pf_hard = torch.zeros(f.shape).cuda() # 32*512
            for k in range(now_batch_size):
                pf_data = pf[4*k:4*k+4,:]
                pf_t = pf_data.transpose(0,1) # 512*4
                ff = f.data[k,:].reshape(1,-1) # 1*512
                score = torch.mm(ff, pf_t) #cosine
                score, rank = score.sort(dim=1, descending = False) #score low == hard
                pf_hard[k,:] = pf_data[rank[0][0],:]

            # loss
            # criterion_triplet = nn.MarginRankingLoss(margin=opt.margin)                
            pscore = torch.sum( f * pf_hard, dim=1) 
            nscore = torch.sum( f * nf_hard, dim=1)
            y = torch.ones(now_batch_size)
            y = Variable(y.cuda())


            _, preds = torch.max(outputs.data, 1)
            #loss_triplet = criterion(f, pf_hard, nf_hard)
            reg = torch.sum((1+nscore)**2) + torch.sum((-1+pscore)**2)
            loss = torch.sum(torch.nn.functional.relu(nscore + opt.margin - pscore))  #Here I use sum
            loss_triplet = loss + opt.alpha*reg
            loss_triplet.backward()
            optimizer.step()

            running_loss += loss_triplet.item() * now_batch_size
            running_corrects += float(torch.sum(pscore>nscore+opt.margin))
            running_margin +=float(torch.sum(pscore-nscore))
            running_reg += reg

        epoch_loss = running_loss / len(image_datasets)
        epoch_reg = opt.alpha*running_reg/ len(image_datasets)
        epoch_acc = running_corrects / len(image_datasets)
        epoch_margin = running_margin / len(image_datasets)
        print('now_margin: %.4f'%opt.margin)           
        print('Epoch {}/{} \t train Loss: {:.4f} \t Acc: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))
        if epoch_margin>last_margin:
            last_margin = epoch_margin            
            last_model_wts = model.state_dict()
        if epoch%10 == 9:
            save_network(model, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model
# Save model
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

# Finetuning the convnet
# Load a pretrainied model and reset final fully connected layer.
model = ft_net(len(image_datasets.classes))
print(model)

if torch.cuda.is_available():
    model = model.cuda()

criterion_triplet = nn.MarginRankingLoss(margin=opt.margin) #nn.CrossEntropyLoss()
ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.01},
        {'params': model.model.fc.parameters(), 'lr': 0.1},
        {'params': model.classifier.parameters(), 'lr': 0.1}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion_triplet, optimizer_ft, exp_lr_scheduler,
                       num_epochs=60)
