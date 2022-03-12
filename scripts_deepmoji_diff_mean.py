import os,argparse,time
import numpy as np
from datetime import datetime
import random
from random import shuffle
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataloaders.deep_moji import DeepMojiDataset
from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator
from dataloaders.FairBatchSampler import FairBatch

from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss
from networks.contrastive_loss import Contrastive_Loss
from networks.center_loss import CenterLoss
from networks.angular_loss import AngularPenaltySMLoss


from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.eval_metrices_updated import leakage_evaluation, tpr_binary, leakage_hidden, leakage_logits

from pathlib import Path, PureWindowsPath
import pandas as pd
import argparse
from collections import defaultdict, Counter
import copy
import time

import matplotlib.pyplot as plt

monitor_overall_loss = []
monitor_group_0_loss = []
monitor_group_1_loss = []
monitor_group_0_class_0_loss = []
monitor_group_1_class_0_loss = []
monitor_group_0_class_1_loss = []
monitor_group_1_class_1_loss = []
monitor_overall_acc = []
monitor_group_0_acc = []
monitor_group_1_acc = []

monitor_micro_f1 = []
monitor_macro_f1 = []
monitor_weighted_f1 = []
monitor_per_class_f1 = []
monitor_class_distribution = []
monitor_per_class_group_0_f1 = []
monitor_per_class_group_1_f1 = []
monitor_group_0_percentage = []
monitor_group_1_percentage = []

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, adv_optimizers, criterion, device, args):
    """"
    Train the discriminator to get a meaningful gradient
    """

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.train()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    
    for batch in iterator:
        
        text = batch[0]
        tags = batch[1].long()
        p_tags = batch[2].long()

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        hs = model.hidden(text).detach()
        
        # iterate all discriminators
        for discriminator, adv_optimizer in zip(discriminators, adv_optimizers):
        
            adv_optimizer.zero_grad()

            adv_predictions = discriminator(hs)

        
            loss = criterion(adv_predictions, p_tags)

            # encrouge orthogonality
            if args.DL == True:
                # Get hidden representation.
                adv_hs_current = discriminator.hidden_representation(hs)
                for discriminator2 in discriminators:
                    if discriminator != discriminator2:
                        adv_hs = discriminator2.hidden_representation(hs)
                        # Calculate diff_loss
                        # should not include the current model
                        difference_loss = args.diff_LAMBDA * args.diff_loss(adv_hs_current, adv_hs)
                        loss = loss + difference_loss
                        
            loss.backward()
        
            adv_optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# evaluate the discriminator
def adv_eval_epoch(model, discriminators, iterator, criterion, device, args):

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.eval()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    

    preds = {i:[] for i in range(args.n_discriminator)}
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).long()
        
        # extract hidden state from the main model
        hs = model.hidden(text)
        # let discriminator make predictions

        for index, discriminator in enumerate(discriminators):
            adv_pred = discriminator(hs)
        
            loss = criterion(adv_pred, p_tags)
                        
            epoch_loss += loss.item()
        
            adv_predictions = adv_pred.detach().cpu()
            preds[index] += list(torch.argmax(adv_predictions, axis=1).numpy())


        tags = tags.cpu().numpy()

        labels += list(tags)
        
        private_labels += list(batch[2].cpu().numpy())
        
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

# train the main model with adv loss
def train_epoch(model, discriminators, iterator, optimizer, criterion, contrastive_loss, center_loss, angular_loss, device, args):
    
    epoch_loss = 0
    epoch_acc = 0
    #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    model.train()
    for discriminator in discriminators:
        discriminator.train()

    # activate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = True
    
    labels = []
    p_labels = []
    tem_preds = []

    overall_loss = []
    group_0_loss = []
    group_1_loss = []
    group_0_class_0_loss = []
    group_1_class_0_loss = []
    group_0_class_1_loss = []
    group_1_class_1_loss = []


    counter = 0
    for batch in iterator:
        
        text = batch[0]
        tags = batch[1].long()
        #p_tags = batch[2].float()
        p_tags = batch[2].long()
        weights = batch[3]
        #print(weights)

        if args.mode in ['fairbatch', 'fairbatch+difference']:
            text = text.to(device)
            text = text.squeeze()
            tags = tags.to(device)
            tags = tags.squeeze()
            p_tags = p_tags.to(device)
            p_tags = p_tags.squeeze()
        else:
            text = text.to(device)
            tags = tags.to(device)
            p_tags = p_tags.to(device)
            weights = weights.to(device)

        
        optimizer.zero_grad()
        # main model predictions
        predictions, features_1, features_2, features = model(text)

        if args.mode == 'vanilla':
            loss = criterion(predictions, tags)
        elif args.mode == 'rw':
            loss = (criterion(predictions, tags)*weights).mean()
        elif args.mode == 'ds':
            loss = criterion(predictions, tags)
        elif args.mode == 'difference':
            loss = criterion(predictions, tags) 
            tem_loss = 0
            tem = []
            for i in range(0, 2):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                tem_0 = criterion(predictions[indices_0], tags[indices_0])
                tem_1 = criterion(predictions[indices_1], tags[indices_1])
                tem_loss+=args.lambda_weight*abs(tem_0-tem_1)#default 0.5
            loss+=tem_loss
        elif args.mode == 'mean':
                loss = criterion(predictions, tags)
                accu_loss = 0
                for i in range(0, args.num_classes):
                    for j in range(0, 2):
                        indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                        indices = list(indices)
                        loss_c_g = criterion(predictions[indices], tags[indices])
                        accu_loss+=args.lambda_weight*abs(loss_c_g-loss) #default 0.5
                loss+=accu_loss
        elif args.mode == 'fairbatch':
            loss = criterion(predictions, tags)
        elif args.mode == 'difference+max':
            loss = criterion(predictions, tags) 
            tem_loss = 0
            tem = []
            for i in range(0, 2):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                tem_0 = criterion(predictions[indices_0], tags[indices_0])
                tem_1 = criterion(predictions[indices_1], tags[indices_1])
                tem_loss+=args.lambda_weight*max(tem_0, tem_1)
            loss+=tem_loss
        elif args.mode == 'difference+min':
            loss = criterion(predictions, tags) 
            tem_loss = 0
            tem = []
            for i in range(0, 2):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                tem_0 = criterion(predictions[indices_0], tags[indices_0])
                tem_1 = criterion(predictions[indices_1], tags[indices_1])
                tem_loss-=args.lambda_weight*min(tem_0, tem_1)
            loss+=tem_loss   
        elif args.mode == 'mean+max':
            loss = criterion(predictions, tags)
            accu_loss = 0
            for i in range(0, args.num_classes):
                for j in range(0, 2):
                    indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                    indices = list(indices)
                    loss_c_g = criterion(predictions[indices], tags[indices])
                    accu_loss+=args.lambda_weight*max(loss_c_g, loss) #default 0.5
            loss+=accu_loss
        elif args.mode == 'mean+min':
            loss = criterion(predictions, tags)
            accu_loss = 0
            for i in range(0, args.num_classes):
                for j in range(0, 2):
                    indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                    indices = list(indices)
                    loss_c_g = criterion(predictions[indices], tags[indices])
                    accu_loss+=args.lambda_weight*min(loss_c_g, loss) #default 0.5
            loss-=accu_loss
        elif args.mode == 'fairbatch+difference':
            loss = criterion(predictions, tags) 
            tem_loss = 0
            tem = []
            for i in range(0, 2):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    tem_loss+=args.lambda_weight*abs(tem_0-tem_1)#default 0.5
            loss+=tem_loss

        if args.adv:
            # discriminator predictions
            p_tags = p_tags.long()

            hs = model.hidden(text)

            for discriminator in discriminators:
                adv_predictions = discriminator(hs)
            
                loss = loss + (criterion(adv_predictions, p_tags) / len(discriminators))

        #predictions1 = copy.deepcopy(predictions)
        predictions1 = predictions.detach().cpu()
        tem_preds += list(torch.argmax(predictions1, axis=1).numpy())
        labels +=tags.cpu().tolist()
        p_labels +=p_tags.cpu().tolist()
                        
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
    accuracy = accuracy_score(labels, tem_preds)

    group_0_index = []
    group_1_index = []
    for i in range(0, len(labels)):
        if p_labels[i] == 0:
            group_0_index.append(i)
        else:
            group_1_index.append(i)

    group_0_labels = [labels[index] for index in group_0_index]
    group_0_preds = [tem_preds[index] for index in group_0_index]
    group_0_acc = accuracy_score(group_0_labels, group_0_preds)

    group_1_labels = [labels[index] for index in group_1_index]
    group_1_preds = [tem_preds[index] for index in group_1_index]
    group_1_acc = accuracy_score(group_1_labels, group_1_preds)

    monitor_overall_loss.append(sum(overall_loss)/float(counter+1))
    monitor_group_0_loss.append(sum(group_0_loss)/float(1+len(group_0_labels)))
    monitor_group_1_loss.append(sum(group_1_loss)/float(1+len(group_1_labels)))

    group_0_class_0 = []
    group_0_class_1 = []
    for i in range(0, len(group_0_labels)):
        if group_0_labels[i] == 0:
            group_0_class_0.append(i)
        else:
            group_0_class_1.append(i)

    monitor_group_0_class_0_loss.append(sum(group_0_class_0_loss)/float(1+len(group_0_class_0)))
    monitor_group_0_class_1_loss.append(sum(group_0_class_1_loss)/float(1+len(group_0_class_1)))

    group_1_class_0 = []
    group_1_class_1 = []
    for i in range(0, len(group_1_labels)):
        if group_1_labels[i] == 0:
            group_1_class_0.append(i)
        else:
            group_1_class_1.append(i)

    monitor_group_1_class_0_loss.append(sum(group_1_class_0_loss)/float(1+len(group_1_class_0)))
    monitor_group_1_class_1_loss.append(sum(group_1_class_1_loss)/float(1+len(group_1_class_1)))

    monitor_overall_acc.append(accuracy)
    monitor_group_0_acc.append(group_0_acc)
    monitor_group_1_acc.append(group_1_acc)
    #print('training accuracy', accuracy)
    return epoch_loss / len(iterator)

# to evaluate the main model
def eval_main(model, iterator, criterion, device, args):
    
    epoch_loss = 0
    
    model.eval()
    
    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        # tags = batch[2] #Reverse
        p_tags = batch[2]
        weights = batch[3]

        text = text.to(device)
        tags = tags.to(device).long()
        #p_tags = p_tags.to(device).float()
        p_tags = p_tags.to(device).long()
        weights = weights.to(device)

        predictions, _, _, _ = model(text)
        
        loss = criterion(predictions, tags)
        if args.mode == 'rw':
            loss=loss.mean()
                        
        epoch_loss += loss.item()
        
        predictions = predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)


def get_leakage_data(model, iterator, filename, device, args):
    model.eval()
    
    data_frame = pd.DataFrame()
    preds = []
    labels = []
    private_labels = []
    second_last_representation = []
    tem_preds = []
    for batch in iterator:
        
        text = batch[0].float()
        tags = batch[1].long()
        
        text = text.to(device)
        tags = tags.to(device)
        
        #text = F.normalize(text, dim=1)
        predictions, _, _, second_last = model(text)
        
        predictions = predictions.detach().cpu()
        preds+=predictions.tolist()
        
        second_last = second_last.detach().cpu()
        second_last_representation+=second_last.tolist()

        tem_preds += list(torch.argmax(predictions, axis=1).numpy())

        labels +=tags.cpu().tolist()
        private_labels += list(batch[2].tolist())
    
    data_frame['prob'] = preds
    data_frame['sentiment'] = labels
    data_frame['race'] = private_labels
    data_frame['second_last_representation'] = second_last_representation
    data_frame['predict'] = tem_preds
    data_frame.to_pickle(filename)
    accuracy = accuracy_score(labels, tem_preds)
    print('Potential', accuracy)

    X_logits = list(data_frame['prob'])
    X_hidden = list(data_frame['second_last_representation'])
    y = list(data_frame['sentiment'])
    gender_label = list(data_frame['race'])

    return (X_logits, X_hidden, y, gender_label)

def load_leakage_data(filename):
    data = pd.read_pickle(filename)
    X_logits = list(data['prob'])
    X_hidden = list(data['second_last_representation'])
    y = list(data['sentiment'])
    gender_label = list(data['race'])
        
    return (X_logits, X_hidden, y, gender_label)


def get_group_metrics(preds, labels, p_labels, train_data):

    preds_0 = []
    labels_0 = []
    preds_1 = []
    labels_1 = []
    for i in range(0, len(p_labels)):
        if p_labels[i] == 0:
            preds_0.append(preds[i])
            labels_0.append(labels[i])
        else:
            preds_1.append(preds[i])
            labels_1.append(labels[i])

    accuracy_0 = 100*accuracy_score(labels_0, preds_0)
    accuracy_1 = 100*accuracy_score(labels_1, preds_1)

    f1_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    f1_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    #print('hello world', f1_group_0, f1_group_1)

    #micro
    micro_f1_0 = 100*f1_score(labels_0, preds_0, average='micro')
    micro_f1_1 = 100*f1_score(labels_1, preds_1, average='micro')
    monitor_micro_f1[-1].append(abs(micro_f1_0-micro_f1_1))
    monitor_micro_f1[-1].append(min(micro_f1_0, micro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_micro_f1[-1].append(micro_f1_0)
    else:
        monitor_micro_f1[-1].append(micro_f1_1)

    #macro
    macro_f1_0 = 100*f1_score(labels_0, preds_0, average='macro')
    macro_f1_1 = 100*f1_score(labels_1, preds_1, average='macro')
    monitor_macro_f1[-1].append(abs(macro_f1_0-macro_f1_1))
    monitor_macro_f1[-1].append(min(macro_f1_0, macro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_macro_f1[-1].append(macro_f1_0)
    else:
        monitor_macro_f1[-1].append(macro_f1_1)

    #weighted
    weighted_f1_0 = 100*f1_score(labels_0, preds_0, average='weighted')
    weighted_f1_1 = 100*f1_score(labels_1, preds_1, average='weighted')
    monitor_weighted_f1[-1].append(abs(weighted_f1_0-weighted_f1_1))
    monitor_weighted_f1[-1].append(min(weighted_f1_0, weighted_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_weighted_f1[-1].append(weighted_f1_0)
    else:
        monitor_weighted_f1[-1].append(weighted_f1_1)       


    if len(preds_0) <= len(preds_1):
        minority = accuracy_0
    else:
        minority = accuracy_1


    #f1_per_class = 100*f1_score(labels, preds, average=None)
    matrix = confusion_matrix(labels, preds)
    f1_per_class = 100*matrix.diagonal()/matrix.sum(axis=1)

    counter = Counter(train_data.y)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(train_data))

    monitor_per_class_f1.append([])
    monitor_class_distribution.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_f1[-1].extend([f1_per_class[i]])
        monitor_class_distribution[-1].append(100*counter[i])

    labels_0_train = []
    labels_1_train = []
    p_labels = train_data.private_label
    for i in range(0, len(p_labels)):
        if p_labels[i] == 0:
            labels_0_train.append(train_data.y[i])
        else:
            labels_1_train.append(train_data.y[i])


    #f1_per_class_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    matrix = confusion_matrix(labels_0, preds_0)
    f1_per_class_group_0 = 100*matrix.diagonal()/matrix.sum(axis=1)

    overall_counter = Counter(train_data.y)
    counter = Counter(labels_0_train)

    for tem in counter:
        counter[tem] = counter[tem]/float(overall_counter[tem])

    monitor_per_class_group_0_f1.append([])
    monitor_group_0_percentage.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_0_f1[-1].extend([f1_per_class_group_0[i]])
        monitor_group_0_percentage[-1].append(100*counter[i])

    #f1_per_class_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    matrix = confusion_matrix(labels_1, preds_1)
    f1_per_class_group_1 = 100*matrix.diagonal()/matrix.sum(axis=1)

    counter = Counter(labels_1_train)

    for tem in counter:
        counter[tem] = counter[tem]/float(overall_counter[tem])
    monitor_per_class_group_1_f1.append([])
    monitor_group_1_percentage.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_1_f1[-1].extend([f1_per_class_group_1[i]])
        monitor_group_1_percentage[-1].append(100*counter[i])

    return abs(accuracy_0-accuracy_1), min(accuracy_0, accuracy_1), 0.5*(accuracy_0+accuracy_1), minority

def get_group_metrics_1(preds, labels, p_labels):

    f1_per_class = 100*f1_score(labels, preds, average=None)
    counter = Counter(labels)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(labels))

    monitor_per_class_f1.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_f1[-1].extend([counter[i], f1_per_class[i]])

    preds_0 = []
    labels_0 = []
    preds_1 = []
    labels_1 = []
    for i in range(0, len(p_labels)):
        if p_labels[i] == 0:
            preds_0.append(preds[i])
            labels_0.append(labels[i])
        else:
            preds_1.append(preds[i])
            labels_1.append(labels[i])

    accuracy_0 = 100*accuracy_score(labels_0, preds_0)
    accuracy_1 = 100*accuracy_score(labels_1, preds_1)

    f1_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    f1_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    print('hello world', f1_group_0, f1_group_1)

    #micro
    micro_f1_0 = 100*f1_score(labels_0, preds_0, average='micro')
    micro_f1_1 = 100*f1_score(labels_1, preds_1, average='micro')
    monitor_micro_f1[-1].append(abs(micro_f1_0-micro_f1_1))
    monitor_micro_f1[-1].append(min(micro_f1_0, micro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_micro_f1[-1].append(micro_f1_0)
    else:
        monitor_micro_f1[-1].append(micro_f1_1)

    #macro
    macro_f1_0 = 100*f1_score(labels_0, preds_0, average='macro')
    macro_f1_1 = 100*f1_score(labels_1, preds_1, average='macro')
    monitor_macro_f1[-1].append(abs(macro_f1_0-macro_f1_1))
    monitor_macro_f1[-1].append(min(macro_f1_0, macro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_macro_f1[-1].append(macro_f1_0)
    else:
        monitor_macro_f1[-1].append(macro_f1_1)

    #weighted
    weighted_f1_0 = 100*f1_score(labels_0, preds_0, average='weighted')
    weighted_f1_1 = 100*f1_score(labels_1, preds_1, average='weighted')
    monitor_weighted_f1[-1].append(abs(weighted_f1_0-weighted_f1_1))
    monitor_weighted_f1[-1].append(min(weighted_f1_0, weighted_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_weighted_f1[-1].append(weighted_f1_0)
    else:
        monitor_weighted_f1[-1].append(weighted_f1_1)       


    f1_per_class_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    counter = Counter(labels_0)
    print(counter)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(labels_0))

    monitor_per_class_group_0_f1.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_0_f1[-1].extend([counter[i], f1_per_class_group_0[i]])

    f1_per_class_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    counter = Counter(labels_1)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(labels_1))

    monitor_per_class_group_1_f1.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_1_f1[-1].extend([counter[i], f1_per_class_group_1[i]])

    if len(preds_0) <= len(preds_1):
        minority = accuracy_0
    else:
        minority = accuracy_1

    print(len(preds_0), len(preds_1), accuracy_0, accuracy_1)
    return abs(accuracy_0-accuracy_1), min(accuracy_0, accuracy_1), 0.5*(accuracy_0+accuracy_1), minority

def log_uniform(power_low, power_high):
    return np.power(10, np.random.uniform(power_low, power_high))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_loss_curve(overall_loss, group_0_loss, group_1_loss, group_0_class_0_loss, group_1_class_0_loss, group_0_class_1_loss, group_1_class_1_loss, overall_acc=None, group_0_acc=None, group_1_acc=None):
    print(overall_loss)
    print(group_0_loss)
    print(group_1_loss)
    print(group_0_class_0_loss)
    print(group_1_class_0_loss)
    print(group_0_class_1_loss)
    print(group_1_class_1_loss)

    #print(overall_acc)
    #print(group_0_acc)
    #print(group_1_acc)
    epochs = [i for i in range(1, 1+len(overall_loss))]
    plt.plot(epochs, overall_loss, '-o', label='overall loss')
    plt.plot(epochs, group_0_loss, '-s', label='group 0 loss')
    plt.plot(epochs, group_1_loss, '-p', label='group 1 loss')
    plt.plot(epochs, group_0_class_0_loss, '->', label='group 0 class 0 loss')
    plt.plot(epochs, group_1_class_0_loss, '-<', label='group 1 class 0 loss')
    plt.plot(epochs, group_0_class_1_loss, '-v', label='group 0 class 1 loss')
    plt.plot(epochs, group_1_class_1_loss, '-^', label='group 1 class 1 loss')

    #plt.plot(epochs, overall_acc, label='overall acc')
    #plt.plot(epochs, group_0_acc, label='group 0 acc')
    #plt.plot(epochs, group_1_acc, label='group 1 acc')
    plt.legend()

    plt.savefig('./moji_loss.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 2304)
    parser.add_argument('--num_classes', type=int, default = 2)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--adv_level', type=int, default = -1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--starting_power', type=int)
    parser.add_argument('--LAMBDA', type=float, default=0.8)
    parser.add_argument('--n_discriminator', type=int, default = 0)
    parser.add_argument('--adv_units', type=int, default = 256)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--positive_class_ratio', type=float, default=0.5)
    parser.add_argument('--DL', action='store_true')
    parser.add_argument('--diff_LAMBDA', type=float, default=1000)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default = 512)    
    parser.add_argument('--seed', type=int, default = 46)
    parser.add_argument('--num_epochs', type=int, default = 60)
    parser.add_argument('--representation_file', default='./analysis/ce+scl1+scl2.txt', type=str, help='the file storing test representation before the classifier layer')
    parser.add_argument('--loss_type', default='ce', type=str, help='the type of loss we want to use')
    parser.add_argument('--lambda_weight', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--experiment_type', default='adv_experiment_lambda', type=str, help='which types of experiments we are doing')
    parser.add_argument('--device_id', type=int, default = 0)
    parser.add_argument('--balance_type', default='stratified', type=str, help='which types of experiments we are balanced on')
    parser.add_argument('--mode', default='vanilla', type=str, help='which types of experiments we are doing')
    parser.add_argument('--dataset_mode', default='gender', type=str, help='which types of manipulation to the dataset, protected level or class level')

    args = parser.parse_args()

    batch_list = [256, 512, 1024, 2048, 4096]
    lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]


    adv_batch_list = [256, 512, 1024, 2048]
    adv_lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    adv_lambda_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    adv_diff_lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    accumulate_acc = []
    count_runs = 0
    accumulate_time = []
    accumulate_group = []

    selected_lambda = args.lambda_weight
    if True:
        if True:
            #for tem_lambda in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:#[1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.06,0.07,0.08,0.09,0.1,0.5,1]:
            #for tem_lambda in adv_diff_lambda_list:
            #for tem_lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            #for tem_lambda in [0.6, 0.7, 0.8, 0.9]:#[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]:
            #for batch_size in [128, 256, 512, 1024, 2048, 5096]:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
            #for tem_lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            #for tem_lambda in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]:#[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
                #args.batch_size = batch_size
                #tem_lambda = batch_size
                #args.lr = args.lr*(batch_size/2048.0)
                args.seed = tem_seed
                tem_lambda = tem_seed
                
                #args.lambda_weight = tem_lambda
                #args.lambda_weight = tem_lambda
                
                #tem_lambda = tem_seed
                #args.lambda_weight = tem_lambda
                
                
                print('============================================================')
                print('batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight)
                #print('batch size', args.batch_size, 'lr', args.lr)

                seed_everything(args.seed)
                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                if args.dataset_mode == 'gender':
                    main_model_path = "./explore/moji_model_{}_{}_{}_{}_{}_{}.pt".format(args.mode, args.lambda_weight, args.batch_size, args.loss_type, args.ratio, args.seed)
                else:
                    main_model_path = "./explore/moji_model_{}_{}_{}_{}_{}_gender_0.8_class_imbalanced_{}.pt".format(args.mode, args.lambda_weight, args.batch_size, args.loss_type, args.positive_class_ratio, args.seed)
                #main_model_path = "./official_models/moji_model.pt" #_class_imbalanced
                #main_model_path = "./difference/moji_model_{}_{}_{}_{}_{}_{}.pt".format(args.mode, args.lr, args.batch_size, args.loss_type, args.ratio, args.seed)
                adv_model_path = "./explore/moji_discriminator_{}_{}_{}_{}.pt"

                # Device
                device = torch.device("cuda:"+str(args.device_id))
                # Init model
                model = DeepMojiModel(args)
                model = model.to(device)

                data_path = args.data_path
                # Load data
                if args.mode == 'ds' or args.mode == 'ds+difference' or args.mode == 'ds+mean':
                    balance_flag = True
                else:
                    balance_flag = False
                  
                controlled_ratio = 1
                train_data = DeepMojiDataset(args, data_path, "train", balanced=balance_flag, balance_type=args.balance_type, shuffle=shuffle, weight_scheme='joint', ratio=args.ratio, n = int(100000*controlled_ratio))

                #train_data = DeepMojiDataset(args, data_path, "train", balanced=False, balance_type=args.balance_type, shuffle=shuffle, weight_scheme='joint', ratio=args.ratio, n = 100000)
                dev_data = DeepMojiDataset(args, data_path, "dev", balanced=False, balance_type=args.balance_type, shuffle=None,)
                test_data = DeepMojiDataset(args, data_path, "test", balanced=False, balance_type=args.balance_type, shuffle=None,)
                # Data loader
                if args.mode in ['fairbatch', 'fairbatch+difference']:
                    sampler = FairBatch(model, torch.tensor(train_data.X).to(device), torch.tensor(train_data.y).long().to(device), torch.tensor(train_data.private_label).long().to(device), batch_size = args.batch_size, alpha=0.1, target_fairness = 'eqopp', replacement = False, seed = args.seed)
                    training_generator = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)
                else:
                    training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
                validation_generator = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                test_generator = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

                # Init discriminators
                # Number of discriminators
                n_discriminator = args.n_discriminator

                discriminators = [Discriminator(args, args.hidden_size, 2) for _ in range(n_discriminator)]
                discriminators = [dis.to(device) for dis in discriminators]

                diff_loss = DiffLoss()
                args.diff_loss = diff_loss

                contrastive_loss = Contrastive_Loss(device=device, temperature=args.temperature, base_temperature= args.temperature)

                center_loss = contrastive_loss
                angular_loss = contrastive_loss

                # Init optimizers
                LEARNING_RATE = args.lr
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

                adv_optimizers = [Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=1e-1*LEARNING_RATE) for dis in discriminators]

                # Init learing rate scheduler
                scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

                # Init criterion
                if args.mode == 'rw':
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')

                else:
                    criterion = torch.nn.CrossEntropyLoss()
                
                if args.mode == 'rw+difference':
                    contrastive_loss = torch.nn.CrossEntropyLoss()

                best_loss, valid_preds, valid_labels, _ = eval_main(
                                                                    model = model, 
                                                                    iterator = validation_generator, 
                                                                    criterion = criterion, 
                                                                    device = device, 
                                                                    args = args
                                                                    )
                best_loss = float('inf')
                best_acc = accuracy_score(valid_labels, valid_preds)
                best_epoch = 0
                start = time.time()

                for i in trange(0, args.num_epochs):
                    train_epoch(
                                model = model, 
                                discriminators = discriminators, 
                                iterator = training_generator, 
                                optimizer = optimizer, 
                                criterion = criterion, 
                                contrastive_loss = contrastive_loss,
                                center_loss = center_loss,
                                angular_loss = angular_loss,
                                device = device, 
                                args = args
                                )

                    valid_loss, valid_preds, valid_labels, _ = eval_main(
                                                                        model = model, 
                                                                        iterator = validation_generator, 
                                                                        criterion = criterion, 
                                                                        device = device, 
                                                                        args = args
                                                                        )
                    valid_acc = accuracy_score(valid_labels, valid_preds)
                    # learning rate scheduler
                    scheduler.step(valid_loss)
            
                    # early stopping
                    if valid_loss < best_loss:
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_epoch = i
                        torch.save(model.state_dict(), main_model_path)
                    else:
                        if best_epoch+5<=i:
                            break

                    # Train discriminator untile converged
                    # evaluate discriminator 
                    best_adv_loss, _, _, _ = adv_eval_epoch(
                                                            model = model, 
                                                            discriminators = discriminators, 
                                                            iterator = validation_generator, 
                                                            criterion = criterion, 
                                                            device = device, 
                                                            args = args
                                                            )
                    best_adv_epoch = -1
                    for k in range(100):
                        adv_train_epoch(
                                        model = model, 
                                        discriminators = discriminators, 
                                        iterator = training_generator, 
                                        adv_optimizers = adv_optimizers, 
                                        criterion = criterion, 
                                        device = device, 
                                        args = args
                                        )
                        adv_valid_loss, _, _, _ = adv_eval_epoch(
                                                                model = model, 
                                                                discriminators = discriminators, 
                                                                iterator = validation_generator, 
                                                                criterion = criterion, 
                                                                device = device, 
                                                                args = args
                                                                )
                            
                        if adv_valid_loss < best_adv_loss:
                                best_adv_loss = adv_valid_loss
                                best_adv_epoch = k
                                for j in range(args.n_discriminator):
                                    torch.save(discriminators[j].state_dict(), adv_model_path.format(experiment_type, j, args.LAMBDA, args.diff_LAMBDA))
                        else:
                            if best_adv_epoch + 5 <= k:
                                break
                    for j in range(args.n_discriminator):
                        discriminators[j].load_state_dict(torch.load(adv_model_path.format(experiment_type, j, args.LAMBDA, args.diff_LAMBDA)))

                end = time.time()
                accumulate_time.append(end-start)
                model.load_state_dict(torch.load(main_model_path))
                
                '''get_leakage_data(model, training_generator, './inlp_input/moji_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, validation_generator, './inlp_input/moji_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, test_generator, './inlp_input/moji_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                train_leakage_data = load_leakage_data('./inlp_input/moji_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                val_leakage_data = load_leakage_data('./inlp_input/moji_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                test_leakage_data = load_leakage_data('./inlp_input/moji_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))'''
                training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                train_leakage_data = get_leakage_data(model, training_generator, './output/moji_train.pickle', device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/moji_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/moji_test.pickle', device, args)
                
                # Evaluation
                test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                preds = np.array(preds)
                labels = np.array(labels)
                p_labels = np.array(p_labels)

                counter = Counter(test_data.y)
                for tem in counter:
                    counter[tem] = counter[tem]/float(len(test_data.y))
                #print(counter)
                rms_diff, weighted_rms_diff, tprs = tpr_binary(preds, labels, p_labels, counter)
                print('rms diff', rms_diff, 'weighted rms diff', weighted_rms_diff)
                logits_leakage = leakage_logits(train_leakage_data, val_leakage_data, test_leakage_data)
                hidden_leakage = leakage_hidden(train_leakage_data, val_leakage_data, test_leakage_data)
                accumulate_rms_diff.append(rms_diff)
                accumulate_weighted_rms_diff.append(weighted_rms_diff)
                accumulate_leakage_logits.append(logits_leakage[1])
                accumulate_leakage_hidden.append(hidden_leakage[1])

                accuracy = accuracy_score(labels, preds)

                micro_f1 = 100*f1_score(labels, preds, average='micro')
                monitor_micro_f1.append([micro_f1])
                macro_f1 = 100*f1_score(labels, preds, average='macro')
                monitor_macro_f1.append([macro_f1])
                weighted_f1 = 100*f1_score(labels, preds, average='weighted')
                monitor_weighted_f1.append([weighted_f1])

                difference, min_performance, macro_average, minority_performance = get_group_metrics(preds, labels, p_labels, train_data)
                accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff, difference, min_performance, macro_average, minority_performance])
                

                count_runs+=1
                print('hello world', count_runs, datetime.now())
                print(accumulate_time[count_runs-1])

                #plot_loss_curve(monitor_overall_loss, monitor_group_0_loss, monitor_group_1_loss, monitor_group_0_class_0_loss, monitor_group_1_class_0_loss, \
                #monitor_group_0_class_1_loss, monitor_group_1_class_1_loss, monitor_overall_acc, monitor_group_0_acc, monitor_group_1_acc)
                #print('plot finished')

                break
            #break
        #break

    #output_file.close()
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], accumulate_acc[i][7])#accumulate_acc[i][9], accumulate_acc[i][10], accumulate_acc[i][11], accumulate_acc[i][12])    
    
    exit()
    '''print('micro')
    for i in range(0, len(monitor_micro_f1)):
        print(monitor_micro_f1[i][0], monitor_micro_f1[i][1], monitor_micro_f1[i][2], monitor_micro_f1[i][3])

    print('macro')
    for i in range(0, len(monitor_macro_f1)):
        print(monitor_macro_f1[i][0], monitor_macro_f1[i][1], monitor_macro_f1[i][2], monitor_macro_f1[i][3])

    print('weighted')
    for i in range(0, len(monitor_weighted_f1)):
        print(monitor_weighted_f1[i][0], monitor_weighted_f1[i][1], monitor_weighted_f1[i][2], monitor_weighted_f1[i][3])'''

    print('per class distribution')
    #for i in range(0, len(monitor_class_distribution)):
    print(monitor_class_distribution[0])

    print('per class')
    for i in range(0, len(monitor_per_class_f1)):
        print(monitor_per_class_f1[i])

    print('average')
    print(list(np.mean(monitor_per_class_f1, axis=0)))
    print('std')
    print(list(np.std(monitor_per_class_f1, axis=0)))

    print('group 0 distribution')
    #for i in range(0, len(monitor_group_0_percentage)):
    print(monitor_group_0_percentage[0])

    print('group 0 per class')
    for i in range(0, len(monitor_per_class_group_0_f1)):
        print(monitor_per_class_group_0_f1[i])

    print('average')
    print(list(np.mean(monitor_per_class_group_0_f1, axis=0)))
    print('std')
    print(list(np.std(monitor_per_class_group_0_f1, axis=0)))

    print('group 1 distribution')
    #for i in range(0, len(monitor_group_1_percentage)):
    print(monitor_group_1_percentage[0])

    print('group 1 per class')
    for i in range(0, len(monitor_per_class_group_1_f1)):
        print(monitor_per_class_group_1_f1[i])

    print('average')
    print(list(np.mean(monitor_per_class_group_1_f1, axis=0)))
    print('std')
    print(list(np.std(monitor_per_class_group_1_f1, axis=0)))
    #print('group distribution')
    #for i in range(0, len(monitor_group_percentage)):
    #    print(monitor_group_percentage[i])

    #print(accumulate_time)