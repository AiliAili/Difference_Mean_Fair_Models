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

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataloaders.bios_dataset_majority import BiosDataset
#from dataloaders.bios_dataset_binary import BiosDataset
from networks.deepmoji_sa import DeepMojiModel
#from networks.deepmoji_sa_xudong import DeepMojiModel
from networks.discriminator import Discriminator
from dataloaders.scheduler import BalancedBatchSampler
from dataloaders.FairBatchSampler_BIOS_majority import FairBatch


from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss
from networks.contrastive_loss import Contrastive_Loss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.eval_metrices_updated import leakage_evaluation, tpr_multi, leakage_hidden, leakage_logits
from collections import defaultdict, Counter

from pathlib import Path, PureWindowsPath
from collections import defaultdict, Counter
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import torch.nn.functional as F
import argparse
from sklearn.linear_model import SGDClassifier, LogisticRegression
import time
import operator

monitor_micro_f1 = []
monitor_macro_f1 = []
monitor_weighted_f1 = []

monitor_class_distribution = []
monitor_per_class_f1 = []
monitor_group_0_percentage = []
monitor_group_1_percentage = []
monitor_per_class_group_0_f1 = []
monitor_per_class_group_1_f1 = []


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
def train_epoch(model, discriminators, iterator, optimizer, criterion, contrastive_loss, contrastive_loss_2, device, args):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for discriminator in discriminators:
        discriminator.train()

    # activate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = True
    
    for batch in iterator:
        
        text = batch[0]
        tags = batch[1].long()
        p_tags = batch[2].long()
        weights = batch[3]

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
        predictions, features_1, features_2, _ = model(text)
        # main tasks loss
        loss = criterion(predictions, tags)
        
        if args.mode == 'vanilla':
            loss = criterion(predictions, tags)
        elif args.mode == 'rw':
            loss = (criterion(predictions, tags)*weights).mean()
        elif args.mode == 'ds':
            loss = criterion(predictions, tags) 
        elif args.mode == 'difference':
            loss = criterion(predictions, tags) 
            for i in range(0, 28):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    loss+= args.lambda_weight*abs(tem_0-tem_1)#0.01        
        elif args.mode == 'mean':
            loss = criterion(predictions, tags)
            accu_loss = 0
            for i in range(0, args.num_classes):
                for j in range(0, 2):
                    indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                    indices = list(indices)
                    if len(indices) > 0:
                        loss_c_g = criterion(predictions[indices], tags[indices])
                        accu_loss+=args.lambda_weight*abs(loss_c_g-loss)#0.005

            loss+=accu_loss
        
        elif args.mode == 'fairbatch':
            loss = criterion(predictions, tags)
        elif args.mode == 'difference+max':
            loss = criterion(predictions, tags) 
            for i in range(0, 28):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    loss+= args.lambda_weight*max(tem_0, tem_1)  
        elif args.mode == 'difference+min':
            loss = criterion(predictions, tags) 
            for i in range(0, 28):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    loss-= args.lambda_weight*min(tem_0, tem_1)
        elif args.mode == 'mean+max':
            loss = criterion(predictions, tags)
            accu_loss = 0
            for i in range(0, args.num_classes):
                for j in range(0, 2):
                    indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                    indices = list(indices)
                    if len(indices) > 0:
                        loss_c_g = criterion(predictions[indices], tags[indices])
                        accu_loss+=args.lambda_weight*max(loss_c_g, loss)#0.005
            loss+=accu_loss      
        elif args.mode == 'mean+min':
            loss = criterion(predictions, tags)
            accu_loss = 0
            for i in range(0, args.num_classes):
                for j in range(0, 2):
                    indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                    indices = list(indices)
                    if len(indices) > 0:
                        loss_c_g = criterion(predictions[indices], tags[indices])
                        accu_loss+=args.lambda_weight*min(loss_c_g, loss)#0.005
            loss-=accu_loss     
        elif args.mode == 'fairbatch+difference':
            loss = criterion(predictions, tags) 
            for i in range(0, 28):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    loss+= args.lambda_weight*abs(tem_0-tem_1)#0.01   

        if args.adv:
            # discriminator predictions
            p_tags = p_tags.long()

            hs = model.hidden(text)

            for discriminator in discriminators:
                adv_predictions = discriminator(hs)
            
                loss = loss + (criterion(adv_predictions, p_tags) / len(discriminators))
                        
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
        
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

        predictions, features_1, features_2, _ = model(text)

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
    data_frame['profession_class'] = labels
    data_frame['gender_class'] = private_labels
    data_frame['second_last_representation'] = second_last_representation
    data_frame['predict'] = tem_preds
    data_frame.to_pickle(filename)
    accuracy = accuracy_score(labels, tem_preds)
    print('Potential', accuracy)

    X_logits = list(data_frame['prob'])
    X_hidden = list(data_frame['second_last_representation'])
    y = list(data_frame['profession_class'])
    gender_label = list(data_frame['gender_class'])

    return (X_logits, X_hidden, y, gender_label)

def load_leakage_data(filename):
    data = pd.read_pickle(filename)
    X_logits = list(data['prob'])
    X_hidden = list(data['second_last_representation'])
    y = list(data['profession_class'])
    gender_label = list(data['gender_class'])
        
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

    #print(len(preds_0), len(preds_1), accuracy_0, accuracy_1)

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
    p_labels = train_data.gender_label
    for i in range(0, len(p_labels)):
        if p_labels[i] == 0:
            labels_0_train.append(train_data.y[i])
        else:
            labels_1_train.append(train_data.y[i])


    #f1_per_class_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    matrix = confusion_matrix(labels_0, preds_0)
    f1_per_class_group_0 = 100*matrix.diagonal()/matrix.sum(axis=1)

    overall_counter = Counter(train_data.y)
    group_0_counter = Counter(labels_0_train)
    for tem in group_0_counter:
        group_0_counter[tem] = group_0_counter[tem]/float(overall_counter[tem])
    
    monitor_per_class_group_0_f1.append([])
    monitor_group_0_percentage.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_0_f1[-1].extend([f1_per_class_group_0[i]])
        monitor_group_0_percentage[-1].append(100*group_0_counter[i])


    #f1_per_class_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    matrix = confusion_matrix(labels_1, preds_1)
    f1_per_class_group_1 = 100*matrix.diagonal()/matrix.sum(axis=1)

    counter = Counter(labels_1_train)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(labels_1_train))

    group_1_counter = Counter(labels_1_train)
    for tem in group_1_counter:
        group_1_counter[tem] = group_1_counter[tem]/float(overall_counter[tem])
    
    monitor_per_class_group_1_f1.append([])
    monitor_group_1_percentage.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_1_f1[-1].extend([f1_per_class_group_1[i]])
        monitor_group_1_percentage[-1].append(100*group_1_counter[i])


    return abs(accuracy_0-accuracy_1), min(accuracy_0, accuracy_1), 0.5*(accuracy_0+accuracy_1), minority


def get_group_metrics_1(preds, labels, p_labels):

    f1_per_class = 100*f1_score(labels, preds, average=None)
    counter = Counter(labels)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(labels))

    sorted_d = sorted(counter.items(), key=operator.itemgetter(1))
    counter = sorted_d
    monitor_per_class_f1.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_f1[-1].extend([counter[i][1], f1_per_class[counter[i][0]]])


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

def get_TPR(y_pred, y_true, i2p, i2g, gender, counter):
    
    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)
    
    for y_hat, y, g in zip(y_pred, y_true, gender):
        
        if y == y_hat:
            scores[i2p[y]][i2g[g]] += 1
        
        prof_count_total[i2p[y]][i2g[g]] += 1
    #print(scores)
    #print(prof_count_total)
    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []
    
    for profession, scores_dict in scores.items():
        #print(profession, scores_dict)
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        tpr_m = 100*(good_m) /float(prof_total_m)
        tpr_f = 100*(good_f) /float(prof_total_f)
        
        tprs[profession]["m"] = tpr_m
        tprs[profession]["f"] = tpr_f
        tprs_ratio.append(0)
        tprs_change[profession] = tpr_f - tpr_m
        #print(profession, (good_m+good_f)/float(prof_total_m+prof_total_f))
    
    value = []
    weighted_value = []
    for profession in tprs_change:
        value.append(tprs_change[profession]**2)
        weighted_value.append(counter[profession]*(tprs_change[profession]**2))

    #return tprs, tprs_change, np.mean(np.abs(tprs_ratio)) 
    return np.sqrt(np.mean(value)), np.sqrt(np.mean(weighted_value)), tprs_change

def load_dictionary(path):
    
    with open(path, "r", encoding = "utf-8") as f:
        
        lines = f.readlines()
        
    k2v, v2k = {}, {}
    for line in lines:
        
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    
    return k2v, v2k

def woman_profession_portion(train_data, dev_data, test_data, i2p, i2g):
    counter = defaultdict(Counter)
    for i in range(0, len(train_data.y)):
        profession = i2p[train_data.y[i]]
        gender = i2g[train_data.gender_label[i]]
        counter[profession][gender]+=1

    for i in range(0, len(dev_data.y)):
        profession = i2p[dev_data.y[i]]
        gender = i2g[dev_data.gender_label[i]]
        counter[profession][gender]+=1

    for i in range(0, len(test_data.y)):
        profession = i2p[test_data.y[i]]
        gender = i2g[test_data.gender_label[i]]
        counter[profession][gender]+=1
    
    prof2fem = dict()
    for k, values in counter.items():
        prof2fem[k] = values['f']/float((values['f']+values['m']))

    return prof2fem

def correlation_plot(tprs, prof2fem, filename):
    professions = list(tprs.keys())
    tpr_lst = [tprs[p] for p in professions]
    proportion_lst = [prof2fem[p] for p in professions]
    plt.plot(proportion_lst, tpr_lst, marker = "o", linestyle = "none")
    plt.xlabel("% women", fontsize = 13)
    plt.ylabel(r'$GAP_{female,y}^{TPR}$', fontsize = 13)

    for p in professions:
        x,y = prof2fem[p], tprs[p]
        plt.annotate(p , (x,y), size = 7, color = "red")
    
    #plt.ylim(-0.4, 0.55)
    z = np.polyfit(proportion_lst, tpr_lst, 1)
    p = np.poly1d(z)
    plt.plot(proportion_lst,p(proportion_lst),"r--")
    plt.savefig("./tem1.png", dpi = 600)
    print("Correlation: {}; p-value: {}".format(*pearsonr(proportion_lst, tpr_lst)))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 768)
    parser.add_argument('--num_classes', type=int, default = 8)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--adv_level', type=int, default = -1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--starting_power', type=int)
    parser.add_argument('--LAMBDA', type=float, default=0.8)
    parser.add_argument('--n_discriminator', type=int, default = 0)
    parser.add_argument('--adv_units', type=int, default = 256)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--DL', action='store_true')
    parser.add_argument('--diff_LAMBDA', type=float, default=1000)
    parser.add_argument('--data_path', type=str, default='/data/scratch/projects/punim0478/xudongh1/data/bios/', help='directory containing the dataset')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default = 1024)    
    parser.add_argument('--seed', type=int, default = 46)
    parser.add_argument('--representation_file', default='./analysis/ce+scl1+scl2.txt', type=str, help='the file storing test representation before the classifier layer')
    parser.add_argument('--loss_type', default='ce', type=str, help='the type of loss we want to use')
    parser.add_argument('--lambda_weight', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--num_epochs', type=int, default = 5)
    parser.add_argument('--device_id', type=int, default = 0)
    parser.add_argument('--experiment_type', default='adv_experiment_lambda', type=str, help='which types of experiments we are doing')
    parser.add_argument('--balance_type', default='stratified', type=str, help='which types of experiments we are doing')
    parser.add_argument('--mode', default='vanilla', type=str, help='which types of experiments we are doing')
    parser.add_argument('--data_manipulation_mode', default='vanilla', type=str, help='in which ways we should manipulate the dataset (not the imbalance method)')


    args = parser.parse_args()

    accumulate_accuracy = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_f1_macro = []
    accumulate_f1_micro = []
    accumulate_f1_weighted = []

    #p2i, i2p = load_dictionary("./profession2index.txt")
    g2i, i2g = load_dictionary("./gender2index.txt")
    i2p = {0: 'teacher', 1: 'psychologist', 2: 'nurse', 3: 'journalist', 4: 'photographer', 5: 'attorney', 6: 'physician', 7: 'professor'}

    #i2p = {0: 'nurse', 1: 'surgeon'}
    #batch_list = [256, 512, 1024, 2048, 4096]
    #lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    #lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
    #lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
    #lambda_1_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
    #lambda_2_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
    batch_list = [256, 512, 1024, 2048]
    lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    #lr_list = [7e-5]
    lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]

    adv_batch_list = [256, 512, 1024, 2048]
    adv_lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    adv_lambda_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    adv_diff_lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

    accumulate_acc = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    count_runs = 0

    accumulate_time = []
    #output_file = open(args.representation_file, 'w')
    #for tem_lambda in [1e-5, 1e-4, 1e-3, 1e-2]:
    #for tem_lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, valid_acc0.8, 0.9, 1.0]:
    #for batch_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    #for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
    #for tem_batch in batch_list:
    #for tem_1 in lambda_1_list:
    #    for tem_2 in lambda_2_list:

    #for tem_batch in adv_batch_list:
    #    for tem_lr in adv_lr_list:
    #        for tem_lambda in adv_lambda_list:
    #for tem_batch in batch_list:
    #    for tem_lr in lr_list:
    #        for tem_lambda in lambda_ratio_list:  
    selected_lambda = args.lambda_weight
    if True:
        if True:
        #for tem_lambda in [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            #for tem_alpha in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]: #[1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.06,0.07,0.08,0.09,0.1,0.5,1]:
            #for tem_lambda in adv_diff_lambda_list:
            #for tem_lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            #for tem_lambda in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]:
            #for tem_lambda in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
            #for tem_lambda in lambda_ratio_list:
            #for tem_temparature in [0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
            #for tem_seed in [40]:
            #for tem_lambda in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:#0.01, 
            #for tem_lambda in adv_lambda_list:
            #for tem_alpha in [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.06,0.07,0.08,0.09,0.1,0.5,1]:
            #if True:
                print('============================================================')
                #args.ratio = tem_lambda
                #args.seed = tem_seed

                tem_lambda = tem_seed
                args.seed = tem_seed
                #args.lambda_weight = tem_lambda
                #tem_lambda = args.lambda_weight

                #tem_lambda = tem_alpha
                #args.lambda_weight = tem_alpha

                print('============================================================')

                print('bios batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.ratio, args.seed)
                #print('batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.lambda_1, args.lambda_2)
                seed_everything(args.seed)
                #args.lambda_weight = 0.5

                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                main_model_path = "./explore/bios_model_majority_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(args.data_manipulation_mode, args.mode, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.ratio, args.seed)
                #main_model_path = "./biographyFull_adv_model.pt"
                adv_model_path = "./explore/bios_discriminator_{}_{}_{}_{}.pt"

                # Device
                device = torch.device("cuda:"+str(args.device_id))
                # Init model
                model = DeepMojiModel(args)
                model = model.to(device)

                balance_type = args.balance_type

                # Load data
                if args.mode == 'ds' or args.mode == 'ds+difference' or args.mode == 'ds+mean':
                    balance_flag = True
                else:
                    balance_flag = False
                
                train_data = BiosDataset(args.data_path, "train", embedding_type = 'cls', balanced=balance_flag, balance_type=balance_type, manipulation_mode=args.data_manipulation_mode, weight_scheme='joint', shuffle=shuffle, ratio=args.ratio)
                
                dev_data = BiosDataset(args.data_path, "dev", embedding_type = 'cls',  balanced=False, balance_type=balance_type, manipulation_mode='vanilla', shuffle=shuffle)
                test_data = BiosDataset(args.data_path, "test", embedding_type = 'cls', balanced=False, balance_type=balance_type, manipulation_mode='vanilla', shuffle=shuffle)

                prof2fem = woman_profession_portion(train_data, dev_data, test_data, i2p, i2g)


                # Data loader
                if args.mode in ['fairbatch', 'fairbatch+difference']:
                    sampler = FairBatch(model, torch.tensor(train_data.X).to(device), torch.tensor(train_data.y).long().to(device), torch.tensor(train_data.gender_label).long().to(device), batch_size = args.batch_size, alpha=args.lambda_weight, target_fairness = 'eqopp', replacement = False, seed = args.seed)
                    training_generator = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)
                else:
                    training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
                
                #training_generator = torch.utils.data.DataLoader(train_data, batch_sampler=balanced_batch_sampler)
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
                contrastive_loss_2 = Contrastive_Loss(device=device, temperature=1*args.temperature, base_temperature= 1*args.temperature)

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

                for i in trange(args.num_epochs):
                    train_epoch(
                                model = model, 
                                discriminators = discriminators, 
                                iterator = training_generator, 
                                optimizer = optimizer, 
                                criterion = criterion, 
                                contrastive_loss = contrastive_loss,
                                contrastive_loss_2 = contrastive_loss_2,
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
                    #print('Valid loss', valid_loss, 'Valid acc', valid_acc, best_epoch, i, args.loss_type)

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
                #print('load model')
                        
                '''get_leakage_data(model, training_generator, './inlp_input/bios_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, validation_generator, './inlp_input/bios_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, test_generator, './inlp_input/bios_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                train_leakage_data = load_leakage_data('./inlp_input/bios_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                val_leakage_data = load_leakage_data('./inlp_input/bios_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                test_leakage_data = load_leakage_data('./inlp_input/bios_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))'''
                
                training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                train_leakage_data = get_leakage_data(model, training_generator, './output/bios_train.pickle', device, args)
                #val_leakage_data = get_leakage_data(model, validation_generator, './inlp_input/bios_val_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                #test_leakage_data = get_leakage_data(model, test_generator, './inlp_input/bios_test_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/bios_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/bios_test.pickle', device, args)  

                # Evaluation
                test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                preds = np.array(preds)
                labels = np.array(labels)
                p_labels = np.array(p_labels)      


                accuracy = accuracy_score(labels, preds)     

                micro_f1 = 100*f1_score(labels, preds, average='micro')
                monitor_micro_f1.append([micro_f1])
                macro_f1 = 100*f1_score(labels, preds, average='macro')
                monitor_macro_f1.append([macro_f1])
                weighted_f1 = 100*f1_score(labels, preds, average='weighted')
                monitor_weighted_f1.append([weighted_f1])       

                test_data = BiosDataset(args.data_path, "test", embedding_type = 'cls', balanced=False, balance_type=balance_type, manipulation_mode='vanilla', shuffle=shuffle)
                tem_counter = Counter(test_data.y)

                for tem in tem_counter:
                    tem_counter[tem] = tem_counter[tem]/float(len(test_data))

                counter = dict()
                for tem in tem_counter:
                    counter[i2p[tem]] = tem_counter[tem]

                #print(counter)

                rms_diff, weighted_rms_diff, tprs = tpr_multi(preds, labels, i2p, i2g, p_labels, counter)
                #for tem in tprs:
                #    print(tem, abs(tprs[tem]))

                print('rms diff', rms_diff, 'weighted rms diff', weighted_rms_diff)
                logits_leakage = leakage_logits(train_leakage_data, val_leakage_data, test_leakage_data)
                hidden_leakage = leakage_hidden(train_leakage_data, val_leakage_data, test_leakage_data)
                accumulate_rms_diff.append(rms_diff)
                accumulate_weighted_rms_diff.append(weighted_rms_diff)
                accumulate_leakage_logits.append(logits_leakage[1])
                accumulate_leakage_hidden.append(hidden_leakage[1])

                difference, min_performance, macro_average, minority_performance = get_group_metrics(preds, labels, p_labels, train_data)
                accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff, difference, min_performance, macro_average, minority_performance])
                #correlation_plot(tprs, prof2fem, 'hello')

                count_runs+=1
                print('hello world', count_runs, datetime.now())
                #print(accumulate_time[count_runs-1])
                #break
                

    #output_file.close()
    print(balance_type)
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'micro', monitor_micro_f1[i][0], 'macro', monitor_macro_f1[i][0], 'weighted', monitor_weighted_f1[i][0], accumulate_acc[i][7], accumulate_acc[i][8], )#accumulate_acc[i][9], accumulate_acc[i][10], accumulate_acc[i][11], accumulate_acc[i][12])    
    
    exit()
    print('micro')
    for i in range(0, len(monitor_micro_f1)):
        print(monitor_micro_f1[i][0], monitor_micro_f1[i][1], monitor_micro_f1[i][2], monitor_micro_f1[i][3])

    print('macro')
    for i in range(0, len(monitor_macro_f1)):
        print(monitor_macro_f1[i][0], monitor_macro_f1[i][1], monitor_macro_f1[i][2], monitor_macro_f1[i][3])

    print('weighted')
    for i in range(0, len(monitor_weighted_f1)):
        print(monitor_weighted_f1[i][0], monitor_weighted_f1[i][1], monitor_weighted_f1[i][2], monitor_weighted_f1[i][3])


    print('per class distribution')
    print(monitor_class_distribution[0])


    print('per class')
    for i in range(0, len(monitor_per_class_f1)):
        print(monitor_per_class_f1[i])

    print('average')
    print(list(np.mean(monitor_per_class_f1, axis=0)))
    print('std')
    print(list(np.std(monitor_per_class_f1, axis=0)))


    print('group 0 distribution')
    print(monitor_group_0_percentage[0])


    print('group 0 per class')
    for i in range(0, len(monitor_per_class_group_0_f1)):
        print(monitor_per_class_group_0_f1[i])

    print('average')
    print(list(np.mean(monitor_per_class_group_0_f1, axis=0)))
    print('std')
    print(list(np.std(monitor_per_class_group_0_f1, axis=0)))

    print('group 1 distribution')
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

    output_file = open(args.representation_file, 'w')
    output_file.write(str(args.data_manipulation_mode)+'  '+args.mode+'\n')
    output_file.write('per class distribution\n')
    output_file.write(','.join([str(t) for t in monitor_class_distribution[0]])+'\n')
    output_file.write('per class\n')
    for i in range(0, len(monitor_per_class_f1)):
        output_file.write(','.join([str(t) for t in monitor_per_class_f1[i]])+'\n')
    
    output_file.write('average\n')
    tem = list(np.mean(monitor_per_class_f1, axis=0))
    output_file.write(','.join([str(t) for t in tem])+'\n')
    output_file.write('std\n')
    tem = list(np.std(monitor_per_class_f1, axis=0))
    output_file.write(','.join([str(t) for t in tem])+'\n')

    output_file.write('group 0 distribution\n')
    output_file.write(','.join([str(t) for t in monitor_group_0_percentage[0]])+'\n')

    output_file.write('group 0 per class\n')
    for i in range(0, len(monitor_per_class_group_0_f1)):
        output_file.write(','.join([str(t) for t in monitor_per_class_group_0_f1[i]])+'\n')

    output_file.write('average\n')
    tem = list(np.mean(monitor_per_class_group_0_f1, axis=0))
    output_file.write(','.join([str(t) for t in tem])+'\n')
    output_file.write('std\n')
    tem = list(np.std(monitor_per_class_group_0_f1, axis=0))
    output_file.write(','.join([str(t) for t in tem])+'\n')


    output_file.write('group 1 distribution\n')
    output_file.write(','.join([str(t) for t in monitor_group_1_percentage[0]])+'\n')
    output_file.write('group 1 per class\n')
    for i in range(0, len(monitor_per_class_group_1_f1)):
        output_file.write(','.join([str(t) for t in monitor_per_class_group_1_f1[i]])+'\n')


    output_file.write('average\n')
    tem = list(np.mean(monitor_per_class_group_1_f1, axis=0))
    output_file.write(','.join([str(t) for t in tem])+'\n')
    output_file.write('std\n')
    tem = list(np.std(monitor_per_class_group_1_f1, axis=0))
    output_file.write(','.join([str(t) for t in tem])+'\n')
    output_file.close()