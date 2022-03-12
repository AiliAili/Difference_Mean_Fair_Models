import logging
from typing import Dict

import numpy as np

import torch
import torch.utils.data as data
from collections import defaultdict, Counter

class DeepMojiDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, split, balanced, balance_type, shuffle=None, weight_scheme=None, ratio: float = 0.5, n: int = 100000):
        self.args = args
        self.data_dir = data_dir
        self.dataset_type = {"train", "dev", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, dev, and test"
        self.split = split
        self.data_dir = self.data_dir+self.split
        self.ratio = ratio
        self.n = n
        # Init 
        self.X = []
        self.y = []
        self.private_label = []
        self.weight = []
        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed deepMoji Encoded data")
        self.load_data()

        if balanced == True:
            assert balance_type in ["y", "g", "ratio", "class", "stratified"], "not implemented yet"
            """
                ratio:  according to its y,g combination, P(y,g)
                y:      according to its main task label y only, P(y)
                g:      according to its protected label g only, P(g)
                class:  according to its protected label within its main task label, P(g|y)
                stratified: keep the y distribution and balance g within y
            """
            
            # init a dict for storing the index of each group.

            group_idx = {}
            if balance_type == "ratio":
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
            elif balance_type == "y":
                group_labels = self.y
            elif balance_type == "g":
                group_labels = self.private_label
            elif balance_type == "class":
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
            elif balance_type == "stratified":
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
            else:
                pass

            for idx, group_label in enumerate(group_labels):
                group_idx[group_label] = group_idx.get(group_label, []) + [idx]


            #print(group_idx)
            selected_index = []
            if balance_type in ["ratio", "y", "g"]:
                # selected = min(len(man_idx), len(woman_idx))
                selected = min([len(i) for i in group_idx.values()])

                for index in group_idx.values():
                    _index = index
                    shuffle(_index)
                    selected_index = selected_index + _index[:selected]
            elif balance_type == "class":
                # balance protected groups with respect to each main task class

                # iterate each main task class
                for y in set(self.y):
                    # balance the protected group distribution
                    y_group_idx = [group_idx[(y, g)] for g in set(self.private_label)]
                    y_selected = min([len(i) for i in y_group_idx])
                    for index in y_group_idx:
                        _index = index
                        shuffle(_index)
                        selected_index = selected_index + _index[:y_selected]
            elif balance_type == "stratified":
                # empirical distribution of y
                weighting_counter = Counter(self.y)

                # a list of (weights, actual length)
                condidate_selected = min([len(group_idx[k])/weighting_counter[k[0]] for k in group_idx.keys()])

                distinct_y_label = set(self.y)
                distinct_g_label = set(self.private_label)
                
                # iterate each main task class
                for y in distinct_y_label:
                    selected = int(condidate_selected * weighting_counter[y])
                    for g in distinct_g_label:
                        _index = group_idx[(y,g)]
                        shuffle(_index)
                        selected_index = selected_index + _index[:selected]


            X = [self.X[index] for index in selected_index]
            self.X = X
            y = [self.y[index] for index in selected_index]
            self.y = y
            private_label = [self.private_label[index] for index in selected_index]
            self.private_label = private_label

        elif balanced == False:
            group_idx = {}
            inverse_weights ={}

            if weight_scheme == 'joint':
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
                for idx, group_label in enumerate(group_labels):
                    group_idx[group_label] = group_idx.get(group_label, []) + [idx]

                for tem in group_idx:
                    inverse_weights[tem] = len(self.y)/float(len(group_idx[tem]))
                
                for i in range(0, len(self.y)):
                    self.weight.append(inverse_weights[(self.y[i], self.private_label[i])])

            else:
                for i in range(0, len(self.y)):
                    self.weight.append(1.0)

            #print(inverse_weights)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)
        if len(self.weight) == 0:
            self.weight = [1]*len(self.y)
        self.weight = np.array(self.weight)
        #print(self.weight[:200])
        #print(Counter(self.y))
        #print(Counter(self.private_label))
        #print(Counter(self.y))
        group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
        print(Counter(group_labels))
        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.private_label.shape, self.weight.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.private_label[index], self.weight[index]
    
    def load_data(self):
        """
        Based on https://github.com/shauli-ravfogel/nullspace_projection/blob/2403d09a5d84b3ae65129b239331a22f89ad69fc/src/deepmoji/deepmoji_debias.py#L24
        """
        # ratios for pos / neg
        n_1 = int(self.n * self.ratio*self.args.positive_class_ratio)
        n_2 = int(self.n * (1.0 - self.ratio)*self.args.positive_class_ratio)

        n_3 = int(self.n * self.ratio*(1.0-self.args.positive_class_ratio))
        n_4 = int(self.n * (1.0 - self.ratio)*(1.0-self.args.positive_class_ratio))
        #print(n_1, n_2)
        if self.args.dataset_mode == 'gender':
            tem = [n_1, n_2, n_2, n_1]
        else:
            tem = [n_1, n_2, n_4, n_3]

        print(tem)
        for file, label, private, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                [1, 1, 0, 0],
                                                [1, 0, 1, 0],
                                                #[n_1, n_2, n_4, n_3]):
                                                #[n_1, n_2, n_2, n_1]):
                                                tem):
            data = np.load('{}/{}.npy'.format(self.data_dir, file))
            #print(data.shape)
            data = list(data[:class_n])
            #print(len(data))
            self.X = self.X + data
            self.y = self.y + [label]*len(data)
            self.private_label = self.private_label + [private]*len(data)
