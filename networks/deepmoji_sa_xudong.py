import torch
import torch.optim as optim
import torch.nn as nn
from networks.discriminator import Discriminator
import torch.nn.functional as F

class DeepMojiModel(nn.Module):
    def __init__(self, args):
        super(DeepMojiModel, self).__init__()
        self.args = args
        self.emb_size = self.args.emb_size
        self.hidden_size = self.args.hidden_size
        self.num_classes = self.args.num_classes
        self.adv_level = self.args.adv_level

        self.drop = nn.Dropout(p=0.5)

        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        self.AF = nn.Tanh()
        try:
            if args.AF == "relu":
                self.AF = self.ReLU
            elif args.AF == "tanh":
                self.AF = self.tanh
        except:
            pass
        self.dense1 = nn.Linear(self.emb_size, self.hidden_size)
        #self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense2 = [nn.Linear(self.hidden_size, self.hidden_size).to('cuda:0') for _ in range (2)]
        self.dense3 = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, input):
        
        out = self.dense1(input)
        out = self.AF(out)
        #out = self.drop(out)
        out1 = F.normalize(out, dim=1)
        #out = self.dense2(out)
        #out = self.tanh(out)

        for hl in self.dense2:
            out=hl(out)
            out=self.tanh(out)        

        #out = self.tanh(self.dense4(out))
        
        out2 = F.normalize(out, dim=1)
        second_last = out
        #out = self.dense3(out2)
        out = self.dense3(out)
        #original
        return out, out1, out2, second_last
        #return out, out1, out2, out2#second_last
    
    def hidden(self, input):
        assert self.adv_level in set([0, -1, -2])
        out = self.dense1(input)
        out = self.AF(out)
        if self.adv_level == -2:
            return out
        else:
            #out = self.dense2(out)
            #out = self.tanh(out)
            for hl in self.dense2:
                out=hl(out)
                out=self.tanh(out)

            if self.adv_level == -1:
                #out = F.normalize(out, dim=1)
                return out
            else:
                out = self.dense3(out)
                return out