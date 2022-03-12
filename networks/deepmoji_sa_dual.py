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
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_classes)


        self.dense1_1 = nn.Linear(self.emb_size, self.hidden_size)
        self.dense2_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3_1 = nn.Linear(self.hidden_size, self.num_classes)


        self.dense1_dual = nn.Linear(self.emb_size, self.hidden_size)
        self.dense2_dual = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3_dual = nn.Linear(self.hidden_size, self.num_classes)
        #self.dense3_dual_1 = [nn.Linear(30, 2).to('cuda:0') for _ in range (10)]
    def forward(self, input):
        
        out = self.dense1(input)
        out = self.AF(out)
        #out = self.drop(out)
        out1 = F.normalize(out, dim=1)
        out = self.dense2(out)
        out = self.tanh(out)

        out2 = F.normalize(out, dim=1)
        second_last = out
        #out = self.dense3(out2)
        #out = nn.ReLU()(self.dense4(out))
        out = self.dense3(out)

        out_dual = self.dense1_dual(input)
        out_dual = self.AF(out_dual)
        #out = self.drop(out)
        out1_dual = F.normalize(out_dual, dim=1)
        out_dual = self.dense2_dual(out_dual)
        out_dual = self.tanh(out_dual)

        out2_dual = F.normalize(out_dual, dim=1)
        second_last_dual = out_dual

        '''out_dual_tem = []
        counter = 0
        for layer in self.dense3_dual_1:
            tem=layer(out_dual[:,30*counter: 30*(counter+1)]) 
            out_dual_tem.append(tem)
            counter+=1'''

        #out = self.dense3(out2)
        #out = nn.ReLU()(self.dense4(out))
        out_dual = self.dense3_dual(out_dual)


        out_tem = self.dense1_1(input)
        out_tem = self.AF(out_tem)
        out_tem = self.dense2_1(out_tem)
        out_tem_1 = self.tanh(out_tem)
        out_tem = self.dense3_1(out_tem_1)
        #original
        return out, out1, out2, second_last, out_dual, out1_dual, out2_dual, second_last_dual, out_tem, out_tem_1#, out_dual_tem
    
    def hidden(self, input):
        assert self.adv_level in set([0, -1, -2])
        out = self.dense1(input)
        out = self.AF(out)
        if self.adv_level == -2:
            return out
        else:
            out = self.dense2(out)
            out = self.tanh(out)
            '''for hl in self.dense2:
                out=hl(out)
                out=self.tanh(out)'''

            if self.adv_level == -1:
                #out = F.normalize(out, dim=1)
                return out
            else:
                out = self.dense3(out)
                return out