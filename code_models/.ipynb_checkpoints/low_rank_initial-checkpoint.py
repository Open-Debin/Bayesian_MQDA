import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pdb
import os

seed = 807
print('random seed: '.format(seed))
import sklearn
print("Sklearn verion is {}".format(sklearn.__version__))

def setup_seed(seed):
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# python Low-Rank_Initial.py
device = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = device
class Low_Rank:
    def __init__(self, rank=1,feature_dim=64):
        self.S = torch.rand(rank,feature_dim,dtype=torch.float32).cuda()
        self.S .requires_grad = True
        
    def forward(self):
        matrix_ = torch.mm(self.S.permute(1,0),self.S)
        base = matrix_.sum().detach()/32.
        return matrix_
    
    def parameters(self):
        yield self.S

class update_lr:
    def __init__(self):
        self.loss = 0
        self.n = 0
    def compare_loss(self,loss, epoch, optimizer):
        self.epoch = epoch
        self.optimizer = optimizer

        
        if (loss < self.loss)**2 < 1e-4:
            self.n += 1
        else:
            self.loss= loss
            self.n = 0   
        
        if self.n >=22:
            self.update()
            self.n = 0
            self.loss = 0
        
    
    def update(self):
    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1

        print('Adjust_learning_rate ' + str(self.epoch))
        print('New_LearningRate: {}'.format(param_group['lr']))

def main(rank=1,feature_dim=2):

    net = Low_Rank(rank=rank,feature_dim=feature_dim)
    label = torch.tensor(np.eye(feature_dim), dtype=torch.float32).cuda()
                                                        
    up = update_lr()
    criterion = nn.MSELoss()
#     pdb.set_trace()
#     temp = [e for e in net.parameters()]
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    cudnn.benchmark = True

#     while True:
    for i in range(300):    
        optimizer.zero_grad()
        outputs = net.forward()

        loss = criterion(outputs, label)
        up.compare_loss(loss.item(),i,optimizer)
        loss.backward()
        optimizer.step()

#         print('loss',loss.item())
#         print(outputs)
        
    temp = [e for e in net.parameters()]
    temp = temp[0].detach()
    print('rank_'+str(rank)+'.tar')
    torch.save(temp,'rank_'+str(rank)+'.tar')

    
if __name__=='__main__':
    for rank in [4, 8, 16, 32]:
        main(rank=rank,feature_dim=64)
