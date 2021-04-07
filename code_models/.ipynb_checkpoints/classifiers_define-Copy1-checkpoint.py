import pdb
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from basic_code import networks
from basic_code import util, load_materials

class MetaQDA(nn.Module):
    def __init__(self, reg_param=0.3, input_process='UN', inverse_reg='Inverse_Regular', bool_autoft=False, cholesky_alpha=False, prior='Gaussian', fea_dim = 512):
        super(MetaQDA, self).__init__()
        if inverse_reg not in ['Inverse_Regular','Regular_Inverse'] :
            raise ValueError(f'inverse_reg should be Inverse_Regular or Regular_Inverse')
        self._cholesky_alpha =cholesky_alpha # cholesky_alpha for loss
        self._bool_inv_reg = inverse_reg
        self.reg_param = reg_param
        self._preprocess_feature = input_process
        self._bool_autoft = bool_autoft # True: learning shift and scale of features than has been processed by L2N or CL2N preprocess feature
        
        # MetaQDA Parameters
        _num_images=600
        self.feature_dim = fea_dim

        gamma_m = float(1 / _num_images)
        gamma_S = 1 
        self.m = torch.nn.Parameter(torch.tensor( np.zeros((self.feature_dim,),)*gamma_m, dtype=torch.float32))
        self.S = torch.nn.Parameter(torch.tensor(np.eye(self.feature_dim)*gamma_S, dtype=torch.float32))
        self.nu = torch.nn.Parameter(torch.tensor(float(self.feature_dim)*gamma_S, dtype=torch.float32))
        self.kappa = torch.nn.Parameter(torch.tensor(1*gamma_m, dtype=torch.float32))
        self.diagonal_param = torch.nn.Parameter(torch.ones(self.feature_dim))
        
        self.diag_triu='abs' # origin, log, abs
        if self.diag_triu in ['abs']:
            self.triu_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=1).t().cuda()
            self.diagonal_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=0).cuda()
        if self.diag_triu == 'origin':
            self.triu_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=0).cuda().t()

        # AutoFeatureTransformer (AutoFT) Parameters
#         self.ft_scale = torch.nn.Parameter(torch.ones(1)*0.3) # scale parameter
#         self.ft_shift  = torch.nn.Parameter(torch.ones(1)*0.5) # shift parameter
#         self.ft_scale = torch.nn.Parameter(torch.ones(1)) # scale parameter
#         self.ft_shift  = torch.nn.Parameter(torch.zeros(1)) # shift parameter
        self.ft_scale = torch.nn.Parameter(torch.zeros(1)) # scale parameter
        self.ft_shift  = torch.nn.Parameter(torch.zeros(1)) # shift parameter
        
        self._prior = prior # Gaussian or t_distri (Student)
        print(self.ft_scale)
        print(self.ft_shift )
        
    def fit(self, X, y, centre_=None):
#         pdb.set_trace()
#         self._device=
        # preprocess feature
        if self._preprocess_feature == 'UN':
            X = X
        elif self._preprocess_feature == 'L2N':
            X = F.normalize(X, p=2, dim=1)
        elif self._preprocess_feature in ['trC_L2N','sptC_L2N']:
            if self._bool_autoft:
                X = F.normalize(X - (centre_ + self.ft_shift), p=2, dim=1)               
#                 X_shift = centre_ + self.ft_shift
#                 X_scale = torch.norm(X-X_shift,p=2,dim=1,keepdim=True) + self.ft_scale
#                 X = (X-X_shift)/X_scale
            else:
                X = F.normalize(X - centre_, p=2, dim=1)
        else:
            raise ValueError('self._preprocess_feature should be UN/L2N/tr_CL2N/spt_CL2N, but your input is', self._preprocess_feature )
        
            
        # AutoFeatureTransformer
#         if self._bool_autoft:
#         X = self.ft_scale*X + self.ft_shift       
        # Learn MetaQDA 
        self.classes_ = np.unique(y.cpu())
        self.mu = []
        self.sigma = []
#         self.sigma_inv = []
        sigma = torch.zeros_like(self.S)
        self.support_N_j_list=[]
        for j in self.classes_:
            X_j = X[y==j]
#             X_j = torch.index_select(X,0,torch.tensor(np.where(y==j)[0]).cuda())
            N_j = X_j.shape[0]
            self.support_N_j_list.append(N_j)
            d = X_j.shape[1]
#             pdb.set_trace()
            mu_j = (self.kappa / (self.kappa + N_j)) * self.m + (N_j / (self.kappa + N_j)) * torch.mean(X_j, dim=0)
            S_j = torch.zeros_like(self.S)
        
            for i in range(X_j.shape[0]):
                S_j += torch.mm(X_j[i, :].unsqueeze(1), X_j[i, :].unsqueeze(0))
#             # Lower triangular matrix 
            if self.diag_triu == 'abs':
                lower_triu_matrix=torch.diag_embed(torch.abs(self.diagonal_param), offset=0) *self.diagonal_mask*self.diagonal_mask.t()+self.S * self.triu_mask
            if self.diag_triu == 'origin':
                lower_triu_matrix=self.S * self.triu_mask
            if self._prior == 'Gaussian':              
                left_eq = 1.0 / (self.nu + N_j + d + 2)
                right_eq = torch.mm(lower_triu_matrix,(lower_triu_matrix).t()) + S_j + self.kappa * torch.mm(self.m.unsqueeze(1), self.m.unsqueeze(0))-(self.kappa + N_j) * torch.mm(mu_j.unsqueeze(1), mu_j.unsqueeze(0))
                sigma_j = left_eq * right_eq
            if self._prior == 't_distri':
                left_eq = (self.kappa + N_j + 1)/((self.kappa+N_j)*(self.nu+N_j-self.feature_dim+1))
                right_eq = torch.mm(lower_triu_matrix,(lower_triu_matrix).t()) + S_j + self.kappa * torch.mm(self.m.unsqueeze(1), self.m.unsqueeze(0))-(self.kappa + N_j) * torch.mm(mu_j.unsqueeze(1), mu_j.unsqueeze(0))
                sigma_j = left_eq * right_eq
                
            sigma += sigma_j
            self.mu.append(mu_j)
            self.sigma.append(sigma_j)
            
        if self._bool_inv_reg in ['Regular_Inverse']: 
#             sigma_list =self.norm_convmatrix(sigma_list)
            sigma_reg_list = self.reg_convmatrix(self.sigma)
            self.sigma_inv = self.inv_convmatrix(sigma_reg_list)
        if self._bool_inv_reg in ['Inverse_Regular']: 
            sigma_inv = self.inv_convmatrix(self.sigma)
#             sigma_inv =self.norm_convmatrix(sigma_inv)
            self.sigma_inv = self.reg_convmatrix(sigma_inv)

    def predict(self, X, centre_=None):
        # preprocess feature
        if self._preprocess_feature == 'UN':
            X = X
        elif self._preprocess_feature == 'L2N':
            X = F.normalize(X, p=2, dim=1)
        elif self._preprocess_feature in ['trC_L2N','sptC_L2N']:
            if self._bool_autoft:
                X = F.normalize(X - (centre_ + self.ft_shift), p=2, dim=1)               
#                 X_shift = centre_ + self.ft_shift
#                 X_scale = torch.norm(X-X_shift,p=2,dim=1,keepdim=True) + self.ft_scale
#                 X = (X-X_shift)/X_scale
            else:
                X = F.normalize(X - centre_, p=2, dim=1)
        else:
            raise ValueError('self._preprocess_feature should be UN/L2N/tr_CL2N/spt_CL2N, but your input is', self._preprocess_feature )
#         if self._bool_autoft:
#             # AutoFeature Shift Scale Transform
#         X = self.ft_scale*X + self.ft_shift
        # Predicts
        
        predicts_matrix=[]
        for i in range(X.shape[0]):
            neg_distrances=[]
            for j in range(len(self.classes_)):
                diff = X[i, :] - self.mu[j]
                gaussian_dist = torch.mm(torch.mm(diff.unsqueeze(0), self.sigma_inv[j]), diff.unsqueeze(1))
                if self._prior == 'Gaussian':
                    neg_distrances.append(-1*gaussian_dist) # simulate the fully-connect layer output by make the output negative
                if self._prior == 't_distri':
                    N_j = self.support_N_j_list[j]
                    common_term = self.nu+N_j+1
                    t_neg_dist = torch.lgamma(0.5*common_term)-torch.lgamma(0.5*(common_term-self.feature_dim))-0.5*self.feature_dim*torch.log(common_term-self.feature_dim)-0.5*torch.logdet(self.sigma[j])-0.5*common_term*torch.log(1+(1/(common_term-self.feature_dim))*gaussian_dist)
                    neg_distrances.append(t_neg_dist) # simulate the fully-connect layer output by make the output negative
    
            predicts_matrix.append(torch.cat(neg_distrances,dim=1))
        predicts_matrix = torch.cat(predicts_matrix,dim=0)
        
        if self.diag_triu == 'abs':
            lower_triu_matrix=torch.diag_embed(torch.abs(self.diagonal_param),offset=0) *self.diagonal_mask*self.diagonal_mask.t()+self.S * self.triu_mask
        if self.diag_triu == 'origin':
            lower_triu_matrix=self.S * self.triu_mask
        index_vector = torch.arange(0, self.feature_dim).unsqueeze(0).cuda()
#         # Old Cholesky loss
#         lambda1=0.5 * (self._cholesky_alpha + self.feature_dim + 1)
#         lambda2=0.5 * (self._cholesky_alpha - self.feature_dim - 1)
#         
#         scale_matrix_s = torch.mm(lower_triu_matrix,lower_triu_matrix.permute(1,0))
#         cholesky_loss_logitem = lambda1 * torch.log(lower_triu_matrix.gather(0, index_vector)).sum() 
#         cholesky_loss_traceitem = lambda2 * torch.cholesky_inverse(lower_triu_matrix).gather(0, index_vector).sum()
        # new Cholesky loss
#         pdb.set_trace()
        S_ = torch.matmul(lower_triu_matrix, lower_triu_matrix.t())
        L_inv = torch.cholesky_inverse(lower_triu_matrix)
        chlsk_loss_log = torch.logdet(S_)
        chlsk_loss_trace =  torch.trace(torch.mm(L_inv.t(), L_inv))
        
        return predicts_matrix, chlsk_loss_log, chlsk_loss_trace, lower_triu_matrix.gather(0, index_vector), S_.gather(0, index_vector), torch.cholesky_inverse(lower_triu_matrix).gather(0, index_vector), lower_triu_matrix
    
    def norm_convmatrix(self, cov_matrix):
        max_values = []
        for item in cov_matrix:
            max_values.append(item.max())
        for index, item in enumerate(cov_matrix):
            cov_matrix[index] = item/float(max_values[index])
        return cov_matrix
    def inv_convmatrix(self, cov_matrix):
        sigma_inv=[]
        for sigma_j in cov_matrix:
            sigma_inv.append(torch.inverse(sigma_j))
        return sigma_inv
    def reg_convmatrix(self, cov_matrix):
        sigm_reg=[]
        for item in cov_matrix:
            sigm_reg.append((1-self.reg_param) * item + self.reg_param * torch.eye(item.shape[0]).cuda())
        return sigm_reg
    def mqda_parameters(self):
        parameters = [self.m, self.S, self.nu, self.kappa, self.diagonal_param]
        for p in parameters:
            yield p
    def autoft_parameters(self):
        parameters = [self.ft_scale, self.ft_shift]
        for p in parameters:
            yield p


class MetaQDA_CPU(BaseEstimator, ClassifierMixin):
    def __init__(self, m, S, kappa, nu, lda=False, reg_param=0.0,inv_reg='-R'):
        self.m = np.array(m)
        self.S = np.array(S)
        self.kappa = kappa
        self.nu = nu
        self.lda = lda
        self.reg_param = reg_param  
        self._inv_reg = inv_reg
        assert inv_reg in ['-R','-I']

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.mu = []
        self.sigma = []
        self.sigma_inv = []
        sigma = np.zeros_like(self.S)
        
        for j in self.classes_:
            X_j = X[y == j, :]
            N_j = X_j.shape[0]
            d = X_j.shape[1]
            mu_j = (self.kappa / (self.kappa + N_j)) * self.m + (N_j / (self.kappa + N_j)) * np.mean(X_j, axis=0)
            S_j = np.zeros_like(self.S)
            for i in range(X_j.shape[0]):
                S_j += np.outer(X_j[i, :], X_j[i, :])
            sigma_j = 1.0 / (self.nu + N_j + d + 2) * (
                        self.S + S_j + self.kappa * np.outer(self.m, self.m) - (self.kappa + N_j) * np.outer(mu_j, mu_j))
            sigma += sigma_j
            self.mu.append(mu_j)
            self.sigma.append(sigma_j)
            
            if self._inv_reg in ['-I']: 
                sigma_reg_list = self.reg_convmatrix(self.sigma)
                self.sigma_inv = self.inv_convmatrix(sigma_reg_list)
            if self._inv_reg in ['-R']: 
                sigma_inv = self.inv_convmatrix(self.sigma)
                self.sigma_inv = self.reg_convmatrix(sigma_inv)
            
        if self.lda:
#             pdb.set_trace()
            sigma *= 1.0 / (len(self.classes_))
            if self._inv_reg in ['-I']:  # first regularization then Inverse
                sigma = (1-self.reg_param) * sigma + self.reg_param * np.eye(sigma.shape[0])
                sigma_inv = np.linalg.inv(sigma)
            if self._inv_reg in ['-R']: # first Inverse then regularization  
                sigma_inv = np.linalg.inv(sigma)
                sigma_inv = (1-self.reg_param) * sigma_inv + self.reg_param * np.eye(sigma_inv.shape[0])
            for i in range(len(self.sigma_inv)):
                self.sigma_inv[i] = sigma_inv

    def predict(self, X):

        y = []
        distance_matrix=[]
        for i in range(X.shape[0]):
            min_dist = float("inf")
            min_class = -1
            neg_distrances=[]
            for j in range(len(self.classes_)):
                diff = X[i, :] - self.mu[j]
                dist = np.dot(np.dot(diff, self.sigma_inv[j]), diff.T)
                neg_distrances.append(-1*dist)
                if dist < min_dist:
                    min_dist = dist
                    min_class = j
            distance_matrix.append(np.array(neg_distrances))
            y.append(self.classes_[min_class])
        distance_matrix = np.stack(distance_matrix)
        return np.array(y), distance_matrix
    
    def inv_convmatrix(self, cov_matrix):
        sigma_inv=[]
        for sigma_j in cov_matrix:
            sigma_inv.append(np.linalg.inv(sigma_j))
        return sigma_inv
    def reg_convmatrix(self, cov_matrix):
        sigm_reg=[]
        for item in cov_matrix:
            sigm_reg.append((1-self.reg_param) * item + self.reg_param * np.eye(item.shape[0]))
        return sigm_reg
        

def rbe_mqda_parameters_gpu(data_platform = '', net='', feature_dim=64, feature_reduction=None, NUM_IMAGES=600 ,feature_logits=1):

    NUM_IMAGES = NUM_IMAGES
    with torch.no_grad():
        C = len(os.listdir(data_platform))  # num_of_classes
#         print(C)
#         feature_dim = Way * Shot # 64?25?
#         NUM_IMAGES = NUM_IMAGES
    
        m = torch.tensor( np.zeros((feature_dim,))).float().cuda() #zero matrix
        S = torch.tensor(np.eye(feature_dim)).float().cuda() #d*d identity matrix
        nu = torch.tensor(feature_dim).float().cuda() # dimension is d, int not float
        kappa = torch.tensor(1).float().cuda()
        gamma_m = torch.tensor(C / (C * NUM_IMAGES)).float().cuda()  # 600 is num_of_instances in each class
        gamma_S = torch.tensor(1.0).float().cuda()
#         pdb.set_trace()
        for category in os.listdir(data_platform):
        # Get all the instances for class i
            rbe_loader = load_materials.LoadRBE(data_platform, category)
            batch_cate = rbe_loader.__iter__().__next__()[0]
#           bs, crops, c, w, h = batch_cate.shape
#           batch_cate = batch_cate.reshape(-1, c, w, h)
            features = net(batch_cate,True)[feature_logits]
#           features = F.normalize(features, p=2, dim=1)
            features = features.cpu().numpy()
            if feature_reduction:
#               feature_reduction.fit(features)
                features = feature_reduction.transform(features)
            mean_i = torch.mean(torch.tensor(features).cuda(), dim=0)#[:feature_dim]# + torch.mean(features, dim=0)[feature_dim:])/2
#           cov = torch.tensor(np.cov(features[:,:feature_dim], rowvar=False)).float().cuda()
            cov = torch.tensor(np.cov(features, rowvar=False)).float().cuda()
    #       pdb.set_trace()
            S += cov * NUM_IMAGES
            m = (kappa * m + NUM_IMAGES * mean_i) / (kappa + NUM_IMAGES)
            kappa += NUM_IMAGES
            nu += NUM_IMAGES
    
    return m, S, nu, kappa, gamma_m, gamma_S

def rbe_mqda_parameters_cpu(data_platform, feature_dict, feature_dim=64, head_root='', NUM_IMAGES=600,logits=True):

    NUM_IMAGES = NUM_IMAGES
    with torch.no_grad():
        
        c_ = len(os.listdir(data_platform))  # num_of_classes
        gamma_m = float(c_ / (c_ * NUM_IMAGES))
        gamma_s = 1.
        m = np.zeros((feature_dim,),)*gamma_m #zero matrix
#         S = torch.triu(torch.ones(feature_dim, feature_dim), diagonal=0).cuda() * gamma_S
        S = np.eye(feature_dim)*gamma_s #d*d identity matrix
        nu = float(feature_dim)*gamma_s # dimension is d, int not float
        kappa = 1*gamma_m
        
        for category in os.listdir(data_platform):
            # Get all the instances for class i
            rbe_loader = load_materials.LoadRBE(data_platform, category, head_root)
            image_names = rbe_loader.__iter__().__next__()[1]
            features_cpu=util.dict2features(feature_dict,image_names,head_root,logits)
            
            mean_i = np.mean(features_cpu, axis=0)
            cov = np.cov(features_cpu, rowvar=False)
            S += cov * NUM_IMAGES
            m = (kappa * m + NUM_IMAGES * mean_i) / (kappa + NUM_IMAGES)
            kappa += NUM_IMAGES
            nu += NUM_IMAGES
    
    return m, S, nu, kappa, gamma_m, gamma_s



def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

''' BASE LINE '''
def mixup_lc_fsl(n_way, n_support, n_query, spt_x, spt_y, qry_x, qry_y):
#     linear_clf = DistLinear(spt_x.shape[-1], n_way)
    linear_clf = nn.Sequential( nn.Linear(spt_x.shape[-1], n_way))
    linear_clf.train()
    optimizer = optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
    temp = [p for p in linear_clf.parameters()]
    cudnn.benchmark = True
    batch_size = 4
    support_size = n_way* n_support
    scores_eval = []
    with torch.enable_grad():
        for epoch in range(301):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ])
                z_batch = spt_x[selected_id]
                y_batch = spt_y[selected_id] 
                scores = linear_clf(z_batch)
                loss = F.cross_entropy(scores,y_batch)
                loss.backward()
                optimizer.step()
            if epoch %100 ==0 and epoch !=0:
                linear_clf.eval()
                scores = linear_clf(qry_x)
                acc, rewards = util.accuracy(scores.data.argmax(dim=1), qry_y)
                scores_eval.append(acc*100)
        scores_spt = linear_clf(spt_x)
    return scores_eval[0], scores_eval[1], scores.detach().cpu(), scores_spt.detach().cpu()

def train_liner_classifier(sample_features, sample_labels, Way, cuda=True):
    linear_clf = nn.Sequential( nn.Linear(sample_features.shape[-1], Way))
    if cuda:
        linear_clf=linear_clf.cuda()
    criterion = nn.CrossEntropyLoss()
    linear_clf.train()
    optimizer = optim.SGD(linear_clf.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    cudnn.benchmark = True
    

    with torch.enable_grad():
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = linear_clf(sample_features)
            outputs.max(dim=1)
            loss = criterion(outputs, sample_labels)
            loss.backward()
            optimizer.step()
            prec1, _ = util.accuracy(outputs.argmax(dim=1), sample_labels)

    return linear_clf

class AutoFeatureTransformer(nn.Module):
    def __init__(self,aft=True, model='LC'):
        super(AFT, self).__init__()
        assert model in ['LC']
        self._aft = aft
        self._model = model
        self._gamma = torch.nn.Parameter(torch.ones(1)*0.3)
        self._beta = torch.nn.Parameter(torch.ones(1)*0.5)
        
        print(self._gamma)
        print(self._beta)
        
    def fit_support(self, support_tensor, support_labels):
        if self._aft:
            support_tensor = self._gamma*support_tensor+self._beta
        if self._model=='LC':
            self.lc = train_liner_classifier(support_tensor, support_labels, 5, cuda=False)
            
    def infer_query(self, query_tensor):
        if self._aft:
            query_tensor = self._gamma*query_tensor+self._beta
        if self._model=='LC':
            outputs = self.lc(query_tensor)
            
        return outputs
    
    def aft_parameters(self):
        parameters = [self._gamma, self._beta]
        for p in parameters:
            yield p
            

class NearestCentroid_SimpleShot():
    def __init__(self):
        self.center_list = 0
    
    def fit(self, features, labels ):
        
        cate_s = list(set([e.item() for e in labels]))
        center_list = []
        
        for c in cate_s:
            index_tensor = (labels==c).unsqueeze(0).float()  # dim: 1 x 25
            centor_feature = torch.mm(index_tensor,features)/5 # dim: 1 x 64 = [1x25]·[25x64]
            center_list.append(centor_feature)
        
        self.center_list = center_list # [[1x64]...[1x64]] 
        
    def predict(self,features):
        
        distance_matrix=[]
        for v in features:
            neg_distrances=[]
            for center_feature in self.center_list :
                dist=F.pairwise_distance(center_feature, v, p=2).item()
                neg_distrances.append(-dist)
#             pdb.set_trace()
            distance_matrix.append(torch.tensor(neg_distrances))
                            
        return torch.stack(distance_matrix)
        
def QDA(reg_param=0.0):
    # This QDA do not have prior
    return QuadraticDiscriminantAnalysis(reg_param=reg_param)

def LDA(solver='svd', shrinkage='auto'):
    return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

# normed Liner classifier (refer: https://github.com/nupurkmr9/S2M2_fewshot/blob/94b4ed862a842897e6ddfa9092c309ad4ccdb3f9/backbone.py#L22)
class DistLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(DistLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores
