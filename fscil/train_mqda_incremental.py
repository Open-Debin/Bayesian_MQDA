import os
import sys
import random
from pathlib import Path
mqda_lib_dir = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(mqda_lib_dir))
import pdb
import pickle
import argparse
import torch
import torch.nn.functional as F
from code_lib import generate, util, load, save, metric
from code_models import config, losses, classifiers, layers
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CUDA_VISIBLE_DEVICES=4 python train_mqda_incremental.py
def main():
    args = config_local_parameters()
    util.set_random(args.seed)
    x_np, x_centre_np = load.numpy_features(args, args.path2features, args.feature_or_logits)      
    args.x_centre = torch.tensor(x_centre_np).to(DEVICE)
    logger = util.Logger(args.log_dir, 'setting', args)

    learner = config.meta_qda(args).to(DEVICE)  
    optimizer = config.optimizer(learner.parameters(), args)
    lr_scheduler = config.lr_scheduler(optimizer, args)

    folders_base  = generate.folders(args.base_path)
    timer = metric.CountdownTimer(total_steps=args.n_epoch * args.n_episode_train)
    for index_epoch in range(args.n_epoch):
        folders_fake_base, folders_fake_novel = generate_fake_base_novel_class_folders(folders_base, args.n_base)
        train(folders_fake_base, folders_fake_novel, x_np, learner, optimizer, lr_scheduler, args, index_epoch, timer, logger)
        save.checkpoint_epoch(learner.state_dict(), index_epoch, optimizer.state_dict(), lr_scheduler.state_dict(), args)
    logger.rename('mtrain')  

def generate_fake_base_novel_class_folders(folders, base_way=20):
    random.shuffle(folders)
    return folders[:base_way], folders[base_way:]    
    
def train(folders_fake_base, folders_fake_novel, x_np, learner, optimizer, lr_scheduler, args, index_epoch, timer, logger):
    accuracies= metric.AverageMeter()
    learner.train()
    task_base = generate.Task(folders_fake_base, args.n_base, 200, args.k_qry)
    (x_spt, y_spt), (x_qry, y_qry) = load.tensor_features_from(task_base, x_np, args)
    x_spt_norm = layers.centre_l2norm(x_spt, args.x_centre)
    x_qry_norm = layers.centre_l2norm(x_qry, args.x_centre)
    
    learner.fit_image_label(x_spt_norm, y_spt)
    outputs = learner.predict(x_qry_norm)
    
    acc_base = metric.accuracy(outputs.argmax(dim=1), y_qry)
    ce_loss = F.cross_entropy(outputs, y_qry)   
    losses.backward_propagation(ce_loss, learner.parameters(), optimizer)
    lr_scheduler.step()  
    
    for index_episode in range(args.n_episode_train):
        acc_s, loss = incremental_process(x_spt_norm, y_spt, x_qry_norm, y_qry, folders_fake_novel, x_np, learner, optimizer, lr_scheduler, args)
        logger.print(args.title_in_screen.format(index_epoch+1, args.n_epoch, index_episode+1, args.n_episode_train, loss, timer.step(), util.time_now()))
        logger.print(args.acc_h_in_screen.format(acc_base, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5]))

def incremental_process(base_x_spt, base_y_spt, base_x_qry, base_y_qry, folders_fake_novel, x_np, learner, optimizer, lr_scheduler, args):
    accuracies= metric.AverageMeter()
    acc_session = [None for i in range(args.n_session)]
    
    for index_session in range(args.n_session):
        folder_session = folders_fake_novel[index_session * args.n_way: (index_session + 1) * args.n_way]
        task_session = generate.Task(folder_session, args.n_way, args.k_spt, args.k_qry)
        (novel_x_spt, novel_y_spt), (novel_x_qry, novel_y_qry) = load.tensor_features_from(task_session, x_np, args)
        novel_x_spt = layers.centre_l2norm(novel_x_spt, args.x_centre)
        novel_x_qry = layers.centre_l2norm(novel_x_qry, args.x_centre)
        
        if index_session ==0:
            joint_x_spt = torch.cat([base_x_spt, novel_x_spt])
            joint_y_spt = torch.cat([base_y_spt, novel_y_spt+args.n_base+args.n_way*index_session])
            joint_x_qry = torch.cat([base_x_qry, novel_x_qry])
            joint_y_qry = torch.cat([base_y_qry, novel_y_qry+args.n_base+args.n_way*index_session]) 
        else:
            joint_x_spt = torch.cat([joint_x_spt, novel_x_spt])
            joint_y_spt = torch.cat([joint_y_spt, novel_y_spt+args.n_base+args.n_way*index_session])
            joint_x_qry = torch.cat([joint_x_qry, novel_x_qry])
            joint_y_qry = torch.cat([joint_y_qry, novel_y_qry+args.n_base+args.n_way*index_session])
            
        learner.fit_image_label(joint_x_spt, joint_y_spt)
        outputs_joint = learner.predict(joint_x_qry)
        
        acc_session[index_session] = metric.accuracy(outputs_joint.argmax(dim=1), joint_y_qry)
        ce_loss = F.cross_entropy(outputs_joint, joint_y_qry)
        losses.backward_propagation(ce_loss, learner.parameters(), optimizer)
        lr_scheduler.step() 
    
    return acc_session, ce_loss

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'train_metaqda_incremental_learning_scripts')      
    parser.add_argument('--n_way', default=5, type=int,  help='we use support way and query way as the same')
    parser.add_argument('--k_spt', default=5, type=int,  help='number of labeled data in each class of support set') 
    parser.add_argument('--k_qry', default=10, type=int,  help='number of data in each class of query set')

    parser.add_argument('--n_epoch', default=6, type=int, help='')
    parser.add_argument('--n_episode_train', default=10, type=int, help='')

    parser.add_argument('--n_base', default=30, type=int, help='number of base class')
    parser.add_argument('--n_session', default=6, type=int, help='number of incremental step')
    
    parser.add_argument('--net_domain', default='mini', help='')
    parser.add_argument('--net_arch', default='resnet18', type=str,help='')
    parser.add_argument('--session', default='session1', type=str,help='')
    parser.add_argument('--feature_or_logits', default=0, help='0: feature; 1: logits')
    
    parser.add_argument('-lr','--lr', default=1e-2, type=float,help='learning rate')
    parser.add_argument('--optimizer', default='sgd', help='sgd, adam')    
    parser.add_argument('--lr_scheduler', default='multsteplr', type=str,help='consinelr, multsteplr')
    parser.add_argument('--strategy', default='map', help='')
    parser.add_argument('--reg_param', default=0.5, type=float)
    
    parser.add_argument('--path2image', default='../../data_src/images/mini_incremental/', help='path to all datasets: CUB, Mini, etc')
    parser.add_argument('--seed', default=4603, type=int, help='')
                        
    args = parser.parse_args()
    args.path2image = str(Path(args.path2image).resolve())+'/'
    args.path2features='../../data_src/fea_mqda/{:}-{:}-incremental-fea-49.pkl'.format(args.net_domain,args.net_arch)
    args.base_path=os.path.join(args.path2image+'{:}/train'.format(args.session))
    args.log_dir = '../log/incremental/{:}_{:}_baseWay{:}_{:}_{:}_{:}_{:}_{:}'.format(args.net_domain,args.net_arch, args.n_base, args.optimizer, args.strategy, args.lr, args.k_spt, ['features','logits'][args.feature_or_logits])
    args.title_in_screen = "epoch [{:3d}/{:3d}], episode:[{:3d}/{:3d}], loss:{:}, needed [{:}], [{:}]" 
    args.acc_h_in_screen = 'acc_base:{:}, acc1:{:}; acc2:{:}; acc3:{:}; acc4:{:}; acc5:{:}; acc6:{:}'
    return args

if __name__=='__main__':
    main()
