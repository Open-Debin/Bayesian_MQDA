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
# CUDA_VISIBLE_DEVICES=4 python train_mqda_incremental_session.py -lr 1e-3 --novel_weight 0.98
def main():
    args = config_local_parameters()
    util.set_random(args.seed)
    x_np, x_centre_np = load.numpy_features(args, args.path2features, args.feature_or_logits)      
    args.x_centre = torch.tensor(x_centre_np).to(DEVICE)
    logger = util.Logger(args.log_dir, 'setting', args)

    classifier = config.meta_qda(args).to(DEVICE)  
    optimizer = config.optimizer(classifier.parameters(), args)
    lr_scheduler = config.lr_scheduler(optimizer, args)

    folders_base  = generate.folders(args.base_path)
    timer = metric.CountdownTimer(total_steps=args.n_epoch * args.n_episode_train)
    for index_epoch in range(args.n_epoch):
        folders_manyshot, folders_fewshot = generate_fake_base_novel_class_folders(folders_base, args.n_base)
        manyshot_fewclass_incremental_training(folders_manyshot, folders_fewshot, x_np, classifier, optimizer, lr_scheduler, args, index_epoch, timer, logger)
        save.checkpoint_epoch(classifier.state_dict(), index_epoch, optimizer.state_dict(), lr_scheduler.state_dict(), args)
    logger.rename('mtrain')  

def generate_fake_base_novel_class_folders(folders, manyshot_way=20):
    random.shuffle(folders)
    return folders[:manyshot_way], folders[manyshot_way:]    
    
def manyshot_fewclass_incremental_training(folders_manyshot, folders_fewshot, np_features, classifier, optimizer, lr_scheduler, args, index_epoch, timer, logger):
    accuracies= metric.AverageMeter()
    classifier.train()
    task_manyshot = generate.Task(folders_manyshot, args.n_base, 200, args.k_qry)
    (ms_x_spt, ms_y_spt), (ms_x_qry, ms_y_qry) = load.tensor_features_from(task_manyshot, np_features, args)
    ms_norm_x_spt = layers.centre_l2norm(ms_x_spt, args.x_centre)
    ms_norm_x_qry = layers.centre_l2norm(ms_x_qry, args.x_centre)
    
    classifier.fit_image_label(ms_norm_x_spt, ms_y_spt)
    outputs_ms = classifier.predict(ms_norm_x_qry)
    acc_base = metric.accuracy(outputs_ms.argmax(dim=1), ms_y_qry)
    ce_loss_ms = F.cross_entropy(outputs_ms, ms_y_qry)   
    losses.backward_propagation(ce_loss_ms, classifier.parameters(), optimizer)
    lr_scheduler.step()  
    
    for index_episode in range(args.n_episode_train):
        acc_s, loss = incremental_process(ms_norm_x_spt, ms_y_spt, ms_norm_x_qry, ms_y_qry, folders_fewshot, np_features, classifier, optimizer, lr_scheduler, args)
        logger.print(args.title_in_screen.format(index_epoch+1, args.n_epoch, index_episode+1, args.n_episode_train, loss, timer.step(), util.time_now()))
#         metric.mean_confidence_interval(acc_s[0])
        logger.print(args.acc_h_in_screen.format(acc_base, acc_s[0], acc_s[1], acc_s[2], acc_s[3], acc_s[4], acc_s[5]))

def incremental_process(ms_norm_x_spt, ms_y_spt, ms_norm_x_qry, ms_y_qry, folders_fewshot, np_features, classifier, optimizer, lr_scheduler, args):
    accuracies= metric.AverageMeter()
#     temp_many_few_mu = classifier.mu[:args.n_base]
#     temp_many_few_sigma = classifier.sigma_inv[:args.n_base]
    acc_session = [None for i in range(args.n_session)]
    
    for index_session in range(args.n_session):
        folder_session = folders_fewshot[index_session * args.n_way: (index_session + 1) * args.n_way]
        task_few_session = generate.Task(folder_session, args.n_way, args.k_spt, args.k_qry)
        (fs_x_spt, fs_y_spt), (fs_x_qry, fs_y_qry) = load.tensor_features_from(task_few_session, np_features, args)
        fs_norm_x_spt = layers.centre_l2norm(fs_x_spt, args.x_centre)
        fs_norm_x_qry = layers.centre_l2norm(fs_x_qry, args.x_centre)
        
        if index_session ==0:
            joint_x_spt, joint_y_spt = joint_many_few_x_y(ms_norm_x_spt, ms_y_spt, fs_norm_x_spt, fs_y_spt+args.n_base+args.n_way*index_session)
            joint_x_qry, joint_y_qry = joint_many_few_x_y(ms_norm_x_qry, ms_y_qry, fs_norm_x_qry, fs_y_qry+args.n_base+args.n_way*index_session)
        else:
            joint_x_spt, joint_y_spt = joint_many_few_x_y(joint_x_spt, joint_y_spt, fs_norm_x_spt, fs_y_spt+args.n_base+args.n_way*index_session)
            joint_x_qry, joint_y_qry = joint_many_few_x_y(joint_x_qry, joint_y_qry, fs_norm_x_qry, fs_y_qry+args.n_base+args.n_way*index_session)
            
        classifier.fit_image_label(joint_x_spt, joint_y_spt)
#         temp_many_few_mu.extend(classifier.mu)
#         temp_many_few_sigma.extend(classifier.sigma_inv)
#         classifier.mu = temp_many_few_mu 
#         classifier.sigma_inv = temp_many_few_sigma
#         pdb.set_trace()
        # A__
        outputs_joint = classifier.predict(joint_x_qry)
        acc_session[index_session] = metric.accuracy(outputs_joint.argmax(dim=1), joint_y_qry)
        ce_loss = F.cross_entropy(outputs_joint, joint_y_qry)
        # __A
#         # B__
#         outputs_novel = classifier.predict(fs_norm_x_qry)
#         acc_session[index_session].append(metric.accuracy(outputs_novel.argmax(dim=1), fs_y_qry+args.n_base+args.n_way*index_session))
#         ce_loss = F.cross_entropy(outputs_novel, fs_y_qry+args.n_base+args.n_way*index_session)
#         # __B        
        losses.backward_propagation(ce_loss, classifier.parameters(), optimizer)
        lr_scheduler.step() 
    
    return acc_session, ce_loss

def joint_many_few_x_y(ms_x, ms_y, fs_x, fs_y):
    joint_x = torch.cat([ms_x, fs_x])
    joint_y = torch.cat([ms_y, fs_y])
    
    return joint_x, joint_y

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'train_metaqda_scripts')        
    parser.add_argument('--k_spt', default=5, type=int,  help='number of labeled data in each class of support set') 
    parser.add_argument('--n_way', default=5, type=int,  help='we use support way and query way as the same') 
    parser.add_argument('--k_qry', default=10, type=int,  help='number of data in each class of query set')

    parser.add_argument('--n_epoch', default=6, type=int, help='')
    parser.add_argument('--n_episode_train', default=10, type=int, help='')
    parser.add_argument('--n_episode_test', default=2, type=int, help='')
    
    # incremental
    parser.add_argument('--n_base', default=30, type=int, help='number of base class')
    parser.add_argument('--n_session', default=6, type=int, help='number of incremental step')
    
    parser.add_argument('--net_domain', default='mini', help='')
    parser.add_argument('--net_arch', default='resnet18', type=str,help='')
    parser.add_argument('--x_domain', default='incremental', type=str,help='')
    parser.add_argument('--session', default='session1', type=str,help='')
    parser.add_argument('--feature_or_logits', default=0, help='0: feature; 1: logits')
    
    parser.add_argument('-lr','--lr', default=1e-4, type=float,help='learning rate')
    parser.add_argument('--optimizer', default='sgd', help='sgd, adam')    
    parser.add_argument('--lr_scheduler', default='multsteplr', type=str,help='consinelr, multsteplr')
    
    parser.add_argument('--reg_param',default=0.5, type=float, help="hope choleky_alpha > feature dim + 1 ")
    parser.add_argument('--choleky_alpha',default=False, help="hope choleky_alpha > feature dim + 1 ")
    parser.add_argument('--strategy', default='map_fixshot', help=''), 
    parser.add_argument('--fix_Nj', default=25, type=int,help='')
    parser.add_argument('--novel_weight', default=1, type=float,help='range: (0,1)')
    
    parser.add_argument('--path2image', default='../../data_src/images/mini_incremental/', help='path to all datasets: CUB, Mini, etc')
    parser.add_argument('--seed', default=4603, type=int, help='')
                        
    args = parser.parse_args()
    args.path2image = str(Path(args.path2image).resolve())+'/'
    args.path2features='../../data_src/fea_mqda/{:}-{:}-{:}-fea-49.pkl'.format(args.net_domain,args.net_arch,args.x_domain)
    args.base_path=os.path.join(args.path2image+'{:}/train'.format(args.session))
    args.val_path=os.path.join(args.path2image+'{:}/val'.format(args.session))
    args.novel_path=os.path.join(args.path2image+'{:}/test'.format(args.session))
    args.log_dir = '../active_log/incremental/{:}_{:}_incremental49_baseSize{:}_{:}_{:}_{:}_{:}_{:}'.format(args.net_domain,args.net_arch, args.n_base, args.optimizer, args.strategy, args.lr, args.k_spt, ['fea','logits'][args.feature_or_logits])
    args.title_in_screen = "epoch [{:3d}/{:3d}], episode:[{:3d}/{:3d}], loss:{:}, needed [{:}], [{:}]" 
    args.acc_h_in_screen = 'acc1:{:} h1:{:}; acc2:{:} h2:{:}; acc3:{:} h3:{:}; acc4:{:} h4:{:}; acc5:{:} h5:{:}; acc6:{:} h6:{:}; acc7:{:} h7:{:}'
    args.acc_h_in_screen = 'acc_base:{:}, acc1:{:}; acc2:{:}; acc3:{:}; acc4:{:}; acc5:{:}; acc6:{:}'
    return args

if __name__=='__main__':
    main()
