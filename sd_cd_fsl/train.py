import os
import sys
import pdb
import pickle
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve().parent))
from code_models import layers, losses, config
from code_lib import generate, load, save, util, metric
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    args = config_local_parameters()
    util.set_random(args.seed)
    x_np, x_centre_np = load.numpy_features(args, args.path2features, args.feature_or_logits)
    
    args.x_centre = torch.tensor(x_centre_np).to(DEVICE)
    logger = util.Logger(args.log_dir, 'setting', args)

    learner = config.meta_qda(args).to(DEVICE)  
    optimizer = config.optimizer(learner.parameters(), args)
    lr_scheduler = config.lr_scheduler(optimizer, args)

    folders = generate.folders(args.base_path, args.val_path)
    timer = metric.CountdownTimer(total_steps=args.n_epoch)
    
    for index_epoch in range(args.n_epoch):    
        acc, h_ = metatrain_epoch(folders, x_np, learner, optimizer, lr_scheduler, args)
        save.checkpoint_epoch(learner.state_dict(), index_epoch, optimizer.state_dict(), lr_scheduler.state_dict(), args)
        logger.print(args.string_in_screen.format(index_epoch+1, args.n_epoch, args.n_episode_train, acc, h_, timer.step(), util.time_now()))
    logger.rename('metrain')
    
def metatrain_epoch(folders, x_np, learner, optimizer, lr_scheduler, args):
    accuracies= metric.AverageMeter()
    learner.train()
    for _ in range(args.n_episode_train+1):
        tasks = generate.Task(folders, args.n_way, args.k_spt, args.k_qry)
        (x_spt, y_spt), (x_qry, y_qry) = load.tensor_features_from(tasks, x_np, args)
        x_spt_norm = layers.centre_l2norm(x_spt, args.x_centre)
        x_qry_norm = layers.centre_l2norm(x_qry, args.x_centre)
        
        learner.fit_image_label(x_spt_norm, y_spt) 
        outputs = learner.predict(x_qry_norm)            
        
        accuracies.update(metric.accuracy(outputs.argmax(dim=1), y_qry))
        cross_entropy_loss = F.cross_entropy(outputs, y_qry)
        losses.backward_propagation(cross_entropy_loss, learner.parameters(), optimizer)
        
        lr_scheduler.step()  
    accuracy, h_ = accuracies.mean_confidence_interval()
    return accuracy, h_

def config_local_parameters():
    parser = argparse.ArgumentParser(description= 'train_metaqda_scripts')        
    parser.add_argument('--k_spt', default=1, type=int,  help='number of labeled data in each class of support set') 
    parser.add_argument('--n_way', default=5, type=int,  help='way of support & query') 
    parser.add_argument('--k_qry', default=15, type=int,  help='number of data in each class of query set')

    parser.add_argument('--n_epoch', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--n_episode_train', default=100, type=int, help='Number of training episode in each epoch')
    
    parser.add_argument('--net_domain', default='mini', help='dataset that used for training feature encoder')
    parser.add_argument('--net_arch', default='s2m2r', type=str,help='conv4, resnet18, wrn')
    parser.add_argument('--feature_or_logits', default=1, type=int,help='0: feature; 1: logits')
    
    parser.add_argument('-lr','--lr', default=1e-4, type=float,help='learning rate')
    parser.add_argument('--optimizer', default='sgd', help='sgd, adam')    
    parser.add_argument('--lr_scheduler', default='multsteplr', type=str,help='multsteplr, consinelr')
    
    parser.add_argument('--strategy', default='map', help='map_fixshot, map (Maximum Posterior Distribution), fb (Fully Bayes)')    
    parser.add_argument('--path2image', default='../../data_src/images/', help='the path that stores all datasets: CUB, Mini, etc')
    parser.add_argument('--reg_param', default=0.5, type=float)
    parser.add_argument('--seed', default=4603, type=int, help='random seed')                    
    args = parser.parse_args()
    
    args.path2features='../data/features/{:}-{:}-{:}-fea.pkl'.format(args.net_domain,args.net_arch,args.net_domain)
    args.base_path=os.path.join(args.path2image+'{:}/train'.format(args.net_domain))
    args.val_path=args.base_path.replace('train','val')
    args.log_dir = '../log/SD_CD_FSL/{:}_{:}_{:}_{:}_{:}_{:}N{:}K_{:}'.format(args.net_domain,args.net_arch, args.optimizer, args.strategy, args.lr, args.n_way, args.k_spt, ['features','logits'][args.feature_or_logits])
    args.string_in_screen = "epoch [{:3d}/{:3d}], episode:{:}, acc:{:} h:{:}, still need [{:}], [{:}]" 
    return args

if __name__=='__main__':
    main()
