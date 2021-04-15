import torch

def checkpoint_epoch(model_state, epoch_index, optimizer_state, lr_scheduler_state, args):
    args.index_epoch = epoch_index
    info = {'args': args, 'optim' : optimizer_state, 'scheduler' : lr_scheduler_state, 'mqda' : model_state}
    save_path = "{:}/epoch-{:}.pth".format(args.log_dir, epoch_index)
    torch.save(info, save_path)

def urt_mqda(args, episode_index, optimizer, lr_scheduler, urt, classifier, save_dir, title='train'):
    args.index_episode = episode_index
    info = {'args': args, 'optim' : optimizer, 'scheduler' : lr_scheduler, 'urt' :urt, 'mqda' : classifier}
    save_path = "{:}/epoch-{:}.pth".format(save_dir, episode_index)
    last_path = "{:}/{:}_last_epoch.pth".format(save_dir, title)
    torch.save(info, save_path)
    torch.save(info, last_path)
    
def checkpoint_episode(args, episode_index, optimizer_state, lr_scheduler_state, model_state, save_dir):
    args.index_episode = episode_index
    info = {'args': args, 'optim' : optimizer_state, 'scheduler' : lr_scheduler_state, 'mqda' : model_state}
    save_path = "{:}/epoch-{:}.pth".format(save_dir, episode_index)
    torch.save(info, save_path)
    
def encoder(state, acc, loss, epoch, title=''):
    save_parent = './model_temp5/'
    if not os.path.exists(save_parent):
        os.makedirs(save_parent)
    save_dir = save_parent+title+'_' + str(epoch) + '_' + str(round(float(acc), 3))+ '_' + str(round(float(loss), 3))
    torch.save(state, save_dir)
    print(save_dir)


def features_in_dict(fea_dict, fea_mean_trva, logits_dict, logits_mean_trva, save_dir):
    save_dict = collections.defaultdict(list)
    save_dict['trCentre']=[fea_mean_trva.numpy(), logits_mean_trva.numpy()]
    for item in logits_dict.keys():
        save_dict[item]=[fea_dict[item][0].numpy(),logits_dict[item][0].numpy()]
    
    with open(save_dir,'wb') as f:
        pickle.dump(save_dict, f)
    print('Eva&Debin')
    print(save_dir)
