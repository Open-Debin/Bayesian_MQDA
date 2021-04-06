import os, time
import random
import torch
import numpy as np
from code_lib import metric
from tabulate import tabulate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{:}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
  return string


def convert_secs2time(epoch_time, string=True, xneed=True):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if string:
    if xneed:
      need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    else:
      need_time = '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
    return need_time
  else:
    return need_hour, need_mins, need_secs


class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val   = 0.0
    self.avg   = 0.0
    self.sum   = 0.0
    self.count = 0.0
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count    

  def __repr__(self):
    return ('{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__))


class ConfusionMatrix():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.mat = np.zeros([n_classes, n_classes])

    def update_mat(self, preds, labels, idxs):
        idxs = np.array(idxs)
        real_pred = idxs[preds]
        real_labels = idxs[labels]
        self.mat[real_pred, real_labels] += 1

    def get_mat(self):
        return self.mat


class Accumulator():
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.ac = np.empty(0)

    def append(self, v):
        self.ac = np.append(self.ac[-self.max_size:], v)

    def reset(self):
        self.ac = np.empty(0)

    def mean(self, last=None):
        last = last if last else self.max_size
        return self.ac[-last:].mean()


class IterBeat():
    def __init__(self, freq, length=None):
        self.length = length
        self.freq = freq

    def step(self, i):
        if i == 0:
            self.t = time.time()
            self.lastcall = 0
        else:
            if ((i % self.freq) == 0) or ((i + 1) == self.length):
                t = time.time()
                print('{0} / {1} ---- {2:.2f} it/sec'.format(
                    i, self.length, (i - self.lastcall) / (t - self.t)))
                self.lastcall = i
                self.t = t


class SerializableArray(object):
    def __init__(self, array):
        self.shape = array.shape
        self.data = array.tobytes()
        self.dtype = array.dtype

    def get(self):
        array = np.frombuffer(self.data, self.dtype)
        return np.reshape(array, self.shape)


def print_res(array, name, file=None, prec=4, mult=1):
    array = np.array(array) * mult
    mean, std = np.mean(array), np.std(array)
    conf = 1.96 * std / np.sqrt(len(array))
    stat_string = ("test {:s}: {:0.%df} +/- {:0.%df}"
                   % (prec, prec)).format(name, mean, conf)
    print(stat_string)
    if file is not None:
        with open(file, 'a+') as f:
            f.write(stat_string + '\n')


def process_copies(embeddings, labels, args):
    n_copy = args['test.n_copy']
    test_embeddings = embeddings.view(
        args['data.test_query'] * args['data.test_way'],
        n_copy, -1).mean(dim=1)
    return test_embeddings, labels[0::n_copy]


def set_determ(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_dicts(dicts, torch_stack=True):
    def stack_fn(l):
        if isinstance(l[0], torch.Tensor):
            return torch.stack(l)
        elif isinstance(l[0], str):
            return l
        else:
            return torch.tensor(l)

    keys = dicts[0].keys()
    new_dict = {key: [] for key in keys}
    for key in keys:
        for d in dicts:
            new_dict[key].append(d[key])
    if torch_stack:
        for key in keys:
            new_dict[key] = stack_fn(new_dict[key])
    return new_dict


def voting(preds, pref_ind=0):
    n_models = len(preds)
    n_test = len(preds[0])
    final_preds = []
    for i in range(n_test):
        cur_preds = [preds[k][i] for k in range(n_models)]
        classes, counts = np.unique(cur_preds, return_counts=True)
        if (counts == max(counts)).sum() > 1:
            final_preds.append(preds[pref_ind][i])
        else:
            final_preds.append(classes[np.argmax(counts)])
    return final_preds


def agreement(preds):
    n_preds = preds.shape[0]
    mat = np.zeros((n_preds, n_preds))
    for i in range(n_preds):
        for j in range(i, n_preds):
            mat[i, j] = mat[j, i] = (
                preds[i] == preds[j]).astype('float').mean()
    return mat


def read_textfile(filename, skip_last_line=True):
    with open(filename, 'r') as f:
        container = f.read().split('\n')
        if skip_last_line:
            container = container[:-1]
    return container


def check_dir(dirname, verbose=True):
    """This function creates a directory
    in case it doesn't exist"""
    try:
        # Create target Directory
        os.makedirs(dirname)
        if verbose:
            print("Directory ", dirname, " was created")
    except FileExistsError:
        if verbose:
            print("Directory ", dirname, " already exists")
    return dirname


def pre_load_results():
    sur_paper = {"ilsvrc_2012":[.563], "omniglot":[.931], "aircraft":[.854], "cu_birds":[.714], "dtd":[.715], "quickdraw":[.813], "fungi":[.631], "vgg_flower":[.828], "traffic_sign":[.704], "mscoco":[.524], "mnist":[.943], "cifar10":[.668], "cifar100":[.566]}
    urt_paper   = {"ilsvrc_2012":[.557], "omniglot":[.944], "aircraft":[.858], "cu_birds":[.763], "dtd":[.718], "quickdraw":[.825], "fungi":[.635], "vgg_flower":[.882], "traffic_sign":[.694], "mscoco":[.522], "mnist":[.948], "cifar10":[.673], "cifar100":[.569]}
    urt_exp   = {"ilsvrc_2012":[.5581], "omniglot":[.9438], "aircraft":[.851], "cu_birds":[.7629], "dtd":[.7147], "quickdraw":[.8144], "fungi":[.6363], "vgg_flower":[.8822], "traffic_sign":[.6761], "mscoco":[.5016], "mnist":[.8982], "cifar10":[.6509], "cifar100":[.5513]}
    mqda_base   = {"ilsvrc_2012":[.4875], "omniglot":[.9062], "aircraft":[.7875], "cu_birds":[.6294], "dtd":[.6667], "quickdraw":[.7138], "fungi":[.5288], "vgg_flower":[.80], "traffic_sign":[.8516], "mscoco":[.2938], "mnist":[.8714], "cifar10":[.72], "cifar100":[.4565]}
    return sur_paper, urt_paper, urt_exp


def show_results(dataset_names, alg2data2accuracy, compares, print_func):
  if not isinstance(compares, tuple):
    raise ValueError('compares input is {:} that caused error',format(compares))
  rows, better_1, better_2 = [], 0, 0
  # debin___
  mean_improve1 = metric.AverageMeter()
  mean_improve2 = metric.AverageMeter()
  # ————debin
  for dataset_name in dataset_names:
    row = [dataset_name]
    xname2acc = {}
    for model_name, data2accs in alg2data2accuracy.items():
      acc = np.array(data2accs[dataset_name]) * 100
      mean_acc = acc.mean()
      conf = (1.96 * acc.std()) / np.sqrt(len(acc))
      row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
      xname2acc[model_name] = mean_acc
    row.append("{:.2f}".format(xname2acc[compares[-1]]-xname2acc[compares[0]]))
    row.append("{:.2f}".format(xname2acc[compares[-1]]-xname2acc[compares[1]]))
    better_1 += xname2acc[compares[-1]] > xname2acc[compares[0]]
    better_2 += xname2acc[compares[-1]] > xname2acc[compares[1]]
    mean_improve1.update(xname2acc[compares[-1]] - xname2acc[compares[0]])
    mean_improve2.update( xname2acc[compares[-1]] - xname2acc[compares[1]])
    rows.append(row)
  acc1, h1 = mean_improve1.mean_confidence_interval()
  acc2, h2 = mean_improve2.mean_confidence_interval()
  alg_names = list(alg2data2accuracy.keys()) + ['{:}>{:} {:02d}/{:02d}, acc:{:}_{:}'.format(compares[-1], compares[0], better_1, len(dataset_names), acc1, h1)] + ['{:}>{:} {:02d}/{:02d}, acc:{:}_{:}'.format(compares[-1], compares[1], better_2, len(dataset_names), acc2, h2)]
  table = tabulate(rows, headers=['model \\ data'] + alg_names, floatfmt=".2f")
  print_func(table)
