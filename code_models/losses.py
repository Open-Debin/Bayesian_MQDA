import pdb
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

def openset_xlogx(outputs_open, power=1):
    loss_open_xlogx = F.softmax(outputs_open, dim=1).pow(power) * F.log_softmax(outputs_open, dim=1) # torch.Size([75, 5])
    loss_open_xlogx = loss_open_xlogx.sum(dim=1) # torch.Size([75])
    return loss_open_xlogx.mean()

def cholesky_loss(lower_triu):
    S_ = torch.matmul(lower_triu, lower_triu.t())
    L_inv = torch.cholesky_inverse(lower_triu)

    return torch.logdet(S_) + torch.trace(torch.mm(L_inv.t(), L_inv))

def backward_propagation(loss, update_parameters, optimizer, retain_graph=False, grad_norm=True):
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_norm:
        torch.nn.utils.clip_grad_norm_(update_parameters, 100)

#     for name, parms in net.named_parameters():	
#         print('-->name:{:} -->grad_requirs:{:} -->max/min:{:}/{:}-->grad_value:{:}'.format(name, parms.requires_grad, round(parms.grad.max().item(),3), round(parms.grad.min().item(),3), parms.grad))
#     pdb.set_trace()
    optimizer.step()


class LogitsWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(LogitsWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1)

    def forward(self, logits):

        return self.temperature_scale(logits.cuda())

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / torch.abs(temperature)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, label):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits = logits.cuda()
        labels = label.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
#         print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NL1
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=10)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
#         print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
#         print('Optimal temperature: %.3f' % self.temperature.item())

        return self


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, predict=False):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.predict = predict
        self.temperature = nn.Parameter(torch.ones(1) * 1)

    def forward(self, input_x):
        if self.predict:
            logits = self.model.predict(input_x)
        else:
            logits = self.model(input_x)
#         pdb.set_trace()
        return self.temperature_scale(logits.cuda())

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / torch.abs(temperature)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, input, label):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        with torch.no_grad():
            if self.predict:
                logits = self.model.predict(input)
            else:
                logits = self.model(input)
            logits = logits.cuda()
            labels = label.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
#         print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=2)
#         optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
#         print('Optimal temperature: %.3f' % self.temperature.item())
#         print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


    
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

def mixup_label_decision(y_a, y_b, lam):
    ''' Using mixup to enlarge the Support Set '''
    if lam > 0.8:
        return y_a, True
    elif lam <0.2:
        return y_b, True
    else:    
        return False, False 
    
    
    
    
    
class ECELoss_NIPs(nn.Module):
    """ Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin.
    Adapted from: https://github.com/gpleiss/temperature_scaling
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss_NIPs, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def calibrate(self, logits, labels, iterations=50, lr=0.01):
        temperature_raw = torch.ones(1, requires_grad=True, device="cuda")
        nll_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.LBFGS([temperature_raw], lr=lr, max_iter=iterations)
        softplus = nn.Softplus() #temperature must be > zero, Softplus could be used
        def closure():
            if torch.is_grad_enabled(): optimizer.zero_grad()
            #loss = nll_criterion(logits / softplus(temperature_raw.expand_as(logits)), labels)
            loss = nll_criterion(logits / temperature_raw.expand_as(logits), labels)
            if loss.requires_grad: loss.backward()
            return loss
        optimizer.step(closure)
        return temperature_raw

    def forward(self, logits, labels, temperature=1.0, onevsrest=False):
        logits_scaled = logits / temperature
        if(onevsrest):
            softmaxes = torch.sigmoid(logits_scaled) / torch.sum(torch.sigmoid(logits_scaled), dim=1, keepdim=True)
        else:
            softmaxes = torch.softmax(logits_scaled, dim=1)

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece