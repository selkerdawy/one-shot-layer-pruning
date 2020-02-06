
import torch.nn.functional as F
class MultipleOptimizer(object):
    def __init__(self,op = []):
        self.optimizers = op

    def add(self,new_opt):
        self.optimizers.append(new_opt)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def load_state_dict(self,param_ops):
        for op,op_new in zip(self.optimizers,param_ops):
            op.load_state_dict(op_new)

    def get_state_dict(self):
        state_dict = []
        for op in self.optimizers:
            state_dict.append(op.state_dict())
        return state_dict

    def update_lr(self,lr):
        for i,op in enumerate(self.optimizers):
            curlr = lr[i]
            for param_group in op.param_groups:
                param_group['lr'] = curlr

def getipnamerank():
    import os
    import subprocess
    f = open(os.environ["RANKMAP"],"r")
    lines = f.readlines();f.close()
    nodes_ranks = {line.split('\n')[0]:x for x,line in enumerate(lines)}
    ip = str(subprocess.Popen("ifconfig eth0 | grep 'inet ' | awk '{print $2}'", shell=True, stdout=subprocess.PIPE).stdout.read().decode()).split()[0]
    name = str(subprocess.Popen("nslookup %s | grep 'name = ' | awk '{print $4}'"%ip, shell=True, stdout=subprocess.PIPE).stdout.read().decode())
    name = name.split('.')[0]

    return ip, name, int(nodes_ranks[name])

def distillation(y, teacher_scores, labels, T):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    return l_kl

def getfilterw(in_channels,inputsz,filterw):
    if filterw == 'uniform':
        mask_weight = 1
    elif filterw == 'fw':
        mask_weight = 3*3*in_channels
    else:
        mask_weight = 3*3*in_channels*inputsz[0]*inputsz[1]
    #print('Mask weight with params %d, %s, %s is %d ' %(in_channels,str(inputsz),filterw,mask_weight))

    return mask_weight



