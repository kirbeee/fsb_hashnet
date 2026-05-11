import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 42
save = True
write_log = True
send_msg = False
send_log = False
method = 'FSB-HashNet'
remarks = 'Default'
dim = 1024
hash_dim = 512

# Other hyperparameters
batch_sub = 7
batch_samp = 2
batch_size = batch_sub * batch_samp
random_batch_size = batch_sub * batch_samp
test_batch_size = batch_size
epochs = 10
epochs_pre = 6
lr = 0.001
lr_sch = [6]
w_decay = 1e-5
dropout = 0.1
momentum = 0.9
pretrained_path = ''

# Activate, or deactivate BatchNorm2D
# bn_flag = 0, 1, 2
bn_flag = 1
bn_moment = 0.1
if bn_flag == 1:
    bn_moment = 0.1

# Softmax Classifiers
af_s = 64
af_m = 0.35

# BatchNorm and Network Description
net_descr = method
b2_flag = False

bn_moment = float(bn_moment)
dropout = float(dropout)
af_s = float(af_s)
af_m = float(af_m)
