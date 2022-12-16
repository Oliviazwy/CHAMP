from utils import champ_motion3d as datasets
from utils.opt import Options
from model import AttModel_CHAMP
from model.CMS_model import *
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim
import random

def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)  # Numpy module.
    np.random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(opt):
    auto_encoder = AutoEncoder()  # bilstm
    auto_encoder.cuda()
    autoencoder_path_len = '{}/autoencoder.pth.tar'.format(opt.ckpt)
    print(">>> loading autoencoder len from '{}'".format(autoencoder_path_len))
    ae = torch.load(autoencoder_path_len)
    chosen_model = ae['chosen_model']
    chosen_model = torch.tensor(chosen_model)
    print("chosem_model= ",chosen_model)

    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = 72
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel_CHAMP.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n, model_prob=chosen_model)
    net_pred.cuda()
    model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))



    ckpt = torch.load(model_path_len)
    start_epoch = ckpt['epoch'] + 1
    err_best = ckpt['err']
    lr_now = ckpt['lr']
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])


    acts = ['still', 'sitDown', 'standUp', 'squat', 'squatUp',
                    'still_raiseUp', 'still_clockwise', 'still_counterclockwise', 'still_keepClose', 'still_keepFar',
                    'still_left', 'still_right', 'still_nod', 'still_shake', 'still_wave',

            'sitDown_clockwise', 'sitDown_counterclockwise', 'sitDown_keepClose', 'sitDown_keepFar', 'sitDown_left',
                  'sitDown_nod', 'sitDown_right', 'sitDown_shake', 'sitDown_wave', 'squatUp_clockwise', 
                  'squatUp_counterclockwise', 'squatUp_keepClose', 'squatUp_keepFar', 'squatUp_left', 'squatUp_nod', 
                  'squatUp_raiseUp', 'squatUp_right', 'squatUp_shake', 'squatUp_wave', 'squat_clockwise', 
                  'squat_counterclockwise', 'squat_keepClose', 'squat_keepFar', 'squat_left', 'squat_nod', 
                  'squat_raiseUp', 'squat_right', 'squat_shake', 'squat_wave', 'standUp_clockwise', 
                  'standUp_counterclockwise', 'standUp_keepClose', 'standUp_keepFar', 'standUp_left', 'standUp_nod', 
                  'standUp_raiseUp', 'standUp_right', 'standUp_shake', 'standUp_wave', 'sitDown_raiseUp']
    errs = np.zeros([len(acts) + 1, opt.output_n])
    for i, act in enumerate(acts):
        test_dataset = datasets.Datasets(input_n=20, output_n=10, skip_rate=1, split=2, actions=[act], act_num=i)
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        print('testing error: {:.3f}'.format(ret_test['#1']))
        ret_log = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
        errs[i] = ret_log
    errs[-1] = np.mean(errs[:-1], axis=0)
    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    value = np.concatenate([acts, errs.astype(np.str)], axis=1)
    log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    net_pred.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.arange(3,75,1)
    seq_in = opt.kernel_size

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=10, itera=itera)  # 32,20,1,72

        p3d_out_all = p3d_out_all / 2

        p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]  # 30,10,72

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]  # 32,10,75
        p3d_out[:, :, dim_used] = p3d_out_all
        p3d_out = p3d_out.reshape([-1, out_n, 25, 3])  # 32,10,25,3

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 25, 3])   # 32,30,25,3

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
    ret = {}
    m_p3d_h36 = m_p3d_h36 / n * 1000
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    torch_seed(2022)
    main(option)
