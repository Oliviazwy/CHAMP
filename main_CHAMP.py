from utils import champ_motion3d as datasets
from utils import champ_motion3d_synthesus as datasets_augmentation
from model import AttModel_CHAMP
from utils.opt import Options
from utils import util
from torch.utils.data import Dataset
from utils import log
from itertools import cycle
from model.CMS_model import *
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
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

    dataset = datasets.Datasets(input_n=20, output_n=10, skip_rate=1, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)
    valid_dataset = datasets.Datasets(input_n=20, output_n=10, skip_rate=1, split=1)
    print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)

    test_dataset = datasets.Datasets(input_n=20, output_n=10, skip_rate=1, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # composit motion synthesis module
    auto_encoder = AutoEncoder()  # bilstm
    optimizer_lstm = optim.Adam(auto_encoder.parameters(), lr=0.00001)
    epochs_lstm = 100
    auto_encoder.train()
    auto_encoder.cuda()

    dim_used = np.arange(3, 75, 1)
    loss_rec = 0

    # model_training
    for j in range(epochs_lstm):
        for i, (p3d_h36) in enumerate(data_loader):
            # print(i)
            batch_size, seq_n, _ = p3d_h36.shape

            p3d_h36 = p3d_h36.float().cuda()
            p3d_src = p3d_h36.clone()[:, :, dim_used]  # 32,30,72

            pred, prob = auto_encoder(input_x=p3d_src)  # (batch, seq_len, input_size)

            single_loss = torch.mean(
                torch.norm(pred.reshape([-1, 30, 24, 3]) - p3d_src.reshape([-1, 30, 24, 3]), dim=3))

            optimizer_lstm.zero_grad()
            single_loss.backward()
            optimizer_lstm.step()
            if j%10 ==0:
                print("Train Step:", i, " loss: ", single_loss)
            if j ==99:
                loss_rec += single_loss
    loss_rec = loss_rec/i
    print("loss= ", loss_rec)

    # motion synthesis
    dataset_upper = datasets_augmentation.Datasets(input_n=20, output_n=10, skip_rate=1, split=0, part_idx=0)
    print('>>> Training dataset length: {:d}'.format(dataset_upper .__len__()))
    data_loader_upper = DataLoader(dataset_upper, batch_size=opt.batch_size, shuffle=True, num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)
    dataset_lower = datasets_augmentation.Datasets(input_n=20, output_n=10, skip_rate=1, split=0, part_idx=1)
    print('>>> Training dataset length: {:d}'.format(dataset_lower.__len__()))
    data_loader_lower = DataLoader(dataset_lower, batch_size=opt.batch_size, shuffle=True, num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)
    data_augmentation = {}
    key = 0

    for i, (chap_upper) in enumerate(data_loader_upper):
        for j, (chap_lower) in enumerate(data_loader_lower):
            if random.randint(1, 10) > 9:
                champ_upper = chap_upper.float().cuda()
                champ_lower = chap_lower.float().cuda()
                upper = champ_upper.clone()[:, :, dim_used]  # 32,30,72
                lower = champ_lower.clone()[:, :, dim_used]  # 32,30,72

                augmented_data, prob = auto_encoder(is_train=False, chap_upper=upper,
                                                          chap_lower=lower)  # 32,30,72
                augmented_data = augmented_data.cpu().data.numpy()
                data_augmentation[key] = augmented_data  # 32,30,72
                key += 1

                # print(i, ' ', torch.cuda.memory_allocated())

    # np.save('CHAMP.npy', data_augmentation)
    print('key: ', key)
    print('******************************')
    print('prob: ', prob)
    print('******************************')
    chosen_model = np.argmax(prob.cpu().detach().numpy())
    log.save_autoencoder({'chosen_model': chosen_model,
                   'state_dict': auto_encoder.state_dict(),
                   'optimizer': optimizer_lstm.state_dict()},opt=opt)


    class Dataset_augmented(Dataset):
        def __init__(self, data_augmentation):
            self.data_augmentation = data_augmentation

        def __len__(self):
            return len(self.data_augmentation)

        def __getitem__(self, idx):
            return self.data_augmentation[idx]

    dataset_augmentation = Dataset_augmented(data_augmentation)
    data_loader_augmentation = DataLoader(dataset_augmentation, batch_size=1, shuffle=True, num_workers=0,
                                          pin_memory=True,
                                          drop_last=True)

    # main: prediction
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 75
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel_CHAMP.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n, model_prob=chosen_model)
    net_pred.cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))



    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, data_loader_augmentation=data_loader_augmentation, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, data_loader_augmentation=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()

    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
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
    st = time.time()

    # 混合训练
    if is_train == 0:
        for i, (p3d_h36, augmentation) in enumerate(zip(cycle(data_loader), data_loader_augmentation)):
            # print(i)
            batch_size, seq_n, _ = p3d_h36.shape
            # when only one sample in this batch
            if batch_size == 1 and is_train == 0:
                continue
            n += batch_size*2
            bt = time.time()

            # original data
            p3d_h36 = p3d_h36.float().cuda()
            p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
                [-1, seq_in + out_n, len(dim_used) // 3, 3])  # 32,20,24,3
            p3d_src = p3d_h36.clone()[:, :, dim_used]  # 32,30,72

            p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera, data_source='original')  # 32,20,1,75
            p3d_out_all = p3d_out_all/2

            p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
            p3d_out = p3d_out.reshape([-1, out_n, 25, 3])  # 32,10,25,3

            p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 25, 3])

            p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])
            # 2d joint loss:
            grad_norm = 0
            if is_train == 0:
                loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
                loss_all = loss_p3d
                optimizer.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
                optimizer.step()
                # update log values
                l_p3d += loss_p3d.cpu().data.numpy() * batch_size

            if is_train <= 1:  # if is validation or train simply output the overall mean error
                mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
            if i % 1000 == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                               time.time() - st, grad_norm))

            # synthesized actions
            augmentation = augmentation.squeeze(0)
            p3d_h36_augmentation = np.zeros([32, 30, 75])
            p3d_h36_augmentation[:, :, dim_used] = augmentation  # 32,30,72
            p3d_h36_augmentation = torch.tensor(p3d_h36_augmentation).float().cuda()

            p3d_sup_augmentation = p3d_h36_augmentation.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
                [-1, seq_in + out_n, len(dim_used) // 3, 3])  # 32,20,24,3
            p3d_src_augmentation = p3d_h36_augmentation.clone()[:, :, dim_used]  # 32,30,72

            p3d_out_all_augmentation = net_pred(p3d_src_augmentation, input_n=in_n, output_n=out_n, itera=itera, data_source='augmentation')  # 32,20,1,75

            p3d_out_augmentation = p3d_h36_augmentation.clone()[:, in_n:in_n + out_n]
            p3d_out_augmentation[:, :, dim_used] = p3d_out_all_augmentation[:, seq_in:, 0]
            p3d_out_augmentation = p3d_out_augmentation.reshape([-1, out_n, 25, 3])  # 32,10,25,3

            p3d_h36_augmentation = p3d_h36_augmentation.reshape([-1, in_n + out_n, 25, 3])

            p3d_out_all_augmentation = p3d_out_all_augmentation.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])
            # 2d joint loss:
            grad_norm = 0
            if is_train == 0:
                loss_p3d = torch.mean(torch.norm(p3d_out_all_augmentation[:, :, 0] - p3d_sup_augmentation, dim=3))
                loss_all = loss_p3d
                optimizer.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
                optimizer.step()
                # update log values
                l_p3d += loss_p3d.cpu().data.numpy() * batch_size

            if is_train <= 1:  # if is validation or train simply output the overall mean error
                mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36_augmentation[:, in_n:in_n + out_n] - p3d_out_augmentation, dim=3))
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size

            if i % 1000 == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader_augmentation), time.time() - bt,
                                                               time.time() - st, grad_norm))

    else:
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
                [-1, seq_in + out_n, len(dim_used) // 3, 3])  # 32,20,24,3
            p3d_src = p3d_h36.clone()[:, :, dim_used]  # 32,30,72

            p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)  # 32,20,1,75
            p3d_out_all = p3d_out_all/2

            p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
            p3d_out = p3d_out.reshape([-1, out_n, 25, 3])  # 32,10,25,3

            p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 25, 3])

            p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])
            

            # 2d joint loss:
            grad_norm = 0
            if is_train == 0:
                loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
                loss_all = loss_p3d
                optimizer.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
                optimizer.step()
                # update log values
                l_p3d += loss_p3d.cpu().data.numpy() * batch_size

            if is_train <= 1:  # if is validation or train simply output the overall mean error
                mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
            else:
                mpjpe_p3d_h36 = torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3)  # 32,10,25
                mpjpe_p3d_h36 = torch.mean(mpjpe_p3d_h36, dim=2)  # 32,10
                mpjpe_p3d_h36 = torch.sum(mpjpe_p3d_h36, dim=0)  # 10
                # mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
            if i % 1000 == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                               time.time() - st, grad_norm))


    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n *1000

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n *1000
    else:
        m_p3d_h36 = m_p3d_h36 / n *1000
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    torch_seed(2022)
    option = Options().parse()
    main(option)
