from torch.utils.data import Dataset
import numpy as np
import torch




class Datasets(Dataset):

    def __init__(self,input_n,output_n,skip_rate, actions=None, split=0, act_num =None):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 2 testing, 1 validation
        :param sample_rate:
        """
        self.path_to_data = "./datasets/ChAP/"
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 1
        self.p3d = {}
        self.data_idx = []
        self.act_num = act_num
        seq_len = self.in_n + self.out_n
        subs = np.array(['x_train', 'x_validation','x_test'])
        atomic_action = ['still', 'sitDown', 'standUp', 'squat', 'squatUp', 'raiseUp', 'keepFar', 'keepClose',
                  'left', 'right', 'clockwise', 'counterclockwise', 'nod', 'shake', 'wave']
        compose_action = ['sitDown_clockwise', 'sitDown_counterclockwise', 'sitDown_keepClose', 'sitDown_keepFar', 'sitDown_left',
                  'sitDown_nod', 'sitDown_right', 'sitDown_shake', 'sitDown_wave', 'squatUp_clockwise',
                  'squatUp_counterclockwise', 'squatUp_keepClose', 'squatUp_keepFar', 'squatUp_left', 'squatUp_nod',
                  'squatUp_raiseUp', 'squatUp_right', 'squatUp_shake', 'squatUp_wave', 'squat_clockwise',
                  'squat_counterclockwise', 'squat_keepClose', 'squat_keepFar', 'squat_left', 'squat_nod',
                  'squat_raiseUp', 'squat_right', 'squat_shake', 'squat_wave', 'standUp_clockwise',
                  'standUp_counterclockwise', 'standUp_keepClose', 'standUp_keepFar', 'standUp_left', 'standUp_nod',
                  'standUp_raiseUp', 'standUp_right', 'standUp_shake', 'standUp_wave', 'sitDown_raiseUp']

        data = np.load('datasets/CHAMP/CHAMP.npz')
        # data format：
        # train_x[label_idx, N, T, C]
        # [label_idx, -1, -1, -1] = N_num
        # [label_idx, N, -1, -1] = T_num
        subs = subs[split]

        self.dim_used = np.arange(0,75,1)

        '''
        train_x = data['x_train']

        test_x = data['x_test']
        
        test_atomic = data['atomic_test']

        validation_x = data['x_validation']
        '''

        key = 0

        if self.split <= 1:
            print("Reading {0}".format(subs))
            if self.split == 0:
                file = data['x_train']
                file_used = np.tile(file[:, :, :, 0:3], (1, 1, 25))
                file_used = file - file_used
            if self.split == 1:
                file = data['x_validation']
                file_used = np.tile(file[:, :, :, 0:3], (1, 1, 25))
                file_used = file - file_used
            label, b, n, d = file_used.shape

            for l in range(label):
                for n in range(int(file_used[l, -1, -1, -1])):
                    the_sequence_num = int(file_used[l, n, -1, -1])
                    if the_sequence_num >= 30:
                        even_list = range(0, the_sequence_num, self.sample_rate)
                        valid_frames = np.arange(0, the_sequence_num - seq_len + 1, skip_rate)
                        the_sequence = np.array(file_used[l, n, even_list, :])  # （the_sequence_num,75）
                    else:
                        idx = [0] * (30 - the_sequence_num) + list(range(0, the_sequence_num, 1))
                        valid_frames = np.arange(0, 1, skip_rate)
                        the_sequence = np.array(file_used[l, n, idx, :])  # 30,75

                    #  data_idx
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    the_sequence = torch.from_numpy(the_sequence).float().cuda()
                    self.p3d[key] = the_sequence.cpu().data.numpy()
                    key += 1


        if self.split == 2:
            print("Reading {0}".format(subs))
            file_atomic = data['atomic_test']  # 15,20,150,75
            file_atomic_used = np.tile(file_atomic[:, :, :, 0:3], (1, 1, 25))
            file_atomic_used = file_atomic - file_atomic_used
            label_atomic, b_atomic, n_atomic, d_atomic = file_atomic_used.shape

            file = data['x_test']  # 40,70,150,75
            file_used = np.tile(file[:, :, :, 0:3], (1, 1, 25))
            file_used = file - file_used
            label, b, n, d = file_used.shape

            if self.act_num == None:
                for l in range(label):
                    for n in range(int(file_used[l, -1, -1, -1])):
                        the_sequence_num = int(file_used[l, n, -1, -1])
                        if the_sequence_num >= 30:
                            even_list = range(0, the_sequence_num, self.sample_rate)
                            valid_frames = np.arange(0, the_sequence_num - seq_len + 1, skip_rate)
                            the_sequence = np.array(file_used[l, n, even_list, :])  # （the_sequence_num,75）
                        else:
                            idx = [0] * (30 - the_sequence_num) + list(range(0, the_sequence_num, 1))
                            valid_frames = np.arange(0, 1, skip_rate)
                            the_sequence = np.array(file_used[l, n, idx, :])  # 30,75

                        #  data_idx
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        self.p3d[key] = the_sequence.cpu().data.numpy()
                        key += 1

                for l in range(label_atomic):
                    for n in range(int(file_atomic_used[l, -1, -1, -1])):
                        the_sequence_num = int(file_atomic_used[l, n, -1, -1])
                        if the_sequence_num >= 30:
                            even_list = range(0, the_sequence_num, self.sample_rate)
                            valid_frames = np.arange(0, the_sequence_num - seq_len + 1, skip_rate)
                            the_sequence = np.array(file_atomic_used[l, n, even_list, :])  # （the_sequence_num,75）
                        else:
                            idx = [0] * (30 - the_sequence_num) + list(range(0, the_sequence_num, 1))
                            valid_frames = np.arange(0, 1, skip_rate)
                            the_sequence = np.array(file_atomic_used[l, n, idx, :])  # 30,75

                        #  data_idx
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        self.p3d[key] = the_sequence.cpu().data.numpy()
                        key += 1

            else:
                for l in range(label):
                    if l == self.act_num-15:
                        for n in range(int(file_used[l, -1, -1, -1])):
                            the_sequence_num = int(file_used[l, n, -1, -1])
                            if the_sequence_num >= 30:
                                even_list = range(0, the_sequence_num, self.sample_rate)
                                valid_frames = np.arange(0, the_sequence_num - seq_len + 1, skip_rate)
                                the_sequence = np.array(file_used[l, n, even_list, :])  # （the_sequence_num,75）
                            else:
                                idx = [0] * (30 - the_sequence_num) + list(range(0, the_sequence_num, 1))
                                valid_frames = np.arange(0, 1, skip_rate)
                                the_sequence = np.array(file_used[l, n, idx, :])  # 30,75

                            #  data_idx
                            tmp_data_idx_1 = [key] * len(valid_frames)
                            tmp_data_idx_2 = list(valid_frames)
                            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                            the_sequence = torch.from_numpy(the_sequence).float().cuda()
                            self.p3d[key] = the_sequence.cpu().data.numpy()
                            key += 1

                    else:
                        continue


                for l in range(label_atomic):
                    if l == self.act_num:
                        for n in range(int(file_atomic_used[l, -1, -1, -1])):
                            the_sequence_num = int(file_atomic_used[l, n, -1, -1])
                            if the_sequence_num >= 30:
                                even_list = range(0, the_sequence_num, self.sample_rate)
                                valid_frames = np.arange(0, the_sequence_num - seq_len + 1, skip_rate)
                                the_sequence = np.array(file_atomic_used[l, n, even_list, :])  # （the_sequence_num,75）
                            else:
                                idx = [0] * (30 - the_sequence_num) + list(range(0, the_sequence_num, 1))
                                valid_frames = np.arange(0, 1, skip_rate)
                                the_sequence = np.array(file_atomic_used[l, n, idx, :])  # 30,75

                            #  data_idx
                            tmp_data_idx_1 = [key] * len(valid_frames)
                            tmp_data_idx_2 = list(valid_frames)
                            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                            the_sequence = torch.from_numpy(the_sequence).float().cuda()
                            self.p3d[key] = the_sequence.cpu().data.numpy()
                            key += 1

                    else:
                        continue








    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]
