import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# case1
# ntu 60 code
#                       3 |
#   23                  2 |                  21
#   24  11   10  9  8   20|  4   5   6    7  22
#                       1 |
#                       0 |
#                16       |  12
#                17       |  13
#                18       |  14
#                19       |  15

# case2
# ntu 60 code
#                      | 3
#   23                 | 2                   21
#   24  11   10  9  8  | 20  4   5   6    7  22
#                      | 1
#                      | 0
#                16    |     12
#                17    |     13
#                18    |     14
#                19    |     15


# case3
# ntu 60 code
#                       3
#   23                  2                   21
#   24  11   10  9  8       4   5   6    7  22
# ---------------------------------------------------
#                       20
#                       1
#                       0
#                16         12
#                17         13
#                18         14
#                19         15

# case4
# ntu 60 code
#                       3
#   23                  2                   21
#   24  11   10  9  8   20  4   5   6    7  22
# ---------------------------------------------------
#                       1
#                       0
#                16         12
#                17         13
#                18         14
#                19         15


# case5
# ntu 60 code
#                       3
#   23                  2                   21
#   24  11   10  9  8   20  4   5   6    7  22
#                       1
# ----------------------------------------------------
#                       0
#                16         12
#                17         13
#                18         14
#                19         15


class AutoEncoder_model(nn.Module):
    def __init__(self, upper_idx=None, lower_idx=None, hidden_layer=100, batch_size=32):

        super(AutoEncoder_model, self).__init__()

        self.input_layer_upper = upper_idx.size*3
        self.input_layer_lower = lower_idx.size*3
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.upper_idx = upper_idx
        self.lower_idx = lower_idx

        self.encoder_upper = nn.LSTM(self.input_layer_upper, self.hidden_layer, batch_first=True, bidirectional=True)
        self.encoder_lower = nn.LSTM(self.input_layer_lower, self.hidden_layer, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(self.hidden_layer * 4, int((self.input_layer_lower + self.input_layer_upper) / 2),
                               batch_first=True, bidirectional=True)

    def forward(self, input_x=None, is_train=True, chap_upper=None, chap_lower=None):
        device = 'cuda:0'
        upper_idx = np.concatenate((self.upper_idx * 3, self.upper_idx * 3 + 1, self.upper_idx * 3 + 2))
        lower_idx = np.concatenate((self.lower_idx * 3, self.lower_idx * 3 + 1, self.lower_idx * 3 + 2))
        upper_idx.sort()
        lower_idx.sort()

        if is_train == True:
            upper = input_x[:, :, upper_idx]
            lower = input_x[:, :, lower_idx]
            # encoder  32,30,100
            encoder_upper, (n, c) = self.encoder_upper(upper, (
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device),
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device)))
            encoder_lower, (n, c) = self.encoder_lower(lower, (
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device),
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device)))
            # concat
            encoder = torch.cat((encoder_upper, encoder_lower), 2)  # 32,30,400
            # decoder 32,30,72
            decoder, (n, c) = self.decoder(encoder, (
                torch.zeros(2, self.batch_size, int((self.input_layer_upper + self.input_layer_lower) / 2),
                            device=device),
                torch.zeros(2, self.batch_size, int((self.input_layer_upper + self.input_layer_lower) / 2),
                            device=device)))
        else:
            upper_used = chap_upper[:, :, upper_idx]
            lower_used = chap_lower[:, :, lower_idx]

            # encoder  32,30,100
            encoder_upper, (n, c) = self.encoder_upper(upper_used, (
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device),
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device)))
            encoder_lower, (n, c) = self.encoder_lower(lower_used, (
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device),
                torch.zeros(2, self.batch_size, self.hidden_layer, device=device)))
            # concat
            encoder = torch.cat((encoder_upper, encoder_lower), 2)  # 32,30,400
            # decoder 32,30,72
            decoder, (n, c) = self.decoder(encoder, (
                torch.zeros(2, self.batch_size, int((self.input_layer_upper + self.input_layer_lower) / 2),
                            device=device),
                torch.zeros(2, self.batch_size, int((self.input_layer_upper + self.input_layer_lower) / 2),
                            device=device)))

        return decoder.squeeze()



class AutoEncoder(nn.Module):
    def __init__(self, hidden_layer=100, batch_size=32):

        super(AutoEncoder, self).__init__()

        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        # model para
        upper_idx1 = np.array([3, 4, 5, 6, 11, 12, 13, 14, 20, 21])
        lower_idx1 = np.array([0, 1, 2, 7, 8, 9, 10, 15, 16, 17, 18, 19, 22, 23])
        upper_idx2 = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 21])
        lower_idx2 = np.array([7, 8, 9, 10, 15, 16, 17, 18, 22, 23])
        upper_idx3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23])
        lower_idx3 = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        upper_idx4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23])
        lower_idx4 = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18])
        upper_idx5 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23])
        lower_idx5 = np.array([11, 12, 13, 14, 15, 16, 17, 18])

        # model1
        self.autoencoder1 = AutoEncoder_model(upper_idx=upper_idx1, lower_idx=lower_idx1)
        # model2
        self.autoencoder2 = AutoEncoder_model(upper_idx=upper_idx2, lower_idx=lower_idx2)
        # model3
        self.autoencoder3 = AutoEncoder_model(upper_idx=upper_idx3, lower_idx=lower_idx3)
        # model4
        self.autoencoder4 = AutoEncoder_model(upper_idx=upper_idx4, lower_idx=lower_idx4)
        # model5
        self.autoencoder5 = AutoEncoder_model(upper_idx=upper_idx5, lower_idx=lower_idx5)

        #
        self.prob = nn.Parameter(torch.FloatTensor([0.2,0.2,0.2,0.2,0.2]))
        self.model_prob = torch.ones(5)
        self.tau = 0.1
        self.tau_decay = -0.00045


    def forward(self, input_x=None, is_train=True, chap_upper=None, chap_lower=None):
        device = 'cuda:0'

        if is_train == True:
            # model1
            augmented_data1 = self.autoencoder1(input_x)
            # model2
            augmented_data2 = self.autoencoder2(input_x)
            # model3
            augmented_data3 = self.autoencoder3(input_x)
            # model4
            augmented_data4 = self.autoencoder4(input_x)
            # model5
            augmented_data5 = self.autoencoder5(input_x)

            augmented_data_list = torch.cat(
                (torch.unsqueeze(augmented_data1, dim=0), torch.unsqueeze(augmented_data2, dim=0), torch.unsqueeze(augmented_data3, dim=0), torch.unsqueeze(augmented_data4, dim=0), torch.unsqueeze(augmented_data5, dim=0)), dim=0)
            # use gumble softmax to choose a pred
            # self.tau = self.tau * np.exp(self.tau_decay)
            self.model_prob = F.gumbel_softmax(self.prob, hard=True, tau=self.tau, dim=0).cuda()
            augmented_data = torch.mul(self.model_prob.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(augmented_data_list),
                             augmented_data_list).sum(0)

        else:
            model_idx = np.argmax(self.prob.cpu().detach().numpy())
            model_list = (self.autoencoder1, self.autoencoder2, self.autoencoder3, self.autoencoder4, self.autoencoder5)
            model = model_list[model_idx]
            augmented_data = model(is_train=is_train, chap_upper=chap_upper, chap_lower=chap_lower)

        return augmented_data,self.prob
