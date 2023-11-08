import torch.utils.data as data
import torch
import numpy as np

class MPMotion(data.Dataset):
    def __init__(self, data_path, in_len = 25, max_len = 50, concat_last = False):
        self.data = np.load(data_path, allow_pickle=True)
        # N, M, T, J, D = motion_data.shape
        print('Loading data:', data_path, self.data.shape)
        self.len = len(self.data)
        self.max_len = max_len
        self.in_len = in_len
        self.concat_last = concat_last
            
    def __getitem__(self, index):
        input_seq=self.data[index][:,:self.in_len,:]             
        output_seq=self.data[index][:,self.in_len:self.max_len,:]
        if self.concat_last:
            last_input=input_seq[:,-1:,:]
            output_seq = np.concatenate([last_input, output_seq], axis=1)
        return input_seq, output_seq
        
    def __len__(self):
        return self.len