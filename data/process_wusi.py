import numpy as np
import os
import argparse

dir_list = ['Wusi']

def process_data(stride, sequence_len):
    # currently, we do not use ball position
    data = []
    for i in range(1):
        dir_base = f'./{dir_list[i]}/'
        # print(dir_base)
        files = os.listdir(dir_base)
        for file in files:
            pose_dir = dir_base+file
            poses = np.load(pose_dir,allow_pickle=True) # 5,x,15,3
            length = poses.shape[1]
            for j in range(0, length, stride):
                if j + sequence_len > length:
                    break
                pose = poses[:, j:j + sequence_len, :, :]
                data.append(pose)
    data = np.array(data)
    print(data.shape)
    save_path = f'data_undivided.npy'
    print(save_path)
    return data


def divide_data(ori_data, ratio):
    len_training = int(ratio * ori_data.shape[0])
    data = ori_data[:len_training,:,:,:,:]
    data=data.reshape(data.shape[0],5,-1,45)
    print('tarining set length = ',data.shape)
    np.save(f'training.npy',data)

    # ############################################
    # #test data
    data = ori_data[len_training:,:,:,:,:]
    data = data.reshape(data.shape[0],5,-1,45)
    print('testing set length = ',data.shape)
    np.save(f'testing.npy',data)
    


if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default='5')
    parser.add_argument('--sequence_len', type=int, default='50')
    parser.add_argument('--ratio', type=float, default='0.8')
    args = parser.parse_args()
    print(args.stride, args.sequence_len)
    stride = args.stride
    sequence_len = args.sequence_len
    ratio = args.ratio
    data = process_data(stride, sequence_len)
    divide_data(data, ratio)
    training_len = int(data.shape[0]*ratio)