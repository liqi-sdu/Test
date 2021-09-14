import torch
import torch.utils.data as data
import os
import scipy.io as io

train_list = []


def populate_train_list(orig_mat_path, recon_mat_path):
    # 获取文件列表
    y_list = os.listdir(orig_mat_path)
    x_list = os.listdir(recon_mat_path)

    # 生成数据对  例如：['data2/ReconCode/x_0.mat', 'data2/CodeData/y_0.mat']
    for i in range(len(x_list)):
        y_root = orig_mat_path + y_list[i]
        x_root = recon_mat_path + x_list[i]

        train_list.append([y_root, x_root])
    return train_list
#
# a = populate_train_list("data/image/", "data/ref/")
# print(a)


class mat_loader(data.Dataset):
    def __init__(self, orig_mat_path, recon_mat_path):
        self.train_list = populate_train_list(orig_mat_path, recon_mat_path)
        # print(self.train_list)
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_y_path, data_x_path = self.train_list[index]
        data_y = io.loadmat(data_y_path)
        data_x = io.loadmat(data_x_path)
        # 提取字典中的数据
        data_y = data_y["image"]
        data_x = data_x["ref"]
        # 转化数据类型
        data_y = torch.from_numpy(data_y).float()
        data_x = torch.from_numpy(data_x).float()

        # 改成（1，256，256）, 符合卷积网络输入
        data_y = data_y.permute(2, 0, 1)
        data_x = data_x.unsqueeze(0)
        return data_y, data_x

    def __len__(self):
        return len(self.train_list)


# T = mat_loader("data/image/", "data/ref/")
# print(T[56])









