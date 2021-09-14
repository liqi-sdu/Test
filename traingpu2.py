import torch
import torch.nn as nn
import torch.optim
import os
import argparse
import dataloader2
import net3
import cv2
import scipy.io as io

# 参数初始化
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


# 导入mask数据，提取字典中的数据
mask1 = io.loadmat('mask/mask1.mat')['mask1']
mask2 = io.loadmat('mask/mask2.mat')['mask2']
mask3 = io.loadmat('mask/mask3.mat')['mask3']
mask4 = io.loadmat('mask/mask4.mat')['mask4']

mask1 = torch.from_numpy(mask1).float().cuda()
mask2 = torch.from_numpy(mask2).float().cuda()
mask3 = torch.from_numpy(mask3).float().cuda()
mask4 = torch.from_numpy(mask4).float().cuda()

# 损失函数
criterion = nn.MSELoss()


def sum_loss(ref, x, image, epoch, iteration):
	ref = ref.resize(4, 1024, 1024)
	image = image.resize(4, 1024, 1024)
	ref1 = ref[0, :, :]
	ref2 = ref[1, :, :]
	ref3 = ref[2, :, :]
	ref4 = ref[3, :, :]

	img1 = image[0, :, :]
	img2 = image[1, :, :]
	img3 = image[2, :, :]
	img4 = image[3, :, :]

	pro1 = img1 * mask1
	pro2 = img2 * mask2
	pro3 = img3 * mask3
	pro4 = img4 * mask4
	pro = pro1 + pro2 + pro3 + pro4

	loss1 = criterion(ref1, img1)
	loss2 = criterion(ref2, img2)
	loss3 = criterion(ref3, img3)
	loss4 = criterion(ref4, img4)
	loss5 = criterion(x, pro)

	Loss = loss1 + loss2 + loss3 + loss4 + loss5
	if iteration % 200 == 0:
		ref1 = ref1.cpu()
		ref_image = ref1.detach().numpy()
		image1_name = 'samples/' + str(epoch+1) + '.bmp'
		cv2.imwrite(image1_name, ref_image)

		image = img1.cpu()
		image = image.detach().numpy()
		image2_name = 'samples/' + str(epoch+1) + '_y.bmp'
		cv2.imwrite(image2_name, image)

	return Loss


def train(config):
	# 加载网络模型
	mat_net = net3.mat_net().cuda()
	mat_net.apply(weights_init)
	# 生成数据集
	train_dataset = dataloader2.mat_loader(config.y_path, config.x_path)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
	# 设置优化器
	optimizer = torch.optim.Adam(mat_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	mat_net.train()
	# 开始训练
	for epoch in range(config.num_epochs):
		for iteration, (y, x) in enumerate(train_loader):

			y = y.cuda()
			x = x.cuda()

			ref_image = mat_net(x)

			loss = sum_loss(ref_image, x, y, epoch, iteration)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if ((iteration + 1) % config.display_iter) == 0:
				print("epoch", ":", epoch, "Loss at iteration", iteration + 1, ":", loss.item())
			if ((epoch + 1) % config.snapshot_iter) == 0:
				torch.save(mat_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch+1) + '.pth')

		torch.save(mat_net.state_dict(), config.snapshots_folder + "Umat.pth")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--y_path', type=str, default="data/image/")
	parser.add_argument('--x_path', type=str, default="data/ref/")
	parser.add_argument('--lr', type=float, default=0.001)  # 学习率
	parser.add_argument('--weight_decay', type=float, default=0.001)  # 衰减率
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)  #
	parser.add_argument('--num_epochs', type=int, default=100)  # 迭代次数
	parser.add_argument('--train_batch_size', type=int, default=1)  # 训练集batch=1.
	parser.add_argument('--display_iter', type=int, default=500)  # 每x次打印一次损失函数
	parser.add_argument('--snapshot_iter', type=int, default=5)  # 每x次保存一次模型参数
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/32/")  # 模型参数保存地址
	parser.add_argument('--sample_output_folder', type=str, default="samples/")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)

	train(config)


