import torch
import torch.optim
import net3
import scipy.io as io
import cv2
import time

def Recon_mat(image_path):
	data_x = image_path

	data_x = torch.from_numpy(data_x).float()
	data_x = data_x.unsqueeze(0)
	data_x = data_x.unsqueeze(1)
	mat_net = net3.mat_net()
	mat_net.load_state_dict(torch.load('snapshots/32/Umat.pth'))

	ref = mat_net(data_x)
	return ref


if __name__ == '__main__':
	# 训练集re'f
	# Mat = io.loadmat("data4/ReconCode/x_1.mat")
	# Mat = Mat["x"]
	time_start = time.time()
	Mat = io.loadmat("test/32.mat")
	Mat = Mat["ref"]
	a = Recon_mat(Mat)
	print(a.shape)
	a = a.resize(4, 1024, 1024)

	for i in range(4):
		b = a[i, :, :]
		print(b.shape)

		ref1 = b.cpu()
		ref_image = ref1.detach().numpy()
		image1_name = 'recon_image/imageNumber/32/' + str(i) + '.bmp'
		cv2.imwrite(image1_name, ref_image)
	time_end = time.time()
	print("Run time :", time_end-time_start)



