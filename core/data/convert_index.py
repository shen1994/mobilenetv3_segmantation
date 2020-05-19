import os
import cv2
import shutil
import numpy as np

if __name__ == "__main__":
	
	color_dict = {(0, 128, 128): 1, (128, 0, 0): 5, (0, 0, 128): 3, 
				  (128, 0, 128): 4, (0, 128, 0): 2, (0, 0, 0): 0}

	'''
	image = cv2.imread("../../dataset/label/train/1.png", 1)
	index = 0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if tuple(image[i, j, :]) not in color_dict.keys():
				color_dict[tuple(image[i, j, :])] = index
				index += 1
				print(color_dict)
	'''

	flag = "val/"

	label_color_path = "../../dataset/label_color/" + flag
	label_save_path = "../../dataset/label/"

	if not os.path.exists(label_save_path):
		os.mkdir(label_save_path)

	if os.path.exists(label_save_path + flag):
		shutil.rmtree(label_save_path + flag)
	os.mkdir(label_save_path + flag)

	for name in os.listdir(label_color_path):
		color_image_path = label_color_path + name
		color_image = cv2.imread(color_image_path, 1)

		index_image = np.zeros(shape=(color_image.shape[0], color_image.shape[1]), dtype=np.uint8)

		for i in range(color_image.shape[0]):
			for j in range(color_image.shape[1]):
				index_image[i, j] = color_dict[tuple(color_image[i, j, :])]
		cv2.imwrite(label_save_path + flag + name, index_image)

		print(color_image_path)
		