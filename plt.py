##plotting script to see and compare input/label/prediction

import numpy as np 
import matplotlib.pyplot as plt

input = np.load("input.npy")
label = np.load("label.npy")
inference = np.load("inference.npy")

plt.close('all')


for batch in range(20):

	plt.figure(batch)
	fig, axs = plt.subplots(4, 10)

	for i in range(len(input[batch])):
		axs[0, i].imshow(input[batch][i][0]*255)
		axs[2, i].imshow(input[batch][i][0]*255)

	for i in range(len(label[batch])):
		axs[1, i].imshow(label[batch][i][0]*255)

	for i in range(10):
		axs[3, i].imshow(inference[i][0][batch]*255)

	plt.show()