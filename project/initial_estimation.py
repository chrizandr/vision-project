import torch
from skimage.io import imread, imsave
# from skimage.transform import rescale
import pdb
import numpy as np
import pickle
# import torch.nn.functional as F
import matplotlib.pyplot as plt


def initialize_LK(blur_img, learning_rate=0.0001):
    """Value for latent image and kernel."""
    # Latent image
    latent_img = blur_img.copy()
    latent_img = np.reshape(latent_img, (1, 1, latent_img.shape[0], latent_img.shape[1]))
    latent_img = torch.from_numpy(latent_img)
    latent_img = latent_img.type('torch.FloatTensor')
    latent_img.requires_grad = True

    # Blur image
    blur_img = np.reshape(blur_img, (1, 1, blur_img.shape[0], blur_img.shape[1]))
    blur_img = torch.from_numpy(blur_img)
    blur_img = blur_img.type('torch.FloatTensor')

    # Blur kernel
    conv = torch.nn.Conv2d(1, 1, (31, 31), stride=1, padding=15, bias=False)
    torch.nn.init.normal_(conv.weight, mean=0, std=1)

    normval = np.inf
    i = 0

    while True:
        # Kernel
        out1 = conv(latent_img)

        norm1 = torch.norm((blur_img-out1), 2)
        G = torch.norm(conv.weight, 2)

        energy = norm1 + G
        conv.zero_grad()
        energy.backward()

        # Updating kernel
        with torch.no_grad():
            for param in conv.parameters():
                param.data -= learning_rate*param.grad

        # Latent image
        out1 = conv(latent_img)

        norm1 = torch.norm((blur_img-out1), 2)
        G = torch.norm(latent_img, 2)

        energy = norm1 + G
        latent_img.grad.data.zero_()
        energy.backward()

        # Update latent image
        with torch.no_grad():
            latent_img -= learning_rate*latent_img.grad

        print('Iteration ', i, "Norm = ", norm1.item())
        i += 1

        # Convergence
        if normval < norm1.item():
            break
        normval = norm1.item()

    latent_img.requires_grad = False
    for param in conv.parameters():
        param.requires_grad = False

    # To CPU
    latent_img = latent_img.cpu().numpy()[0][0]
    kernel = conv.weight.cpu().numpy()[0][0]
    blur_img = blur_img.cpu().numpy()[0][0]

    return latent_img, kernel, blur_img


if __name__ == "__main__":
    img_name = "test.jpg"

    blur_img = imread('test.jpg', as_gray=True)
    # blur_img = rescale(blur_img, 1.0/2, multichannel=False, )

    L, K, B = initialize_LK(blur_img)
    pdb.set_trace()

    pickle.dump((L, K), open(img_name.split(".")[0] + "_init.pkl", "wb"))
    imsave(img_name.split(".")[0] + "L0.png", L)
    imsave(img_name.split(".")[0] + "K0.png", K)
