import torch
from skimage.io import imread
from skimage.transform import rescale
import pdb
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def initialize_LK(blur_img, learning_rate=0.0001):
    """Value for latent image and kernel."""
    # blur_img = rescale(blur_img, 1.0 / 2.0, anti_aliasing=False)
    latent_img = blur_img.copy()
    latent_img = np.reshape(latent_img, (1, 1, latent_img.shape[0], latent_img.shape[1]))
    latent_img = torch.from_numpy(latent_img)
    latent_img = latent_img.type('torch.FloatTensor')
    latent_img.requires_grad = True

    blur_img = np.reshape(blur_img, (1, 1, blur_img.shape[0], blur_img.shape[1]))
    blur_img = torch.from_numpy(blur_img)
    blur_img = blur_img.type('torch.FloatTensor')

    # sobel filter
    s_x = torch.Tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    s_x = s_x.view((1, 1, 3, 3))

    s_y = torch.Tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    s_y = s_y.view((1, 1, 3, 3))

    conv = torch.nn.Conv2d(1, 1, (31, 31), stride=1, padding=15, bias=False)
    torch.nn.init.normal_(conv.weight, mean=0, std=1)

    normval = np.inf
    i = 0

    while True:
        # Minimizing Kernel
        out1 = conv(latent_img)
        norm1 = torch.norm((blur_img-out1), 2)

        Gx = F.conv2d(conv.weight, s_x)
        Gy = F.conv2d(conv.weight, s_y)
        G = torch.norm(conv.weight, 2)

        energy = norm1 + G
        conv.zero_grad()
        energy.backward()
        print('Updating conv weight', norm1)
        with torch.no_grad():
            for param in conv.parameters():
                if torch.sum(torch.isnan(param.grad))>0:
                     pdb.set_trace()
                param.data -= learning_rate*param.grad

        #  Minimizing latent img
        out1 = conv(latent_img)
        norm1 = torch.norm((blur_img-out1), 2)

        Gx = F.conv2d(latent_img, s_x)
        Gy = F.conv2d(latent_img, s_y)
        # G = torch.sum(torch.sqrt(torch.pow(Gx, 2) + torch.pow(Gy, 2)))
        G = torch.norm(latent_img, 2)
        if torch.isnan(G):
            pdb.set_trace()
        energy = norm1 + G
        
        print('Updating latent_img', norm1)
        latent_img.grad = torch.zeros(latent_img.grad.shape)
        energy.backward()

        with torch.no_grad():
            if torch.sum(torch.isnan(latent_img.grad))>0:
                pdb.set_trace()
            latent_img -= learning_rate*latent_img.grad

        if normval < norm1.item():
            break
        normval = norm1.item()

    latent_img.requires_grad = False
    for param in conv.parameters():
        param.requires_grad = False
    latent_img = latent_img.cpu().numpy()[0][0]
    kernel = conv.weight.cpu().numpy()[0][0]

    return latent_img, kernel


if __name__ == "__main__":
    blur_img = imread('test.jpg', as_gray=True)
    # L0, K = initialize_LK(blur_img[:, :, 0])
    # L1, K = initialize_LK(blur_img[:, :, 1])
    # L2, K = initialize_LK(blur_img[:, :, 2])
    # pdb.set_trace()
    # L = np.dstack([L0, L1, L2])
    L, K = initialize_LK(blur_img)
    #plt.subplot(211)
    plt.imshow(L, 'gray')
    # plt.subplot(212)
    # plt.imshow(blur_img, 'gray')
    plt.show()
    
    print(L.shape, K.shape)
