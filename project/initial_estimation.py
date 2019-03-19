import torch
from skimage.io import imread
from skimage.transform import rescale
import pdb
import numpy as np
import torch.nn.functional as F


def initialize_LK(blur_img, iterations=500, learning_rate=0.0001):
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
        i += 1
        if i % 2 == 1:
            out1 = conv(latent_img)
            norm1 = torch.norm((blur_img-out1), 2)

            # calculating total variation as regularization term.
            Gx = F.conv2d(conv.weight, s_x)
            Gy = F.conv2d(conv.weight, s_y)
            G = torch.norm(conv.weight, 2)

            energy = norm1 + G
            conv.zero_grad()
            energy.backward()
            print(norm1)
            with torch.no_grad():
                for param in conv.parameters():
                    param.data -= learning_rate*param.grad

        else:
            out1 = conv(latent_img)
            norm1 = torch.norm((blur_img-out1), 2)

            # calculating total variation as regularization term.
            Gx = F.conv2d(latent_img, s_x)
            Gy = F.conv2d(latent_img, s_y)
            G = torch.sum(torch.sqrt(torch.pow(Gx, 2) + torch.pow(Gy, 2)))
            energy = norm1 + 0.05
            print(norm1)
            conv.zero_grad()
            energy.backward()

            with torch.no_grad():
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
    L, K = initialize_LK(blur_img)
    pdb.set_trace()
    print(L.shape, K.shape)
