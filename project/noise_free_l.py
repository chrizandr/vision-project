import torch
from torch.autograd import Variable
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import pdb
import numpy as np
import pickle
# import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_l_zero(b_theta, latent_img, kernel, w1=0.05, w2=1, learning_rate=0.0002, verbose=False):
    """Value for latent image and kernel."""
    latent_img_shape = latent_img.shape
    l_1 = rescale(latent_img, 1.0/2, multichannel=False, anti_aliasing=False)

    latent_img = np.reshape(latent_img, (1, 1, latent_img.shape[0], latent_img.shape[1]))
    latent_img = Variable(torch.FloatTensor(latent_img).cuda(), requires_grad=True)

    b_theta = np.reshape(b_theta, (1, 1, b_theta.shape[0], b_theta.shape[1]))
    b_theta = torch.from_numpy(b_theta)
    b_theta = b_theta.type('torch.FloatTensor')
    b_theta.requires_grad = False

    conv_k = torch.nn.Conv2d(1, 1, kernel_size=31, stride=1, padding=15, bias=False)
    conv_k.weight = torch.nn.Parameter(torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0))

    # Tensors to compute gradients
    s_x = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    conv_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = torch.nn.Parameter(torch.from_numpy(s_x).float().unsqueeze(0).unsqueeze(0))

    s_y = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    conv_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = torch.nn.Parameter(torch.from_numpy(s_y).float().unsqueeze(0).unsqueeze(0))

    for param in conv_x.parameters():
        param.requires_grad = False
    for param in conv_y.parameters():
        param.requires_grad = False
    for param in conv_k.parameters():
        param.requires_grad = False

    # Downscale, find gradient and upscale u(delta(L_1))
    l_1 = np.reshape(l_1, (1, 1, l_1.shape[0], l_1.shape[1]))
    l_1 = torch.from_numpy(l_1)
    l_1 = l_1.type('torch.FloatTensor')
    l_1.requires_grad = False

    L_x = conv_x(l_1)
    L_y = conv_y(l_1)
    delta_l_1 = torch.sqrt(torch.pow(L_x, 2) + torch.pow(L_y, 2))
    delta_l_1.requires_grad = False
    delta_l_1 = delta_l_1.cpu().numpy()[0][0]

    udelta_l = resize(delta_l_1, latent_img_shape, anti_aliasing=True)
    udelta_l = np.reshape(udelta_l, (1, 1, udelta_l.shape[0], udelta_l.shape[1]))
    udelta_l = torch.from_numpy(udelta_l)
    udelta_l = udelta_l.type('torch.FloatTensor')
    udelta_l.requires_grad = False

    # Gradient of blur image delat(B_0)
    B_x = conv_x(b_theta)
    B_y = conv_y(b_theta)
    delta_B = torch.sqrt(torch.pow(B_x, 2) + torch.pow(B_y, 2))
    delta_B = delta_B.detach()

    if torch.cuda.device_count() > 0:
        delta_B = delta_B.cuda()
        udelta_l = udelta_l.cuda()
        conv_x = conv_x.cuda()
        conv_y = conv_y.cuda()
        conv_k = conv_k.cuda()

    normval = np.inf
    i = 0
    optimizer = torch.optim.Adagrad([latent_img], lr=0.005)
    while True:
        # Minimizing Kernel
        L_x = conv_x(latent_img)
        L_y = conv_y(latent_img)
        delta_L = torch.sqrt(torch.pow(L_x, 2) + torch.pow(L_y, 2))

        out = conv_k(delta_L)

        norm1 = torch.norm((delta_B - out), 2)
        norm2 = w1 * torch.norm((delta_L - udelta_l), 2)
        R = w2 * torch.norm(delta_L, 2)

        energy = norm1 + norm2 + R
        optimizer.zero_grad()
        energy.backward()

        optimizer.step()

        if verbose and i % 100 == 0:
            print('Iteration ', i, "Norm = ", norm1.item())
        i += 1

        if normval - norm1.item() < 0.0001:
            break
        normval = norm1.item()

    latent_img = latent_img.detach()
    latent_img = latent_img.cpu().numpy()[0][0]

    return latent_img


if __name__ == "__main__":
    img_name = "test.jpg"

    blur_img = imread('test.jpg', as_gray=True)
    L, K = pickle.load(open(img_name.split(".")[0] + "_init.pkl", "rb"))
    # blur_img = rescale(blur_img, 1.0/2, multichannel=False, )

    latent_img = compute_l_zero(blur_img, L, K, verbose=True)
    pdb.set_trace()

    imsave(img_name.split(".")[0] + "L0.png", L)
    imsave(img_name.split(".")[0] + "K0.png", K)
