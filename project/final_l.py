import torch
import torch.nn as nn
from skimage.io import imread, imsave
import numpy as np
import pickle
from skimage.restoration import denoise_nl_means, estimate_sigma


def final_L(blur_img, latent_img, kernel, w3=0.05, learning_rate=0.0001):
    """Final latent image using non-local means denoising."""
    # Latent image
    latent_img = np.reshape(latent_img, (1, 1, latent_img.shape[0], latent_img.shape[1]))
    latent_img = torch.from_numpy(latent_img)
    latent_img = latent_img.type('torch.FloatTensor')
    latent_img.requires_grad = True

    # Blur image
    blur_img = np.reshape(blur_img, (1, 1, blur_img.shape[0], blur_img.shape[1]))
    blur_img = torch.from_numpy(blur_img)
    blur_img = blur_img.type('torch.FloatTensor')
    blur_img.requires_grad = False

    # Blur kernel
    conv = torch.nn.Conv2d(1, 1, (31, 31), stride=1, padding=15, bias=False)
    conv.weight = nn.Parameter(torch.nn.Parameter(torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)))

    if torch.cuda.device_count() > 0:
        latent_img = latent_img.cuda()
        blur_img = blur_img.cuda()
        conv = conv.cuda()

    normval = np.inf
    i = 0
    sigma = None
    optimizer = torch.optim.Adagrad([latent_img], lr=0.005)

    while True:
        # Kernel
        l_dash = latent_img.numpy()
        l_dash, sigma = NLM(l_dash, sigma)
        l_dash = torch.from_numpy(l_dash)
        l_dash = l_dash.type("torch.FloatTensor")

        if torch.cuda.device_count() > 0:
            l_dash = l_dash.cuda()

        out1 = conv(latent_img)

        norm1 = torch.norm((blur_img - out1), 2)
        G = w3 * torch.norm((latent_img - l_dash), 2)

        energy = norm1 + G
        optimizer.zero_grad()
        energy.backward()

        optimizer.step()

        i += 1

        if normval - norm1.item() < 0.0001:
            break
        normval = norm1.item()

    latent_img = latent_img.detach()
    latent_img = latent_img.cpu().numpy()[0][0]

    return latent_img


def NLM(img, sigma):
    """Non local means denoising."""
    sigma_est = np.mean(estimate_sigma(img, multichannel=False))
    patch_kw = dict(patch_size=5,
                    patch_distance=6,
                    multichannel=False)
    result = denoise_nl_means(img, h=0.8 * sigma_est, fast_mode=True,
                              **patch_kw)
    return result


if __name__ == "__main__":
    img_name = "test.jpg"

    blur_img = imread('test.jpg', as_gray=True)
    # blur_img = rescale(blur_img, 1.0/2, multichannel=False, )
    L, K = pickle.load(open(img_name.split(".")[0] + "_init.pkl", "rb"))
    L = final_L(blur_img, L, K)

    pickle.dump((L, K), open(img_name.split(".")[0] + "_init.pkl", "wb"))
    imsave(img_name.split(".")[0] + "L0.png", L)
    imsave(img_name.split(".")[0] + "K0.png", K)
