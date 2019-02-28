import torch
import cv2
import pdb
import numpy as np
import torch.nn.functional as F


def initialize_LK(blur_img, iterations=5, learning_rate=0.001):
    """Value for latent image and kernel."""
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
    pdb.set_trace()

    for i in range(iterations):
        if i % 2 == 1:
            out1 = conv(latent_img)
            norm1 = torch.norm((blur_img-out1), 2)

            # calculating total variation as regularization term.
            Gx = F.conv2d(conv.weight, s_x)
            Gy = F.conv2d(conv.weight, s_y)
            G = torch.sum(torch.sqrt(torch.pow(Gx, 2) + torch.pow(Gy, 2)))

            energy = norm1 + G
            conv.zero_grad()
            energy.backward()

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
            energy = norm1 + G

            conv.zero_grad()
            energy.backward()

            with torch.no_grad():
                latent_img -= learning_rate*latent_img.grad

    latent_img = latent_img.cpu().numpy()[0][0]
    kernel = conv.weight.cpu().numpy()[0][0]

    cv2.imwrite('out_img.jpg', latent_img)

    return latent_img, kernel


if __name__ == "__main__":
    blur_img = cv2.imread('image.JPG', 0)
    L, K = initialize_LK(blur_img)
    print(L.shape, K.shape)
