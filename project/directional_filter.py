"""Directional filtering of image."""
import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb


def directional_filter(theta, sigma):
    """Create directional_filter for direction theta using size sigma."""
    theta = np.pi * theta / 180

    kernel = np.zeros((sigma, sigma), np.float32)
    center = sigma // 2

    t = 0

    while(True):
        cords_a = np.array([center, center]) + t*np.array([np.cos(theta), np.sin(theta)])

        x, y = cords_a.astype(np.int)

        if y < sigma and x < sigma:
            if 0 <= x and 0 <= y:
                kernel[x, y] = np.exp(-np.square(t)/(2*np.square(sigma)))
        else:
            break

        cords_b = np.array([center, center]) - t*np.array([np.cos(theta), np.sin(theta)])
        x, y = cords_b.astype(np.int)

        if y < sigma and x < sigma:
            if 0 <= x and 0 <= y:
                kernel[x, y] = np.exp(-np.square(t)/(2*np.square(sigma)))
        else:
            break

        t = t+1

    return kernel


def apply_filter(image, filter):
    img = image.reshape(1, 1, image.shape[0], image.shape[1])
    img = torch.from_numpy(img)
    img = img.type('torch.FloatTensor')

    conv = torch.nn.Conv2d(1, 1, filter.shape, stride=1,
                           padding=(filter.shape[0]//2, filter.shape[1]//2),
                           bias=False)
    conv.weight = torch.nn.Parameter(torch.from_numpy(filter).float().unsqueeze(0).unsqueeze(0))

    output = conv(img)
    output = output.cpu().numpy()[0][0]

    return output


if __name__ == "__main__":
    kernel = directional_filter(135, 30)
    plt.imshow(kernel, 'gray')
    plt.show()
    pdb.set_trace()
