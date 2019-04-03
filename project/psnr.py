"""PSNR."""
import numpy as np
from skimage.io import imread
import sys


def psnr(img, gt):
    """Calculate PSNR."""
    noise = np.sum((img - gt)**2)
    size = img.shape[0] * img.shape[1]
    psnr = 10 * np.log(1/(noise/size))
    return psnr / np.log(10)

if __name__ == "__main__":
    assert len(sys.argv) > 0
    img_name = sys.argv[1]
    blur_img = sys.argv[2]

    img = imread(img_name, as_gray=True)
    blur = imread(blur_img, as_gray=True)

    img = img.astype(np.float) / 255
    blur_img = blur.astype(np.float) / 255
    print(psnr(img, blur_img))
