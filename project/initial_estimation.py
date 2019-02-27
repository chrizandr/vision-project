import torch
import cv2
import numpy as np
import torch.nn.functional as F

blur_img = cv2.imread('image.JPG', 0)
latent_img = blur_img.copy()

latent_img = np.reshape(latent_img, (1, 1, latent_img.shape[0], latent_img.shape[1]))
latent_img = torch.from_numpy(latent_img)
latent_img = latent_img.type('torch.FloatTensor').cuda()
latent_img.requires_grad = True

blur_img = np.reshape(blur_img, (1, 1, blur_img.shape[0], blur_img.shape[1]))
blur_img = torch.from_numpy(blur_img)
blur_img = blur_img.type('torch.FloatTensor').cuda()

# sobel filter
a = torch.Tensor([[1, 0, -1],
[2, 0, -2],
[1, 0, -1]]).cuda()
a = a.view((1,1,3,3))

b = torch.Tensor([[1, 2, 1],
[0, 0, 0],
[-1, -2, -1]]).cuda()
b = b.view((1,1,3,3))

conv = torch.nn.Conv2d(1, 1, (31, 31), stride=1, padding=15, bias=False).cuda()
torch.nn.init.normal_(conv.weight, mean=0, std=1)
learning_rate = 0.001

for i in range(5):
	if i%2==1:
	    out1 = conv(latent_img)
	    norm1 = torch.norm( (blur_img-out1), 2 ).cuda()

	    #calculating total variation as regularization term.
	    Gx = F.conv2d(conv.weight, a) 
	    Gy = F.conv2d(conv.weight, b)
	    G =  torch.sum(torch.sqrt(torch.pow(Gx,2)+ torch.pow(Gy,2)))

	    energy = norm1 + G
	    conv.zero_grad()
	    energy.backward()

	    with torch.no_grad():
	    	for param in conv.parameters():
	    		param.data -= learning_rate*param.grad
	    
	    for param in conv.parameters():
	    	param.requires_grad = False
	    latent_img.requires_grad = True
	else:
	    out1 = conv(latent_img)
	    norm1 = torch.norm( (blur_img-out1), 2 ).cuda()

	    #calculating total variation as regularization term.
	    Gx = F.conv2d(latent_img, a) 
	    Gy = F.conv2d(latent_img, b)
	    G =  torch.sum(torch.sqrt(torch.pow(Gx,2)+ torch.pow(Gy,2)))
	    energy = norm1 + G

	    conv.zero_grad()
	    energy.backward()

	    with torch.no_grad():
	    	latent_img -= learning_rate*latent_img.grad
	    
	    for param in conv.parameters():
	    	param.requires_grad = True
	    latent_img.requires_grad = False

latent_img.requires_grad=False
out_img = latent_img.cpu().numpy()[0][0]
print(out_img.shape)
cv2.imwrite('out_img.jpg', out_img)
