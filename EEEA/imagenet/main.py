import torch

# download pre-trained from ->
model = torch.jit.load('eeea_c.pt')

model.eval()

# input must be resized to 192x192 pixel 
input = torch.rand(1, 3, 192, 192)

output = model(input)