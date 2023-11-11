import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import cnet
import scipy.io


## LOAD DATA
loadFile = './datasets/mimic/ppg_bp_waveform.mat'
loaddata = scipy.io.loadmat(loadFile)
X = loaddata['X']  # style 1 training minibatches of size [frames,batchSize,dim]
Y = loaddata['Y']  # style 2 training minibatches of size [frames,batchSize,dim]
x = loaddata['feats_x']  # style 1 test data of size [frames,dim]
y = loaddata['feats_y']  # style 2 test data of size [frames,dim]

## PARAMETERS
residual_channels = 256
filter_width = 11
dilations = [1, 1, 1, 1, 1, 1]
input_channels = X[0][0].shape[2]
output_channels = X[0][0].shape[2]
cond_dim = None
postnet_channels = 256
do_postproc = True
do_gu = True

# residual_channels = 2
# filter_width = 3
# dilations = [1, 1, 1, 1]
# input_channels = X[0][0].shape[2]
# output_channels = X[0][0].shape[2]
# cond_dim = None
# postnet_channels= 5
# do_postproc = False

# Define the model architectures for G, F, D_y, and D_x using nn.Module
G = cnet.CNET( name='G',
               input_channels=input_channels,
               output_channels=output_channels,
               residual_channels=residual_channels,
               filter_width=filter_width,
               dilations=dilations,
               postnet_channels=postnet_channels,
               cond_dim=cond_dim,
               do_postproc=do_postproc,
               do_GU=do_gu)

F = cnet.CNET( name='F',
               input_channels=input_channels,
               output_channels=output_channels,
               residual_channels=residual_channels,
               filter_width=filter_width,
               dilations=dilations,
               postnet_channels=postnet_channels,
               cond_dim=cond_dim,
               do_postproc=do_postproc,
               do_GU=do_gu)

D_x = cnet.CNET( name='D_x',
                 input_channels=output_channels,
                 output_channels=1,
                 residual_channels=residual_channels,
                 filter_width=filter_width,
                 dilations=dilations,
                 postnet_channels=postnet_channels,
                 cond_dim=cond_dim,
                 do_postproc=do_postproc,
                 do_GU=do_gu)

D_y = cnet.CNET( name='D_y',
                 input_channels=output_channels,
                 output_channels=1,
                 residual_channels=residual_channels,
                 filter_width=filter_width,
                 dilations=dilations,
                 postnet_channels=postnet_channels,
                 cond_dim=cond_dim,
                 do_postproc=do_postproc,
                 do_GU=do_gu)

# optimizer parameters
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
n_critic = 500

# training parameters
num_epochs = 200  # 3#200

# Create optimizers for G, F, and D
optimizer_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=adam_lr,
                         betas=(adam_beta1, adam_beta2))
optimizer_D = optim.Adam(list(D_y.parameters()) + list(D_x.parameters()), lr=adam_lr,
                         betas=(adam_beta1, adam_beta2))

# Convert the placeholders to PyTorch tensors
x_real = torch.tensor(X, dtype=torch.float32)  # Assuming X is your input data
y_real = torch.tensor(Y, dtype=torch.float32)  # Assuming Y is your output data

# Loop for training
for epoch in range(num_epochs):
    # Train discriminator
    idx = torch.randperm(X.shape[0])
    for batch_i in range(X.shape[0]):
        for critic_i in range(n_critic):
            optimizer_D.zero_grad()
            lossD, lossD_gan_x, lossD_gan_y, lossD_grad_y, lossD_grad_x, lossD_zero_x, lossD_zero_y = ...  # Compute the loss for D
            lossD.backward()
            optimizer_D.step()

    # Train generator
    for batch_i in range(X.shape[0]):
        optimizer_G.zero_grad()
        if epoch < 50:
            lossGen, lossG, lossF, loss_reconX, loss_reconY, loss_idX, loss_idY = ...  # Compute the loss for generator
        else:
            lossGen, lossG, lossF, loss_reconX, loss_reconY = ...  # Compute the loss for generator
        lossGen.backward()
        optimizer_G.step()

    print("Errors for epoch %d : Gen loss is %f, D loss is %f " % (epoch, lossGen, lossD))

# Inference
with torch.no_grad():
    # Perform inference using the trained models
    y_pred, x_recon, x_pred_id = ...  # Perform inference for x
    x_pred, y_recon, y_pred_id = ...  # Perform inference for y

# Saving the results
saveFile1 = './pred_res.pt'
saveFile2 = './errors.pt'
torch.save({'y_pred': y_pred, 'x_recon': x_recon, 'x_pred_id': x_pred_id}, saveFile1)

scipy.io.savemat(saveFile1, {"y_pred": y_pred,
                             "x_recon": x_recon,
                             "x_pred_id": x_pred_id,
                             "x_pred": x_pred,
                             "y_recon": y_recon,
                             "y_pred_id": y_pred_id})