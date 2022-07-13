import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tifffile
from tqdm import tqdm


class DoubleConv(nn.Module):
    """double 2d convolution with 3x3 kernel size"""

    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels_in, n_channels_out, 3, padding=1)
        self.conv2 = nn.Conv2d(n_channels_out, n_channels_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Denoiser(nn.Module):
    """simple convolutional neural network for denoising"""
    def __init__(self):
        super().__init__()
        self.conv1 = DoubleConv(1, 64)
        self.conv2 = DoubleConv(64, 64)
        self.conv3 = DoubleConv(64, 64)
        self.conv4 = DoubleConv(64, 64)
        self.outconv = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.outconv(x)
        return x


def even_odd_downsize(img):

    Ny, Nx = img.shape

    if Ny % 2 == 1:
        Ny -= 1
    if Nx % 2 == 1:
        Nx -= 1

    # halved row, "squeeze up"
    row_even_img = img[0:Ny:2, :]
    row_odd_img = img[1:Ny:2, :]

    # halved column, "squeeze left"
    col_even_img = img[:, 0:Nx:2]
    col_odd_img = img[:, 1:Nx:2]

    return (row_even_img[None, None, :, :], row_odd_img[None, None, :, :]), (
        col_even_img[None, None, :, :],
        col_odd_img[None, None, :, :],
    )


def denoise_frame(
    input_frame: torch.Tensor, last_n_frames=100, verbose=False
) -> np.array:
    """run denoising on a single 2D image
    
    Args:
    input_frame (torch.Tensor): 2D image as pytorch Tensor type
    last_n_frames (int): number of frames that will be averaged after validation
    step has reached the best MSE score.
    
    """
    # normalize input image so that it ranges from [0,1]
    img_max = input_frame.max()
    img_min = input_frame.min()
    img_range = img_max - img_min
    normimg = (input_frame - img_min) / img_range

    # generate downsized pairs of images (also adds batch and channel dimensions)
    squeeze_up_pair, squeeze_left_pair = even_odd_downsize(normimg)

    # define neural network and move to same device as input array
    nnet = Denoiser()
    nnet.to(input_frame.device)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = optim.Adam(nnet.parameters())

    niter = 0
    running_loss = 0.0
    random_index = [0, 1]

    last_cleaned = []
    best_mse = np.inf

    while niter < last_n_frames:
        # choose data pair randomly
        data = random.choice([squeeze_up_pair, squeeze_left_pair])
        # shuffle the 'left' or 'right' index from the pair
        random.shuffle(random_index)

        inputimg = data[random_index[0]]
        labelimg = data[random_index[1]]

        optimizer.zero_grad()

        # compute network prediction
        result = nnet(inputimg)

        # compute loss
        loss = criterion(nnet(inputimg), labelimg)

        # since batch is 1, no need to scale the running loss
        running_loss += loss.item()

        # back-propagate
        loss.backward()
        # update weights
        optimizer.step()

        # validation step
        with torch.no_grad():
            denoised_output = torch.sigmoid(nnet(normimg[None, None, :, :]))
            # compute mse
            arg = normimg - denoised_output
            mse = (arg * arg).mean()

            if verbose:
                print(f"\rmse = {mse:12.8e}, niter = {niter:d}...", end="")
            if mse < best_mse:
                # model still improving
                best_mse = mse
                # reset iteration counter
                niter = 0
                # reset collection of validation results
                last_cleaned = []
            else:
                cleanimg = denoised_output * img_range + img_min
                last_cleaned.append(cleanimg.cpu().numpy().squeeze())
                niter += 1

    if verbose:
        print("")

    # take the average of the last n frames
    avgcleaned = np.mean(last_cleaned, axis=0)

    return avgcleaned


def denoise_stack(input_stack, device, **kwargs):
    """denoise n-D image assuming the last two dimensions are rows and columns

    Args:
    input_stack (n-D numpy.array): multidimensional image to be denoised
    device (torch.device): GPU or CPU device to run denoising on. Use 
    torch.device("cuda") for NVIDIA GPU and torch.device("mps") for M1 GPU.


    """
    inputshape = input_stack.shape
    output = np.zeros(inputshape, dtype=np.float32)
    Nstacks = np.prod(inputshape[:-2])

    # assume that the last two axes are for the Row x Column of 2d images
    for nonsliceidx in tqdm(np.ndindex(inputshape[:-2]), total=Nstacks):
        # print(f"denoising ==> {100.0 * (n+1) / Nstacks: 0.2f}%...")
        idxslices = nonsliceidx + (slice(None), slice(None),)
        slice2d = input_stack[idxslices].astype(np.float32)
        wrkframe = torch.from_numpy(slice2d).to(device)
        output[idxslices] = denoise_frame(wrkframe)
    
    print("")
    return output


def readtiff(fn):
    """read tiff file and get pixel dimension
    
    Args:
        fn (str): tif image filename

    Returns:
        numpy array
    """
    data = tifffile.imread(fn)
    with tifffile.TiffFile(fn) as tif:
        page = tif.pages[0]
        xres = page.tags["XResolution"].value
        dx = xres[1] / xres[0]
        unit = pg.tags["ResolutionUnit"].value
    return data, dx, unit

def writetiff(fn, arr, dxy):
    """save array as tiff file with a given pixel size in micron
    
    Args:
        fn (str): filename
        arr (numpy nd-array): numpy array to be saved as tif file. Float64 data not supported
        dxy (float): pixel size in micron

    """
    tifffile.imwrite(fn, arr, resolution=(1/dxy, 1/dxy, 5))
