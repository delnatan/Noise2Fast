import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tifffile
from tqdm import tqdm


EPS = torch.tensor(1e-6)


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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.outconv(x)
        x = self.sigmoid(x)
        return x


class PoissonMLE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        x = torch.maximum(inputs, EPS)
        y = torch.maximum(targets, EPS)
        arg = x - y - y * torch.log(x/y)
        return torch.sum(arg)


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


def sparse_hessian_regularizer(x):
    Dy, Dx = torch.gradient(x, dim=(-2, -1))
    Dyy, Dyx = torch.gradient(Dy, dim=(-2, -1))
    Dxy, Dxx = torch.gradient(Dx, dim=(-2, -1))
    return torch.sum(
        torch.abs(Dyy) + torch.abs(Dxx) + torch.abs(Dxy) + torch.abs(Dyx)
    )


def pos_neg_entropy(x, m=1.0):
    Dy, Dx = torch.gradient(x, dim=(-2, -1))
    psi_y = torch.sqrt(Dy * Dy + 4 * m**2)
    psi_x = torch.sqrt(Dx * Dx + 4 * m**2)
    s_y = psi_y - 2*m - Dy * torch.log((psi_y + Dy)/(2*m))
    s_x = psi_x - 2*m - Dx * torch.log((psi_x + Dx)/(2*m))
    return torch.sum(s_y) + torch.sum(s_x)


def ER_penalty(x):
    Dy, Dx = torch.gradient(x, dim=(-2, -1))
    Dyy, Dyx = torch.gradient(Dy, dim=(-2, -1))
    Dxy, Dxx = torch.gradient(Dx, dim=(-2, -1))
    arg = x*x + Dyy*Dyy + Dxx*Dxx + Dxy*Dxy + Dyx*Dyx
    return torch.sum(torch.log(arg + 1e-7))


def denoise_frame_ptype(
        input_frame: torch.Tensor,
        max_iter: int = 100,
        regularization_weight: int = 0,
        patience: int = 10,
        reltol: int = 1e-4,
) -> np.array:

    # normalize input image
    img_max = input_frame.max()
    img_min = input_frame.min()
    img_range = img_max - img_min
    normimg = (input_frame - img_min) / img_range

    # generate downsized pairs of images
    squeeze_up_pair, squeeze_left_pair = even_odd_downsize(normimg)

    # define neural net
    model = Denoiser()
    model.to(input_frame.device)

    poisson_loss = PoissonMLE()
    # bce_loss = nn.BCELoss(reduction="sum")
    # mse_loss = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters())

    random_index = [0, 1]
    niter = 0
    counter = 0
    mse_list = []
    train_loss = []
    previous_loss = None

    while niter < max_iter:
        niter += 1
        # choose working data from downsized pair
        data = random.choice([squeeze_up_pair, squeeze_left_pair])
        random.shuffle(random_index)

        input_image = data[random_index[0]]
        target_image = data[random_index[1]]

        # loss = poisson_loss(model(input_image), target_image)
        output = model(input_image)
        loss = poisson_loss(output, target_image)

        if regularization_weight > 1e-8:
            # add regularizer to `loss`
            # pen1 = sparse_hessian_regularizer(output)
            # pen1 = pos_neg_entropy(output, m=1.0)
            pen1 = ER_penalty(output)
            loss += regularization_weight * pen1

        current_loss = loss.item()
        train_loss.append(current_loss)

        # update network parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation step
        with torch.no_grad():
            output_image = model(normimg[None, None, :, :])
            mse = torch.mean((normimg - output_image)**2)
            mse_list.append(mse.cpu().numpy())

        # compute convergence
        if previous_loss is not None:
            relative_loss = abs(current_loss - previous_loss) / abs(previous_loss)
            print(f"\r (iter {niter:d}) rel_loss = {relative_loss:12.5f}", end="")
            if relative_loss < reltol:
                counter += 0
                if counter >= patience:
                    print("No more improvements. Stopping iteration")
                    break
            else:
                counter = 0

        previous_loss = current_loss

    # create new line
    print("")
    clean_image = output_image * img_range + img_min

    return clean_image.cpu().squeeze().numpy(), mse_list, train_loss


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


def denoise_stack(input_stack, device, last_n_frames=100, verbose=False):
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
        idxslices = nonsliceidx + (
            slice(None),
            slice(None),
        )
        slice2d = input_stack[idxslices].astype(np.float32)
        wrkframe = torch.from_numpy(slice2d).to(device)
        output[idxslices] = denoise_frame(wrkframe,
                                          last_n_frames=last_n_frames,
                                          verbose=verbose)

    print("")
    return output


def readtiff_with_metadata(fn):
    """read tiff file and returns pixel and imagej metadata

    Args:
        fn (str or Path): filename or Path object to image file

    Returns:
        image (numpy array), (xres, yres), dictionary of imageJ metadata

    """
    data = tifffile.imread(fn)
    with tifffile.TiffFile(fn) as tif:
        page = tif.pages[0]
        xres = (
            page.tags["XResolution"].value[1]
            / page.tags["XResolution"].value[0]
        )
        yres = (
            page.tags["YResolution"].value[1]
            / page.tags["YResolution"].value[0]
        )
        imagej_metadata = tif.imagej_metadata
    return data, (xres, yres), imagej_metadata


def writetiff_with_metadata(fn, arr, xyres, metadata):
    """write array as an imageJ tif file with metadata

    Args:
        fn (str or Path): filename or Path for saved image
        arr (numpy array): image data to be saved
        xyres (tuple of floats): pixel size in x and y (x,y). Must be in units of 'micron'
        metadata (dict): dictionary containing imageJ metadata. 'min' and 'max' key will be added
        for image display in imageJ.

    Returns:
        None

    """
    # float32 type is not JSON serializable, convert to native float(64)
    metadata["min"] = float(arr.min())
    metadata["max"] = float(arr.max())
    metadata["axes"] = "TYX"
    tifffile.imwrite(
        fn,
        arr,
        imagej=True,
        resolution=(1 / xyres[0], 1 / xyres[1], 5),
        metadata=metadata,
    )
