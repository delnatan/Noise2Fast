import argparse
import torch
from tqdm import tqdm
from tifffile import imread, imwrite
from .core import denoise_stack, readtiff_with_metadata, writetiff_with_metadata
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(
        description="Noise2Fast",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_args = parser.add_argument_group("Input arguments")
    input_args.add_argument(
        "input_folder", help="Folder containing noisy TIF images."
    )
    input_args.add_argument(
        "device",
        default="cuda",
        help="hardware for doing denoising: cuda, mps, or cpu",
    )
    input_args.add_argument(
        "--n_postvalidation_frames",
        default=100,
        help="number of frames that will be average post-validation",
    )

    output_args = parser.add_argument_group("Output arguments")
    output_args.add_argument(
        "--output_folder",
        default="denoised",
        help="Name of folder where denoised images will be saved inside input_folder",
    )

    output_args.add_argument(
        "--verbose", action="store_true",
    )

    args = parser.parse_args()

    inputfolder = Path(args.input_folder)
    outputfolder = inputfolder / args.output_folder
    outputfolder.mkdir(exist_ok=True)

    device = torch.device(args.device)

    flist = [f for f in inputfolder.glob("*.tif")]

    # exclude denoised images
    flist = [f for f in flist if "denoised" not in f.name]

    print(f":: Denoised images will be saved in {str(outputfolder)} ::")

    for f in tqdm(flist):
        input_image, xyres, imagej_metadata = readtiff_with_metadata(f)
        denoised = denoise_stack(
            input_image, device,
            last_n_frames=args.n_postvalidation_frames,
            verbose=args.verbose
        )
        if imagej_metadata:
            writetiff_with_metadata(
                outputfolder / f"{f.stem}_denoised.tif",
                denoised,
                xyres,
                imagej_metadata,
            )
        else:
            imwrite(outputfolder / f"{f.stem}_denoised.tif",
                    denoised)


if __name__ == "__main__":

    main()
