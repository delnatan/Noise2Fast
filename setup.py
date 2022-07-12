from setuptools import setup

setup(
    name="Noise2Fast",
    version="0.1.0",
    author="Daniel Elnatan",
    author_email="delnatan@ucdavis.edu",
    description="an implementation of a single-shot image denoiser (Noise2Fast)",
    install_requires=[
        "numpy",
        "torch",
    ],
)

