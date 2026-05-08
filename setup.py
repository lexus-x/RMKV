from setuptools import setup, find_packages

setup(
    name="kanflow-vla",
    version="0.1.0",
    description="KANFlow-VLA: RWKV-GroupKAN Flow-Matching VLA for Few-Shot Robotic Manipulation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "transformers>=4.36.0",
        "timm>=0.9.0",
        "h5py>=3.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "train": ["wandb>=0.16.0"],
        "eval": ["metaworld"],
    },
)
