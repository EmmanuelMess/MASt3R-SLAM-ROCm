from pathlib import Path
from setuptools import setup

croco = Path(__file__).parent / "croco"

setup(
    install_requires=[
        "torch",
        "torchvision",
        "scikit-learn",
        "roma",
        "gradio",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "scipy",
        "einops",
        "trimesh",
        "tensorboard",
        "pyglet",
        "huggingface-hub[torch]>=0.22",
        f"croco @ {croco.as_uri()}",
    ],
)
