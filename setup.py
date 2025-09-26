"""
Setup script for NoReward-RL: Enhanced Curiosity-Driven Exploration.

This package provides a modernized implementation of curiosity-driven exploration
for reinforcement learning with enhanced logging, environment support, and evaluation tools.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README_MODERN.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="noreward-rl",
    version="2.0.0",
    author="Enhanced Implementation",
    author_email="your-email@example.com",
    description="Enhanced curiosity-driven exploration for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/noreward-rl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
            "flake8>=5.0.0",
        ],
        "vizdoom": [
            "vizdoomgym>=1.2.0",
        ],
        "mario": [
            "gym-super-mario-bros>=7.4.0",
        ],
        "atari": [
            "gymnasium[atari]>=0.29.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "noreward-train=src.train_modern:main",
            "noreward-eval=scripts.eval_and_record:main",
            "noreward-benchmark=scripts.benchmark_hard_exploration:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.wad", "*.gif", "*.md"],
    },
    zip_safe=False,
)

