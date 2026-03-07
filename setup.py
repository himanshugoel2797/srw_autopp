from setuptools import setup, find_packages

setup(
    name="srw-parameter-advisor",
    version="0.1.0",
    description="AI-driven propagation parameter optimization for SRW",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'srw-train=training.cli:main',
        ],
    },
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "training": [
            "torch>=2.0",
            "scikit-learn>=1.0",
            "tensorboard>=2.12",
        ],
    },
)
