from setuptools import setup, find_packages

setup(
    name="hmp-ai",
    version="0.1",
    packages=find_packages(),
    install_require=[
        "alibi==0.9.4",
        "hsmm_mvpy==0.1.0",
        "innvestigate==2.1.2",
        "keras==2.13.1",
        "matplotlib==3.7.0",
        "mne==1.4.0",
        "numpy==1.23.3",
        "scikit_learn==1.1.2",
        "tensorflow==2.13.1",
        "tensorflow_intel==2.13.1",
        "xarray==2023.5.0",
        "xbatcher==0.3.0",
    ],
)
