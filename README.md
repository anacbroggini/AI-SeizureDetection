# ai-seizure-detectives

## Environment

In some cases it is necessary to install the Rust compiler for the transformers library.

```BASH
brew install rustup
rustup-init
```
Than press ```1``` for the standard installation.

Then we can go on to install hdf5:

```BASH
 brew install hdf5
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.14.1

pip install -U pip
pip install --no-binary=h5py h5py
pip install -r requirements.txt
```
If you are working on Windows type the following commands in the PowerShell:

```sh
python -m venv .venv
.venv\Scripts\Activate.ps1
```

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file.*