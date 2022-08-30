# DeepDFT Model Implementation

This is the official Implementation of the DeepDFT model for charge density prediction.

## Setup

Create and activate a virtual environment and install the requirements:

	$ pip install -r requirements.txt

## Data

Training data is expected to be a tar file containing `.cube` (Gaussian) or `.CHGCAR` (VASP) density files.
For best performance the tar files should not be compressed, but the individual files inside the tar
can use `zlib` compression (add `.zz` extension) or lz4 compression (add `.lz4` extension).
The data can be split up in several tar files. In that case create a text (.txt) file
in the same directory as the tar files. The text file must contain the file names of the tar files, one on each line.
Then the text file can then be used as a dataset.

## Training the model

Inspect `runner.py` arguments:

	$ python runner.py --help

Example used for training the model on QM9:

	$ python runner.py --dataset datadir/qm9vasp.txt --split_file datadir/splits.json --ignore_pbc --cutoff 4 --num_interactions 6 --max_steps 100000000 --node_size 128

Or to train the equivariant model on the ethylene carbonate dataset:

	$ python runner.py --dataset datadir/ethylenecarbonate.txt --split_file datadir/splits.json --cutoff 4 --num_interactions 3 --use_painn_model --max_steps 100000000 --node_size 128

The json file contains two keys "train", and "validation" each with a list of indices for the train and validation sets. If the argument is omitted the data will be randomly split.

## Running the model on new data

To use a trained model to predict the electron density around a new structure use the script `predict_with_model.py`.
The first argument is the output directory of the runner script in which the trained model is saved.
The second argument is an ASE compatible xyz file with atom coordinates for the structure to be predicted.

For example:

	$ python predict_with_model.py pretrained_models/qm9_schnet example_molecule.xyz

For more options see the `predict_with_model.py` optional arguments:

	$ python predict_with_model.py --help

