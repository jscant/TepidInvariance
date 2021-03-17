# Deep SE(3)-Invariant Protein Hotspot Predictions

This is the beginnings of a project for infering protein hotspots from real
data. The models are based on
[LieConv](https://arxiv.org/abs/2002.12880) and
[LieTransformer](https://github.com/mfinzi/LieConv) SE(3)-equivariant neural
networks.

To install (including dependencies):

```
conda install -c conda-forge rdkit -y
conda install -c openbabel openbabel -y
git clone https://github.com/jscant/TepidInvariance.git
cd TepidInvariance
pip install -r requirements.txt
```

An installation of [PyMOL](https://pymol.org) is recommended for visualisation
of results.

In general, receptors should be in .pdb format, with ligands in .sdf or .mol2
format. The directory setup for these files should laid out exactly as follows
(receptor files must be under `receptors/<receptor_name>/receptor.pdb`, ligand
files must all be in the same sdf or mol2 file under
`ligands/<receptor_name>/ligands.[sdf|mol2])`:

```
<dataset_output_base_path>
├── ligands
│   ├── receptor_a
│   │   └── ligands.sdf
│   └── receptor_b
│       └── ligands.sdf
└── receptors
    ├── receptor_a
    │   └── receptor.pdb
    └── receptor_a
        └── receptor.pdb
```

## Preparation of input data

The inputs to the model are in the more memory-efficient pandas parquet format.
The script `preprocessing/pdb_to_parquet.py` handles converting pdb, sdf
and mol2 files into parquet files, as well as converting the atom type to
a smina-style `types` number (1-12), depending on the atomic number and
chemical environment. The distance of the receptor atoms to the nearest ligand
`acceptor`, `donor` and `aromatic` atoms are stored. The result
of this process is a file containing the cartesian coordinates, the sminatype,
and distance to each type of ligand ligand atom of every receptor atom. This is 
calculated for each ligand in each `ligands.sdf` file, by running:

```python3 preprocessing/pdb_to_parquet.py <dataset_base_path> <dataset_output_base_path>```

## Training the model

After running this, models can be trained:

```python3 tepid_invariance.py lietransformer <dataset_output_base_path> <save_path> -b 8 -lr 0.002 -e 1 --weight_decay 1e-5 --layers 6 --activation swish --attention_fn dot_product --max_suffix 0 --binary_threshold 6 -k 16```

The model will use regression on the distance to closest aromatic ligand atom
if `--binary_threshold` is not specified, otherwise it will train for binary
classification with receptor atoms closer than `<binary_threshold>` Angstroms to
an aromatic ligand atom labelled as 1 with others labelled 0. 

The script options are as follows:

```
positional arguments:
  model                 Type of point cloud network to use (lietransformer or
                        lieconv)
  filter                One of aromatic, hba, hbd or any.
  train_data_root       Location of structure training *.parquets files.
                        Receptors should be in a directory named receptors,
                        with ligands located in their specific receptor
                        subdirectory under the ligands directory.
  save_path             Directory in which experiment outputs are stored.

optional arguments:
  -h, --help            show this help message and exit
  --load_weights LOAD_WEIGHTS, -l LOAD_WEIGHTS
                        Load a model.
  --test_data_root TEST_DATA_ROOT, -t TEST_DATA_ROOT
                        Location of structure test *.parquets files. Receptors
                        should be in a directory named receptors, with ligands
                        located in their specific receptor subdirectory under
                        the ligands directory.
  --translated_actives TRANSLATED_ACTIVES
                        Directory in which translated actives are stored. If
                        unspecified, no translated actives will be used. The
                        use of translated actives are is discussed in
                        https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Number of examples to include in each batch for
                        training.
  --epochs EPOCHS, -e EPOCHS
                        Number of times to iterate through training set.
  --channels CHANNELS, -k CHANNELS
                        Channels for feature vectors
  --train_receptors [TRAIN_RECEPTORS [TRAIN_RECEPTORS ...]],
                        -r [TRAIN_RECEPTORS [TRAIN_RECEPTORS ...]]
                        Names of specific receptors for training. If
                        specified, other structures will be ignored.
  --test_receptors [TEST_RECEPTORS [TEST_RECEPTORS ...]],
                        -q [TEST_RECEPTORS [TEST_RECEPTORS ...]]
                        Names of specific receptors for testing. If specified,
                        other structures will be ignored.
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate for gradient descent
  --weight_decay WEIGHT_DECAY, -w WEIGHT_DECAY
                        Weight decay for regularisation
  --wandb_project WANDB_PROJECT
                        Name of wandb project. If left blank, wandb logging
                        will not be used.
  --wandb_run WANDB_RUN
                        Name of run for wandb logging.
  --layers LAYERS       Number of group-invariant layers
  --channels_in CHANNELS_IN, -chin CHANNELS_IN
                        Input channels
  --liftsamples LIFTSAMPLES
                        liftsamples parameter in LieConv
  --radius RADIUS       Maximum distance from a ligand atom for a receptor
                        atom to be included in input
  --nbhd NBHD           Number of monte carlo samples for integral
  --load_args LOAD_ARGS
                        Load yaml file with command line args. Any args
                        specified in the file will overwrite other args
                        specified on the command line.
  --double              Use 64-bit floating point precision
  --kernel_type KERNEL_TYPE
                        One of 2232, mlp, overrides attention_fn (see original
                        repo) (LieTransformer)
  --attention_fn ATTENTION_FN
                        One of norm_exp, softmax, dot_product: activation for
                        attention (overridden by kernel_type) (LieTransformer)
  --activation ACTIVATION
                        Activation function
  --kernel_dim KERNEL_DIM
                        Size of linear layers in attention kernel
                        (LieTransformer)
  --feature_embed_dim FEATURE_EMBED_DIM
                        Feature embedding dimension for attention; paper had
                        dv=848 for QM9 (LieTransformer)
  --mc_samples MC_SAMPLES
                        Monte carlo samples for attention (LieTransformer)
  --dropout DROPOUT     Chance for nodes to be inactivated on each trainin
                        batch (LieTransformer)
  --binary_threshold BINARY_THRESHOLD
                        Threshold for distance in binary classification. If
                        unspecified, the distance value will be used for
                        regression instead.
  --max_suffix MAX_SUFFIX
                        Maximum integer at end of filename: for example,
                        CHEMBL123456_4.parquet would be included with
                        <--max_suffix 4> but not <--max_suffix 3>.
  --inverse             Regression is performed on the inverse distance.
```

## Visualising predictions

#### Proteins
`colour_structure.py` can be used to visualise the hotspots generated by the
model like so:

`python3 colour_structure.py <saved_model.pt> <receptor.pdb> <ligand.sdf>
<output_file.pdb>`

where `<saved_model.pt>` is a result of model.save() (automatically saved to 
`<save_path/checkpoints>` at the end of each training epoch), `<receptor.pdb>`
is a pdb file and `ligand.sdf` is a ligand file. The output is saved to
`<output_file.pdb>`, and can be visualised in PyMOL using `PyMOL 
<output_file.pdb>`. In PyMOL, the predictions can be shown by colour using:

```
spectrum b, white_red
```

#### Ligands

`label_ligands.py` is used to colour ligand atoms by aromaticity.

```
usage: label_ligands.py [-h] ligands output_path

positional arguments:
  ligands      SDF file containing ligand coordinates (possibly multiple
               molecules)
  output_path  Directory in which to save output

optional arguments:
  -h, --help   show this help message and exit
```

The input sdf file can contain one or more molecules, and the output path should
be a directory where the resulting pdb files are to be saved. The pdb files can
be loaded into PyMOL, and the aromatic atoms can be labelled using the PyMOL
command:

```
color pink, b > 0
```
