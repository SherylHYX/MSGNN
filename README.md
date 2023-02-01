# MSGNN
MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian (LoG 2022)

For details, please read [our paper](https://arxiv.org/pdf/2209.00546.pdf). You are also welcome to read our [poster](https://github.com/SherylHYX/MSGNN/blob/main/LoG2022_MSGNN_poster.pdf).

MSGNN is also implemented in the [PyTorch Geometric Signed Directed](https://github.com/SherylHYX/pytorch_geometric_signed_directed) library.

**Citing**


If you find MSGNN useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{he2022msgnn,
  title={MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian},
  author={He, Yixuan and Perlmutter, Michael and Reinert, Gesine and Cucuringu, Mihai},
  booktitle={Learning on Graphs Conference},
  pages={40--1},
  year={2022},
  organization={PMLR}
}
```

--------------------------------------------------------------------------------

## Environment Setup
### Overview
<!-- The underlying project environment composes of following componenets: -->
The project has been tested on the following environment specification:
1. Ubuntu 18.04.6 LTS
2. Nvidia Graphic Card (NVIDIA Tesla T4 with driver version 450.142.00) and CPU (Intel Xeon Platinum 8259CL CPUs @ 2.50GHz)
3. Python 3.7
4. CUDA 11.0
5. Pytorch 1.10.1 (built against CUDA 10.2)
6. Other libraries and python packages (See below)

The codebase is implemented in Python 3.7. package versions used for development are below.
```
networkx                        2.6.3
numpy                           1.20.3
scipy                           1.5.4
argparse                        1.1.0
sklearn                         0.23.2
torch                           1.10.1
torch-geometric                 2.0.3 (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
torch-geometric-signed-directed 0.22.0
tensorboard                     2.4.0
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use [parallel](https://www.gnu.org/software/parallel/), which can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores processed data sets.

- ./src/ stores files to train various models, utils and metrics.

- ./*_results/ stores results for different data sets.

## Options
<p align="justify">
MSGNN provides the following command line arguments, which can be viewed in the ./src/parser_node.py for the node clustering tasks and ./src/parser_link.py for the link prediction tasks.
</p>

## Reproduce results
First, get into the ./execution/ folder:
```
cd execution
```
To reproduce all results for MSGNN on GPU 0 (group a):
```
bash a0.sh
```
Other execution files are similar to run.

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files.

You can also use CPU for training if you add ``--cpu".

## Direct execution with training files

First, get into the ./src/ folder:
```
cd src
```
Then, below are various options to try:

Creating an MSGNN model for SDSBM on the node clustering task of the default setting on the 3c data set (remeber to specify a data set name).
```
python ./main_SDSBM_node.py --dataset 3c --weighted_input_feat --sd_input_feat
```
Creating a model for the FiLL-pvCLCL data set in the year 2010 on the link prediction task 4C with some custom learning rate and epoch number, for the SigMaNet method.
```
python ./main_signed_directed_link.py --dataset pvCLCL --year 2010 --lr 0.001 --epochs 300 --num_classes 4 --method SigMaNet --weighted_input_feat --sd_input_feat
```
Creating a model for the FiLL-OPCL data set in the year 2016 on the link prediction task 3C and use CPU, for the SDGNN method.
```
python ./main_signed_directed_link.py --dataset OPCL --year 2016 --cpu --num_classes 5 --direction_only_task --method SDGNN
```

## Note
- The convention of the definition of the K parameter needs a bit of attention: in the paper, K denotes the degree of the Chebyshev polynomial, while in the code implmentation, K denotes the size of the Chebyshev filter (followed from the [Cheb_conv definition from PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv)). Therefore, though we have by defualt K=1 as stated in our paper, this default means K=2 in our code.

--------------------------------------------------------------------------------
