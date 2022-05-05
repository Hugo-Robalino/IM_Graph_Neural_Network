# Implementation of the Center Loss in a Graph Neural Network for Fake News Detection

Tools used:
- GNN model from UPFD framework: https://github.com/safe-graph/GNN-FakeNews
- Center Loss: https://github.com/KaiyangZhou/pytorch-center-loss
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/

This projects investigates if an improvement in performance in the GNN model can be achieved by implementing the center loss, which provides the model with not only separability but also discriminative power.

It was found that, although the center loss does give the model discriminative power, there is no improvement in performance.

## Setting up data and virtual environment
The dataset is integrated in [PyTorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.UPFD). Instructions for how to acces it can be found in the original UPFD framework [GitHub page](https://github.com/safe-graph/GNN-FakeNews).

The [utils](https://github.com/safe-graph/GNN-FakeNews/tree/main/utils) directoy as well as the [center_loss](https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py) file were uploaded again to ensure that the main.py file works properly.

All the environment requirements can be found in the .yml file.
