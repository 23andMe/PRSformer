# PRSformer: Disease Prediction from Million-Scale Individual Genotypes
[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue)](#-citation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

**Authors:** Payam Dibaeinia, Chris German, Suyash Shringarpure, Adam Auton, Aly A. Khan


## Overview

PRSformer is a scalable deep learning architecture designed for end-to-end, multitask disease prediction directly from million-scale individual genotypes. It uses neighborhood attention, achieving linear O(L) complexity per layer, making Transformers tractable for genome-scale inputs. This design allows PRSformer to learn complex, potentially non-linear, and long-range interactions directly from raw genotypes.

## Model Architecture

The PRSformer architecture is built using PyTorch and consists of the following key components:

-   **`g2p_transformer_ExplicitNaNDose2`**: The main model class. It handles the learnable embeddings for genetic variants (SNPs) and their dosages. It distinguishes between observed genotypes and missing data, which is crucial for real-world genomic datasets.
-   **`TransformerLayer`**: At the core of PRSformer are stacked Transformer layers. These layers use Neighborhood Attention (`MultiHeadAttention_natten`) to efficiently model local dependencies within the genome, which is a key to the model's scalability.
-   **`PhenoHead_FC_on_flatten`**: After processing the genomic sequence through the Transformer layers, a phenotype head is used to predict disease risk. This module flattens the output of the transformer and passes it through a fully connected layer to generate predictions for multiple diseases in a multi-task setting.

The code is organized into two main files:
-   `src/model.py`: Contains the main model architectures, including `g2p_transformer_ExplicitNaNDose2`.
-   `src/modules.py`: Contains the building blocks of the models, such as `TransformerLayer`, `MultiHeadAttention_natten`, and various `PhenoHead` implementations.



## Dependencies
- Python ≥ 3.9
- PyTorch ≥ 2.0
- numpy ≥ 1.23, tqdm ≥ 4.66  
- natten — enables fast Neighborhood Attention. If unavailable, set `kernel_size=None` to use PyTorch attention.


## Citation

If you use PRSformer, please cite the paper:

```bibtex
@inproceedings{Dibaeinia2025PRSformer,
  title     = {PRSformer: Disease Prediction from Million-Scale Individual Genotypes},
  author    = {Payam Dibaeinia and Chris German and Suyash Shringarpure and Adam Auton and Aly A. Khan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```


## Acknowledgements
The authors thank the past and present employees and research participants of 23andMe for making this work possible. We are grateful to Akele Reed, Teague Sterling, David Hinds, Steve Pitts, Wei Wang, Bertram Koelsch, Michael Holmes, Stella Aslibekyan, Cordell Blakkan and Barry Hicks for their valuable contributions and insightful comments on the manuscript, and to Ali Hassani for helpful discussions on employing Neighborhood Attention. The authors also gratefully acknowledge the support of AWS for providing GPU computing resources and credits. AAK is supported in part by a Chan Zuckerberg Investigator Award.

## License
Released under a **23andMe Research License**. See [`LICENSE.txt`](./LICENSE.txt) for full terms.


