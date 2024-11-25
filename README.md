# Extending CLIP for Category-to-Image Retrieval in E-commerce

This repository contains the implementation and resources used for the experiments in the paper "[Extending CLIP for Category-to-image Retrieval
in E-commerce](https://mariyahendriksen.github.io/files/ecir22.pdf)" published at **ECIR 2022**.

<div align="center">

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2112.11294)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

![CLIP-ITA model configuration](images/clip-ita.png)

## Overview
This project extends the **CLIP** model to improve category-to-image retrieval tasks in zero-shot vs. fine-tuned settings.

## Getting Started  

### Prerequisites  

- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- A GPU is recommended for training and evaluation.  

### Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/<your-repo>/clip-category-retrieval.git  
   cd clip-category-retrieval
   ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## License
This repository is licensed under the MIT License. Feel free to use, modify, and distribute the code. If you make significant modifications, please link back to this repository as a courtesy.


## Citing and Authors
If you find this repository helpful, please consider citing our paper:
```latex
@inproceedings{hendriksen-2022-extending-clip,
author = {Hendriksen, Mariya and Bleeker, Maurits and Vakulenko, Svitlana and van Noord, Nanne and Kuiper, Ernst and de Rijke, Maarten},
booktitle = {ECIR 2022: 44th European Conference on Information Retrieval},
month = {April},
publisher = {Springer},
title = {Extending CLIP for Category-to-image Retrieval in E-commerce},
year = {2022}}
```

