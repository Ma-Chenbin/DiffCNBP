# DiffCNBP
#### *by: *


## Supplementary Material and code for **DiffCNBP**

## Datasets
### Available Datasets
- [MIMIC-III Waveform](https://archive.physionet.org/physiobank/database/mimic3wdb/)
- [Mindray Dataset]()

## Implementation Code
### Requirmenets:
- Python3
- Pytorch==1.0.1
- Numpy==1.16.2
- scikit-learn==0.20.3
- Pandas==0.23.4
- Matplotlib==3.0.2
- skorch==0.9.0 (For DEV risk calculations)
- openpyxl==2.5.8 (for classification reports)
- Wandb=0.8.28 (for sweeps)

### Adding New Dataset
#### Structure of data

#### Configurations


### Existing Algorithms
- [DiffCNBP]()
- [Pulse2Pulse](https://pubmed.ncbi.nlm.nih.gov/34753975/)
- [WaveGAN*](https://pubmed.ncbi.nlm.nih.gov/34753975/)
- [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [WGAN](https://github.com/Yangyangii/GAN-Tutorial)
- [WGAN-GP](https://github.com/mcclow12/wgan-gp-pytorch)
- [Pix2Pix](https://arxiv.org/abs/1611.07004)
- [V-Net](https://www.nature.com/articles/s41598-021-94913-y)
- [U-Net](https://github.com/inventec-ai-center/bp-benchmark/tree/main)
- [NARX](https://ieeexplore.ieee.org/document/9336230)

### Adding New Algorithm
To add a new algorithm, place it in `algorithms/algorithms.py` file.

## Training procedure


## Results

## Citation
If you found this work useful for you, please consider citing it.
```
@article{DiffCNBP,
  title   = {####},
  author  = {####},
  journal = {####},
  year    = {2023}
}
```

## Contact
For any issues/questions regarding the paper or reproducing the results, please contact any of the following.

Department of Biomedical Engineering, Beihang University, 
37 Xueyuan Road, Beijing, 100853

## Acknowledgement
We would like to thank the following repositories for their valuable contributions to this work:
- [Pulse2Pulse](https://github.com/vlbthambawita/Pulse2Pulse)
- [WaveGAN*](https://github.com/mazzzystar/WaveGAN-pytorch)
- [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [WGAN](https://github.com/Yangyangii/GAN-Tutorial)
- [WGAN-GP](https://github.com/mcclow12/wgan-gp-pytorch)