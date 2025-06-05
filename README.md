# KairosAD: A SAM-Based Model for Industrial Anomaly Detection on Embedded Devices #

Official implementation of the paper [KairosAD: A SAM-Based Model for Industrial Anomaly Detection on Embedded Devices](https://intelligolabs.github.io/KairosAD/) accepted at the 23rd International Conference on Image Analysis and Processing (ICIAP 2025).


## Installation ##

**1. Repository setup:**
* `$ git clone --recurse-submodules https://github.com/intelligolabs/KairosAD`

Or, if you have already has already cloned the repo:
* `$ git submodule update --init --recursive`
* `$ cd KairosAD`
* On file `MobileSAM/mobile_sam/utils/transforms.py`, update the function apply_image() with the following code:
```python
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        image = image.permute(2, 0, 1)
        
        return resize(image, target_size)
```
* Download the MVTec AD dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
* Download the FACES dataset from https://github.com/amazon-science/spot-diff

**2. Conda environment setup:**
* `$ conda create -n kairos_ad python=3.10`
* `$ conda activate kairos_ad`
* `$ cd MobileSAM`
* `$ pip install -e .`
* `$ pip install -r requirements.txt`

Optionally, you can also log the training and evaluation to [wandb](https://wandb.ai).
* Update line 102 of the file `main.py`, specifying `project=''` and `entity=''`


## Authors ##

Uzair Khan, Franco Fummi, Luigi Capogrosso

*Department of Engineering for Innovation Medicine, University of Verona, Italy*

`name.surname@univr.it`


## Citation ##

If you use [**KairosAD**](https://arxiv.org/abs/2505.24334), please, cite the following paper:
```
@Article{khan2025kairosad,
  title   = {{KairosAD: A SAM-Based Model for Industrial Anomaly Detection on Embedded Devices}},
  author  = {Khan, Uzair and Fummi, Franco and Capogrosso, Luigi},
  journal = {arXiv preprint arXiv:2505.24334},
  year    = {2025}
}
```
