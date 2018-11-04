# Cross-domain disentantanglement network

Code for the paper ["Image-to-image translation for cross-domain disentanglement"](https://arxiv.org/abs/1805.09730), NIPS 2018.

<img src="https://abelgonzgarc.com/wp-content/uploads/2018/11/model.png" alt="Cross-domain disentanglement network" width="90%">

Based on [this pix2pix implementation](https://github.com/affinelayer/pix2pix-tensorflow/) by Christopher Hesse, extensively explained in [this article](https://affinelayer.com/pix2pix/). 

## Setup
Please follow the setup described [here](https://github.com/affinelayer/pix2pix-tensorflow/).
Tested with Tensorflow 1.8.0.

See DATA/MNISTCDCB/ for example images of our MNIST-CD/CB dataset.

## Usage

In order to train a MODEL using DATA, run
```sh
python run_cross_domain_disen.py \
  --mode train \ 
  --output_dir checkpoints/MODEL \ 
  --input_dir DATA/train/  
```

Once the model finished training, it can be tested by running
```sh
python run_cross_domain_disen.py \ 
  --mode test \ 
  --output_dir test/MODEL \
  --checkpoint checkpoints/MODEL \
  --input_dir DATA/test/  
```

In order to extract disentangled features for other tasks (e.g. cross-domain retrieval), run
```sh
python run_cross_domain_disen.py \ 
  --mode features \ 
  --output_dir features/MODEL \ 
  --checkpoint checkpoints/MODEL \ 
  --input_dir DATA/test/  
```

## Citation
Please, cite the following paper if you use this code:
```
@inproceedings{gonzalez-garcia2018NIPS,
  title={Image-to-image translation for cross-domain disentanglement},
  author={Gonzalez-Garcia, Abel and van de Weijer, Joost and Bengio, Yoshua},
  booktile={NIPS},
  year={2018}
}
```
