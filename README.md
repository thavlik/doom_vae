# EARLY WORK IN PROGRESS

Variational autoencoders (VAE) trained on DOOM 1/2 gameplay videos

## Motivation
Latent representations and unsupervised pretraining boost data efficiency on more challenging supervised [1] and reinforcement learning tasks [2]. The goal of this project is to provide both the Doom and machine learning communities with:
- High quality datasets comprised of Doom gameplay
- Various ready-to-run VAE experiments
- Suitable boilerplate for derivative projects

### Relevant Literature
1. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
2. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
3. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
4. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## TODO
- Progressive growing decoder a la [3]
- Implement beta loss term from [4]
- Implement FID loss
- Dataset compiler
- ~~Doom gameplay video links~~
- ~~Implement entrypoints~~
- ~~Implement datasets~~
- ~~Resnet boilerplate~~

