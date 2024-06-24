# DCGAN_Diffusion_Implementation

## GAN/DCGAN Implementation

## Table of Contents
- [GAN/DCGAN Implementation](#gandcgan-implementation)
  - [Dataset Preparation](#dataset-preparation)
  - [Discriminator-Generator Network](#discriminator-generator-network)
  - [Training](#training)
  - [FID Calculation](#fid-calculation)
- [Generative Diffusion](#generative-diffusion)
  - [Diffusion Model](#diffusion-model)
  - [Network Architecture](#network-architecture)
  - [Image Generation](#image-generation)
- [References and Notes]

### Dataset Preparation
- The dataset used is CelebA, which consists of 64x64 images.
- A custom dataset and dataloader are created to ingest this data into the model.

### Discriminator-Generator Network
- The Discriminator and Generator networks are based on the DCGAN architecture.
- The Discriminator consists of several convolutional layers to downsample the input image.
- The Generator uses transposed convolutional layers to upsample the noise input to a 64x64 image.

### Training
- The models are trained on the CelebA dataset to generate new images.

### FID Calculation
- The Fréchet Inception Distance (FID) is used to evaluate the quality of the generated images.
- Code is provided to calculate FID by comparing the statistics of real images with generated images.

## Generative Diffusion

### Dataset Preparation
- The same dataset CelebA is used here too.
- A custom dataset and dataloader are created to ingest this data into the model in a manner similar to the GAN model.

### Diffusion Model
- Diffusion models work by gradually adding noise to an image and then learning to reverse this process to generate new images.
- The Gaussian Diffusion process involves 1000 timesteps.

### Network Architecture
- The network used is a U-Net model with attention mechanisms to handle different image resolutions.
- The model is trained to denoise the noisy images generated at each step of the diffusion process.

### Image Generation
- Code is provided to generate new images using the trained diffusion model.
- The provided model weights are used to generate images to avoid computational expense.

## References and Notes:
- The Diffusion implementation has been run using pre-exisiting scripts which impments an existing pre-trained diffusion model details and the scripts in question are available in the [DLStudio Library](https://engineering.purdue.edu/kak/distDLS/#109), due to the fact that training the diffusion mode would have required a level of compute not availble with me.
- For more information on the implementation an outputs and pictorial representation of the outputs please refer to the attached PDF file.
- It is required for this implementation to have the DLStudio Library to be installed, which is available [here](https://engineering.purdue.edu/kak/distDLS/)


