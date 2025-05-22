## Fast Reconstruction of Monocular Human Video Based on KAN
This repository is the official codebase for the paper Fast Reconstruction of Monocular Human Video Based on KAN, published in IEEE Sensors Journal. 

Creating 3D digital people from monocular video provides many possibilities for a wide range of users and rich applications. In this paper, we propose a fast, high-quality, and effective method for creating 3D digital humans from monocular videos, achieving fast training (2.5 minutes) and real-time rendering. Specifically, we use 3D Gaussian Splatting (3DGS), based on the introduction of Skinned Multi-Person Linear Model (SMPL) human structure prior, and an optimized KAN (Kolmogorov-Arnold-Net) neural network to build effective posture and linear blend skinning (LBS) weight estimation module to quickly and accurately learn the fine details of the 3D human body. In addition, to achieve fast optimization in the densification and prune stages, we propose a two-stage optimization method. First, the local 3D area that needs to be densified is extracted based on LightGlue, and then KL divergence combined with human body prior is further used to guide Gaussian splitting/cloning and merging operations. We conducted extensive experiments on the ZJU\_MoCap dataset, and the Peak Signal-to-Noise Ratio (PSNR) and Learned Perceptual Image Patch Similarity (LPIPS) metrics indicate that we effectively improved rendering quality while ensuring rendering speed.

### 📚 Paper Overview​

    Publication: IEEE Sensors Journal​

    Title: Fast Reconstruction of Monocular Human Video Based on KAN​

Our code is currently being organized and will be made public soon.
