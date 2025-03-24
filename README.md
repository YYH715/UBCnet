# UBCnet：View-Guided Point Cloud Completion Network with Unified Multimodal Encoding and Bilateral Complementary Fusion
This is the official source code for the work, UBCnet: View-Guided Point Cloud Completion Network with Unified Multimodal Encoding and Bilateral Complementary Fusion
## Background
In IoT scenarios, due to sensor viewing Angle limitation, occlusion, noise and other reasons, point cloud data is generally missing and sparse and uneven in a wide range, which directly leads to a decline in the accuracy of pose estimation, robot grasping and subsequent tasks.

![image](https://github.com/user-attachments/assets/ee21d4ce-0757-4051-a3ab-9a70e937308c)



## Introduction
To address this, we propose a cross-modal completion framework for integrating view images and incomplete point clouds. The framework combines a unified multi-modal encoding strategy with a complementary bilateral fusion mechanism, effectively leveraging image appearance textures and point cloud geometric priors. First, during the encoding phase, the network maps 2D and 3D features into a compatible latent representation space, ensuring alignment between modalities during subsequent multi-channel fusion. Second, a bilateral attention pathway is introduced to enhance "image-guided point cloud" and "point cloud-refined image" representations bidirectionally. This is further supported by a decoding scheme that integrates skeleton-level and block-based refinement to balance the reconstruction of global structure and local details. Experimental results on public datasets ShapeNet-ViPC and real-world IoT scenarios demonstrate the superiority of the proposed method, significantly outperforming existing single-modal and multi-modal approaches, particularly in completing slender or concave–convex objects, offering a viable high-precision solution for 3D perception and operation in IoT applications.

![image](https://github.com/user-attachments/assets/69560454-6834-496a-a7a3-ab31f1061d08)

## Installation
- Install CUDA 11.8

- Set up python environment from requirement.txt:
```bash
pip install -r requirement.txt
```
- Install Chamer Distance:
```bash
cd ../../metrics/CD/chamfer3D/
python setup.py install
```
- Install Eeath Movers' Distance:
```bash
cd ../metrics/EMD/
python setup.py install
```
- Install PointNet++ utils:
```bash
cd /models/pointnet2_batch/
python setup.py install
```
- Install Furthest Point Sampling:
```bash
cd ../../tool/furthestPointSampling/
python setup.py install
```
## Results
### Results on ViPC-ShapeNet

![image](https://github.com/user-attachments/assets/bf3a3b73-065a-48d9-8d5a-4839767ee729)

### Results on our IoT datasets

![image](https://github.com/user-attachments/assets/46e051c7-776a-4a7d-88a4-eddd0397c742)

![image](https://github.com/user-attachments/assets/53b59c1d-71b8-4edf-8bbc-44935e7157e0)
