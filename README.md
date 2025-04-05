# UBCnet：View-Guided Point Cloud Completion Network with Unified Multimodal Encoding and Bilateral Complementary Fusion
This is the official source code for the work, UBCnet: View-Guided Point Cloud Completion Network with Unified Multimodal Encoding and Bilateral Complementary Fusion
## Background
In dynamic 3D scene reconstruction, 3D point clouds collected by sensors often suffer from extensive missing data and uneven sparsity due to factors such as occlusion, limited viewpoints, motion blur, and noise. This presents significant challenges in meeting the accuracy requirements of applications such as intelligent manufacturing and autonomous driving.

![image](https://github.com/user-attachments/assets/ee21d4ce-0757-4051-a3ab-9a70e937308c)



## Introduction
To address this, we propose a cross-modal completion framework for integrating view images and incomplete point clouds. The framework combines a unified multi-modal encoding strategy with a complementary bilateral fusion mechanism, effectively leveraging image appearance textures and point cloud geometric priors. First, during the encoding phase, the network maps 2D and 3D features into a compatible latent representation space, ensuring alignment between modalities during subsequent multi-channel fusion. Second, a bilateral attention pathway is introduced to enhance "image-guided point cloud" and "point cloud-refined image" representations bidirectionally. This is further supported by a decoding scheme that integrates skeleton-level and block-based refinement to balance the reconstruction of global structure and local details. Experimental results on public datasets ShapeNet-ViPC and real-world captured datasets demonstrate the superiority of the proposed method, significantly outperforming existing single-modal and multi-modal approaches, particularly in completing slender or concave–convex objects, offering a viable high-precision solution for 3D perception and operation in complex vision tasks.

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
### Dataset
- Download the [ShapeNet-ViPC dataset](https://pan.baidu.com/share/init?surl=NJKPiOsfRsDfYDU_5MH28A)(code: ar8l) and unzip it:
```bash
cat ShapeNetViPC-Dataset.tar.gz* | tar zx
```
- Then, you will get `ShapeNetViPC-Partial`, `ShapeNetViPC-GT` and `ShapeNetViPC-View`. Use the code in tool/dataloader.py to load the dataset.
  
### Running
```bash
# Training
python main.py

# Evaluation
# Specify the checkpoint path in Config.py
__C.CONST.WEIGHTS = "path to your checkpoint"

python main.py --test
```

## Results
### Results on ViPC-ShapeNet
- Quantitative results on ShapeNet-ViPC dataset:

![image](https://github.com/user-attachments/assets/a0d86466-480e-45b3-a302-740be2613547)
![image](https://github.com/user-attachments/assets/ccc8274a-c2b1-45fb-8b56-3e184ed480be)

- Qualitative results on ShapeNet-ViPC dataset:
  
![image](https://github.com/user-attachments/assets/bf3a3b73-065a-48d9-8d5a-4839767ee729)

### Results on our IoT datasets
- To further validate the applicability and robustness of the proposed method in real-world IoT scenarios, we constructed a small-scale dataset based on five representative IoT objects. During the point cloud generation process, we deliberately introduced partial occlusions or blind spots to simulate the common local occlusions and missing regions encountered in real-world IoT deployments to better reflect the complexity of real application scenarios. The dataset includes both regular objects with relatively simple appearances, such as square packaging boxes and cylindrical bottles, and irregular components with detailed structural features, such as circuit breakers and metal ventilation grilles.

![image](https://github.com/user-attachments/assets/46e051c7-776a-4a7d-88a4-eddd0397c742)

![image](https://github.com/user-attachments/assets/53b59c1d-71b8-4edf-8bbc-44935e7157e0)

## License
This project is licensed under the [MIT license](https://github.com/YYH715/UBCnet/blob/main/LICENSE)
