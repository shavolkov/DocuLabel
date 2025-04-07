# DocuLabel
 Various document differentiation 
## Dataset Preparation

The project begins with sorting and processing the data:

1. **Dataset Filtering**:
   - From the original dataset of **400,000 images**, a subset of **100,000 images** is selected for training testing, and evaluation.
   - The script `generate_sample.py` is responsible for filtering and selecting this subset.

2. **Randomized Dataset Splitting**:
   - The filtered images are randomly assigned to one of several folders for training, validation, and testing purposes.

---

## Model Architecture and Training

The classification model in **DocuLabel** is implemented in `model.py` and utilizes **ResNet-50** as the backbone. Here are it's key features:

1. **Model Architecture**:
- Built on **ResNet-50**, pretrained on ImageNet, for feature extraction.
2. **Data Augmentation**:
   - Implements various augmentations to improve model generalization, such as random rotations and flipping.
3. **Optimizer**:
   - Uses the **Adam optimizer** 
4. **Model State and Label Saving**:
   - At the end of each training it saves:
        - Filtered data
        - Model State
5. **Model Loading and Saving**:
   - Supports reloading saved models
---


## How to Run
1. **Dataset Preparation**:
   - Use `generate_sample.py` to filter and organize the dataset in a folder called reduced_data
   - Modify it accordingly to split between test, train, and val
   - Enure three directories are made called 'test', 'train', and 'val' 

2. **Training the Model**:
   - Train the ResNet-50-based model using `model.py`.
   - Create a directory called models to store past models
   - Create a directory called graphical_data to store benchmarking
   - Run 'model.py' for appropriate amount of epochs changing augmentations if necessesary 

3. **Visualizing Results**:
   - Generate Grad-CAM heatmaps using `visualize.py`.
   - Run `visualize.py` on a saved model. Ensure model is in folder called visualize_model
   - The following link is to the final model for the sake of testing: [Final model](https://drive.google.com/drive/folders/170_fdK1Nz79sQNVQ1f4YvUUVzTfK0d5i?usp=sharing)
