ResNet for Butterfly Classification and Zero-Shot SegmentationThis repository contains the PyTorch implementation of a custom Residual Network (ResNet) for classifying 100 species of butterflies and moths. It also features a novel, training-free method for image segmentation that leverages the trained classifier's gradients.Key Achievements:Classification Accuracy: 96.2% on a 100-class dataset.Segmentation mIoU: 0.796 using a zero-shot approach.Table of ContentsProject OverviewKey FeaturesArchitecture1. ResNet-14 for Classification2. Zero-Shot Segmentation PipelineDatasetResultsSetup and InstallationUsageTraining the ClassifierEvaluation and SegmentationFile StructureProject OverviewThis project tackles two significant computer vision tasks:Image Classification: A custom 14-layer ResNet is built from the ground up and trained on a challenging fine-grained dataset of butterflies and moths. The training pipeline is heavily optimized with modern techniques to achieve high accuracy.Image Segmentation: After training, the classifier is repurposed to perform instance segmentation. By analyzing the model's gradients, we generate saliency maps that highlight the object of interest. These maps are then used to automatically seed the GrabCut algorithm, producing high-quality segmentation masks without any additional training.Key FeaturesCustom ResNet from Scratch: Implementation of a 6n+2 layer ResNet (n=2, 14 layers total) in PyTorch.Optimized Training Pipeline: Utilizes label smoothing, a Cosine Annealing learning rate scheduler, and mixed-precision training for faster convergence and better performance.Extensive Data Augmentation: Employs random flips, rotations, and color jittering to improve model generalization.Zero-Shot Segmentation: A novel pipeline that combines gradient-based saliency maps with the GrabCut algorithm to perform segmentation without a dedicated segmentation model or labels.Reproducibility: Code is seeded for reproducible results.Architecture1. ResNet-14 for ClassificationThe network architecture is based on the ResNet paper, constructed with n=2. The total number of layers is 6n+2 = 14.Input: 224x224x3 images.Layer 1: A 3x3 convolution layer.Stage 1 (n=2): Two residual blocks with 32 filters. Output feature map size: 224x224.Stage 2 (n=2): Two residual blocks with 64 filters. The first block uses a stride of 2 for down-sampling. Output feature map size: 112x112.Stage 3 (n=2): Two residual blocks with 128 filters. The first block uses a stride of 2 for down-sampling. Output feature map size: 56x56.Output: Global Average Pooling followed by a Fully Connected layer with 100 units for classification.2. Zero-Shot Segmentation PipelineThe segmentation process is training-free and leverages the internal knowledge of the trained classifier.Flowchart:Input Image -> Trained ResNet-14 -> Get Predicted Class Logit
                                     |
                                     V
        Compute Gradient of Logit w.r.t. Input Image Pixels
                                     |
                                     V
                 Generate Saliency Map (Gradient Magnitude)
                                     |
                                     V
        Automated Seeding via Quantile Thresholding (Find definite foreground/background)
                                     |
                                     V
                      Run GrabCut with Automated Seeds
                                     |
                                     V
                          Final Segmentation Mask
DatasetThis project uses the Butterfly & Moths Image Classification dataset.Classes: 100Training Images: ~12,000Validation Images: 500Test Images: 100The dataset also includes ground-truth segmentation masks for validation and testing, which were used to calculate the mIoU score.ResultsTaskMetricScoreImage ClassificationAccuracy96.2%Image SegmentationmIoU0.796Example Segmentation Output:Original ImageSaliency MapFinal MaskSetup and InstallationClone the repository:git clone [https://github.com/your-username/resnet-zeroshot-segmentation.git](https://github.com/your-username/resnet-zeroshot-segmentation.git)
cd resnet-zeroshot-segmentation
Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:A requirements.txt file should be created containing:torch
torchvision
numpy
opencv-python
Pillow
Then run:pip install -r requirements.txt
UsageTraining the ClassifierTo train the ResNet model from scratch, run the train.py script.python train.py <path_to_train_data_dir> <path_to_model_output_dir>
<path_to_train_data_dir>: The directory containing the training images, organized into subfolders by class.<path_to_model_output_dir>: The directory where the trained model checkpoint (resnet_model.pth) will be saved.Example:python train.py ./dataset/train ./checkpoints
Evaluation and SegmentationTo evaluate the trained model on a test set and generate segmentation masks, run the evaluate.py script.python evaluate.py <path_to_model_ckpt> <path_to_test_imgs_dir>
<path_to_model_ckpt>: Path to the saved model checkpoint (e.g., ./checkpoints/resnet_model.pth).<path_to_test_imgs_dir>: Directory containing the test images.Example:python evaluate.py ./checkpoints/resnet_model.pth ./dataset/test
This will produce two outputs:submission.csv: A file containing the predicted class for each test image.seg_maps/: A new directory containing the generated segmentation mask for each test image.File Structure.
├── checkpoints/          # Directory to save model weights
├── dataset/
│   ├── train/
│   └── test/
├── seg_maps/             # Output directory for segmentation masks
├── train.py              # Script to train the ResNet classifier
├── evaluate.py           # Script to run inference and segmentation
├── requirements.txt      # Project dependencies
└── README.md             # This file
