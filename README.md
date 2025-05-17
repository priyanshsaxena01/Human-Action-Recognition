# Human Action Recognition with R(2+1)D and Enhanced Dynamic Mixture of Experts (MoE)

This project implements a human action recognition system using a pre-trained R(2+1)D (ResNet 2+1D) backbone, enhanced with a custom Motion Dynamics Module and a sophisticated Mixture of Experts (MoE) layer featuring dynamic MLP-based gating and noisy top-k routing. The system is designed to be trained and evaluated on the HMDB51 dataset within a Kaggle Notebook environment.

## Overview

The primary goal is to accurately classify human actions from video clips. The script handles the entire pipeline:
1.  **Data Acquisition:** Downloads the HMDB51 dataset (videos and train/test splits) and extracts them.
2.  **Data Preprocessing:** Samples frames from videos, resizes them, and normalizes them.
3.  **Model Architecture:**
    *   Utilizes a pre-trained **R(2+1)D-18** network as a feature extractor.
    *   Introduces a **Motion Dynamics Module** to learn temporal patterns from adjacent feature maps.
    *   Employs an **Enhanced Dynamic Top-K Mixture of Experts (MoE)** layer for classification:
        *   **MLP-based Gating Network:** For more expressive routing decisions.
        *   **Noisy Top-K Gating:** Adds noise to gate logits during training for improved robustness and exploration.
        *   **Residual MLP Experts:** Experts are MLPs with residual connections for increased capacity.
        *   **Load Balancing Loss:** Encourages balanced utilization of experts.
4.  **Training:** Trains the model on a specified split of HMDB51. If a pre-trained checkpoint for the current split exists, it loads it instead.
5.  **Evaluation:** Evaluates the trained/loaded model on the test set, providing metrics like accuracy, classification report, and a confusion matrix.
6.  **Inference:**
    *   Demonstrates inference on sample videos from the test set with thumbnail visualization.
    *   Provides an interactive widget for users to upload their own videos and get predictions.

## Key Features & Enhancements

*   **R(2+1)D Backbone:** Leverages a strong pre-trained 3D CNN for spatio-temporal feature extraction.
*   **Motion Dynamics Module:** A custom module designed to explicitly capture and learn from frame-to-frame changes in the feature space, enhancing the model's understanding of motion.
*   **Dynamic Top-K Mixture of Experts (MoE):**
    *   **MLP Gate:** Uses a multi-layer perceptron for the gating network, allowing for more complex routing decisions compared to simpler linear gates.
    *   **Noisy Top-K Routing:** During training, Gaussian noise is added to the gate logits before selecting the top-k experts. This encourages exploration and can lead to more robust models.
    *   **Residual Experts:** Individual expert networks are MLPs with residual connections, allowing them to be deeper and potentially more powerful.
    *   **Load Balancing:** Includes an auxiliary loss term to encourage the gate to distribute samples more evenly across experts.
*   **End-to-End Pipeline:** Handles data downloading, extraction, preprocessing, training, evaluation, and inference.
*   **Kaggle Optimized:** Designed for Kaggle Notebooks, including handling of Kaggle paths and GPU utilization.
*   **Interactive Inference:** Allows users to upload their videos for on-the-fly action recognition.
*   **Robust Data Handling:** Includes checks for file existence, download integrity, and extraction success, along with error handling for video processing.

## Technologies Used

*   **Programming Language:** Python
*   **Core Libraries:**
    *   PyTorch: For deep learning model definition, training, and inference.
    *   TorchVision: For pre-trained models (R(2+1)D) and video utilities.
    *   OpenCV (cv2): For video frame processing.
    *   NumPy: For numerical operations.
    *   Scikit-learn: For evaluation metrics (accuracy, classification report, confusion matrix).
    *   Matplotlib & Seaborn: For plotting (confusion matrix, thumbnails).
    *   Pandas: For data manipulation (e.g., confusion matrix dataframe).
    *   `rarfile` & `unrar`: For extracting RAR archives.
    *   `requests`: For downloading files.
    *   `tqdm`: For progress bars.
    *   `ipywidgets`: For the interactive file upload widget.
*   **Dataset:** HMDB51
*   **Environment:** Kaggle Notebooks (GPU recommended: T4, P100, or similar).

## Prerequisites

1.  **Kaggle Account:** To run the notebook in the Kaggle environment.
2.  **GPU Enabled:** The notebook heavily relies on a GPU for efficient training and inference. Ensure a GPU accelerator is selected in your Kaggle Notebook settings.
3.  **Internet Connection (for first run):** Required for downloading libraries, the HMDB51 dataset, and split files. This should be enabled in Kaggle Notebook settings.

## File Structure (within `/kaggle/working/`)

*   `data/`
    *   `hmdb51_org.rar`: Downloaded HMDB51 video archive.
    *   `test_train_splits.rar`: Downloaded HMDB51 split files archive.
    *   `hmdb51_extracted/`: Directory containing extracted video files, organized by action class.
        *   `brush_hair/`
        *   `cartwheel/`
        *   ... (51 action classes)
    *   `splits_extracted/`: Directory containing extracted train/test split definition files.
        *   `brush_hair_test_split1.txt`
        *   ...
*   `models/`
    *   `hmdb51_dynamic_moe_split<N>_best.pth`: Saved model checkpoint for the best performing model on split `N`.
*   `user_uploads/`: Temporary directory for videos uploaded by the user via the widget.

## How to Run

The script is structured as a series of cells, intended to be run sequentially in a Kaggle Notebook.

1.  **Cell 1: Installations**
    *   Installs `rarfile` and the `unrar` system utility.

2.  **Cell 2: Imports**
    *   Imports all necessary Python libraries.

3.  **Cell 3: Configuration**
    *   Sets up device (GPU/CPU), paths, dataset parameters (number of classes, frames, image size), MoE hyperparameters (number of experts, top-k, hidden dimensions, noisy gating std), motion module parameters, and training hyperparameters (split number, batch size, learning rate, epochs, load balancing alpha, weight decay).
    *   Defines URLs for dataset download and the model checkpoint filename.

4.  **Cell 4 & 5: Download and Extraction**
    *   Defines `download_file` and `extract_rar` functions.
    *   Executes the download of `hmdb51_org.rar` and `test_train_splits.rar`.
    *   Executes the extraction of these RAR files into `DATASET_PATH` and `SPLIT_PATH` respectively. Handles nested RARs within HMDB51.
    *   Includes checks and sample listings of extracted content.
    *   Sets `data_ready_for_loading` flag.

5.  **Cell 6: Dataset Definitions**
    *   `load_split()`: Loads video filenames and labels for the specified train/test split.
    *   `sample_frames()`: Samples a fixed number of frames uniformly from a video, resizes, and converts them.
    *   `HMDB51Dataset(Dataset)`: PyTorch Dataset class to load and preprocess video data. Handles normalization specific to R(2+1)D.
    *   `collate_fn()`: Custom collate function to handle potential `None` values from failed video loading.

6.  **Cell 7: Model Architecture Definitions**
    *   `MotionDynamicsModule(nn.Module)`: Learns motion patterns from differences in adjacent feature maps.
    *   `ResidualExpertMLP(nn.Module)`: Defines an MLP expert with a residual connection.
    *   `DynamicTopKMoE(nn.Module)`: The MoE layer with an MLP gate, noisy top-k selection, and load balancing loss calculation.
    *   `ActionMoEDynamicModel(nn.Module)`: The complete model integrating the R(2+1)D backbone, Motion Dynamics Module, pooling, and the DynamicTopKMoE layer.

7.  **Cell 8: Training and Validation Loop Functions**
    *   `train_epoch()`: Logic for training the model for one epoch.
    *   `validate_epoch()`: Logic for evaluating the model on the validation set for one epoch.

8.  **Cell 9: Load Data Splits & Create Test Loader**
    *   Loads the train/test video lists and action map for the `CURRENT_SPLIT`.
    *   Creates the PyTorch `DataLoader` for the test set.
    *   Sets `data_loading_ok` flag.

9.  **Cell 10 (Optional): Clear Saved Model Checkpoint**
    *   Provides commented-out code to remove the `models` directory, forcing retraining if uncommented and run.

10. **Cell 11: Load or Train Model**
    *   Checks if a pre-trained model checkpoint (`BEST_MODEL_PATH`) exists for the `CURRENT_SPLIT`.
    *   **If exists:** Loads the model.
    *   **If not exists (or loading fails):**
        *   Creates training `DataLoader`.
        *   Initializes the `ActionMoEDynamicModel`, optimizer (Adam), and loss function (CrossEntropyLoss).
        *   Includes a learning rate scheduler (`ReduceLROnPlateau`).
        *   Runs the training loop for `NUM_EPOCHS`, validating after each epoch.
        *   Saves the model with the best validation accuracy to `BEST_MODEL_PATH`.
        *   Sets `model_ready` flag.

11. **Cell 12: Full Test Set Evaluation**
    *   If `model_ready` and test data is available:
        *   Evaluates the final model on the entire test set.
        *   Calculates and prints accuracy, classification report.
        *   Calculates and prints inference time statistics.
        *   Generates and displays a confusion matrix plot.

12. **Cell 13: Inference Function Definition**
    *   `predict_video()`: Function to take a single video path, preprocess it, run inference using the loaded model, and return the predicted action name and confidence.

13. **Cell 14: Inference Example with Thumbnails**
    *   Selects `NUM_VIS_SAMPLES` random videos from the test set.
    *   Generates thumbnails for these videos.
    *   Runs `predict_video()` on each.
    *   Displays the thumbnails in a grid with actual and predicted labels (colored by correctness) and confidence scores.

14. **Cell 15: User Input Video Inference**
    *   Creates an `ipywidgets.FileUpload` widget to allow users to upload a video file.
    *   `on_upload_change()`: Handles file upload, saves the video, and displays its thumbnail and a "Predict Action" button.
    *   `on_predict_button_clicked()`: When the button is clicked, it calls `predict_video()` on the uploaded file and displays the result.

## Configuration Parameters (Key ones in Cell 3)

*   `DEVICE`: Automatically set to `cuda` if available, else `cpu`.
*   `CURRENT_SPLIT`: HMDB51 split to use (1, 2, or 3).
*   `NUM_CLASSES`: 51 for HMDB51.
*   `NUM_FRAMES`: Number of frames sampled per video (e.g., 16).
*   `IMG_SIZE`: Spatial size of frames (e.g., 112x112).
*   `NUM_EXPERTS`: Number of experts in the MoE layer.
*   `TOP_K`: Number of experts to route to in MoE.
*   `NOISY_GATING_STD`: Standard deviation for noise added to gate logits during training (0 to disable).
*   `MOTION_FEATURE_DIM`: Intermediate dimension for the Motion Dynamics Module.
*   `BATCH_SIZE`: Batch size for training and evaluation.
*   `LEARNING_RATE`: Initial learning rate for the optimizer.
*   `NUM_EPOCHS`: Number of training epochs (if no checkpoint is found).
*   `LOAD_BALANCE_ALPHA`: Coefficient for the MoE load balancing loss.
*   `WEIGHT_DECAY`: Weight decay for the optimizer.

## Output

*   **Saved Model:** The best trained model checkpoint is saved to `/kaggle/working/models/hmdb51_dynamic_moe_split<N>_best.pth`.
*   **Console Output:** Detailed logs during data download, extraction, training progress, and evaluation results.
*   **Evaluation Metrics:** Accuracy, classification report, average inference time.
*   **Visualizations:**
    *   Confusion matrix plot for the test set evaluation.
    *   Grid of thumbnails with predictions for sample test videos.
    *   Thumbnail and prediction for user-uploaded videos.
*   **Extracted Data:** The HMDB51 dataset is extracted to `/kaggle/working/data/`.

## Troubleshooting & Notes

*   **CUDA Out of Memory:** If you encounter `CUDA out of memory` errors, try:
    *   Reducing `BATCH_SIZE`.
    *   Restarting the Kaggle kernel and running cells again. The script includes `gc.collect()` and `torch.cuda.empty_cache()` at various points, but sometimes a full restart is needed.
*   **Download/Extraction Issues:** Ensure "Internet" is turned ON in your Kaggle Notebook settings. If downloads or extractions fail, check the error messages. The script attempts to verify extractions.
*   **Long Runtimes:**
    *   Downloading and extracting HMDB51 (especially `hmdb51_org.rar`) can take a significant amount of time (15-30+ minutes for extraction).
    *   Training for `NUM_EPOCHS` can also be time-consuming, depending on the GPU and batch size.
*   **Dataset Path Errors:** Ensure the paths in the Configuration cell are correct, especially if modifying the script to run outside Kaggle.
*   **Widget Issues:** If the upload widget or buttons in Cell 15 don't behave as expected, try re-running Cell 15. Sometimes, re-registering observers helps.

## Forcing Retraining

To force the model to retrain even if a checkpoint exists for the `CURRENT_SPLIT`:
1.  Manually delete the specific `.pth` file from the `/kaggle/working/models/` directory via the Kaggle UI.
2.  OR, uncomment and run the "Optional: Clear Saved Model Checkpoint" cell (Cell 10 in the provided script structure), which removes the entire `models` directory.

This will cause the script in Cell 11 to proceed with training a new model.
