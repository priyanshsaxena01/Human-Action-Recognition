# HMDB51 Action Recognition with R(2+1)D and Motion-Aware Dynamic Mixture of Experts

This project implements a video action recognition model using a pre-trained R(2+1)D backbone enhanced with a custom Motion Dynamics Module and a Dynamic Mixture of Experts (MoE) layer. The model is trained and evaluated on the HMDB51 dataset.

The script is designed to run in a Jupyter Notebook environment like Kaggle or Google Colab and includes functionality for:
1.  Automatic download and extraction of the HMDB51 dataset and its official train/test splits.
2.  Training the `ActionMoEDynamicModel` or loading a pre-trained checkpoint.
3.  Evaluating the model on the test set, providing metrics like accuracy, classification report, and a confusion matrix.
4.  Visualizing predictions on sample videos from the test set.
5.  An interactive interface for users to upload their own videos and get action predictions.

## Features

*   **Backbone:** Uses the R(2+1)D-18 model, pre-trained on Kinetics, as a feature extractor.
*   **Motion Dynamics Module:** A custom module that processes intermediate features from the backbone to explicitly capture temporal changes (motion) between adjacent feature map frames.
*   **Dynamic Mixture of Experts (MoE):**
    *   **MLP Gate:** A multi-layer perceptron acts as the gating network, providing more representational power for routing decisions.
    *   **Residual MLP Experts:** Experts are implemented as MLPs with residual connections to improve gradient flow and capacity.
    *   **Noisy Top-K Gating:** During training, Gaussian noise is added to gate logits before selecting the top-K experts, encouraging exploration and preventing expert collapse.
    *   **Load Balancing Loss:** An auxiliary loss term is used to encourage balanced utilization of experts.
*   **Dataset Handling:** Manages download, extraction (including nested RARs for HMDB51), and parsing of HMDB51 and its standard splits.
*   **Train/Load:** Automatically checks for an existing model checkpoint for the selected split and loads it. If not found, it trains a new model.
*   **Evaluation:** Comprehensive evaluation with accuracy, per-class precision/recall/F1-score, and confusion matrix.
*   **Inference:**
    *   Demonstrates inference on sample videos from the test set with thumbnail display.
    *   Provides an `ipywidgets`-based UI for uploading and predicting actions in user-provided videos.

## Requirements

*   Python 3.7+
*   PyTorch & TorchVision
*   `rarfile` (Python library for RAR files)
*   `unrar` (system utility, installed via `apt-get`)
*   NumPy
*   OpenCV-Python (`cv2`)
*   Requests
*   tqdm
*   scikit-learn
*   Seaborn
*   Matplotlib
*   Pandas
*   ipywidgets (for the interactive upload interface)

The script includes commands to install `rarfile` and `unrar` if run in a suitable environment (e.g., Kaggle/Colab). Other dependencies might need to be installed manually or via a `requirements.txt` file (not provided here, but can be generated).

```bash
pip install torch torchvision torchaudio
pip install rarfile numpy opencv-python requests tqdm scikit-learn seaborn matplotlib pandas ipywidgets
# On Debian/Ubuntu based systems for unrar:
# sudo apt-get update && sudo apt-get install -y unrar
