# Waste Classification Web App

This web application allows users to classify waste by uploading an image. The app leverages neural network models like **MobileNet** and **DeepLab** to classify waste efficiently and generate a segmentation mask. Additionally, it provides recycling tips and local recycling resources based on the user’s province or territory. The app is built using **Gradio** and has been deployed on **Hugging Face Spaces**.

---

## Features

1. **Image Upload:**
   - Upload an image of waste by selecting it from a local folder or using the phone camera.

2. **Efficient Classification:**
   - Quickly classify the waste using **MobileNet**, designed for fast processing.

3. **Detailed Results:**
   - Display the classification result and its confidence score.

4. **Waste Highlighting:**
   - Generate a segmentation mask highlighting the waste in the uploaded image using **DeepLab**, along with a segmentation confidence score.

5. **Recycling Tips:**
   - Provide actionable tips for each waste class, such as cleaning instructions and the appropriate disposal bin.

6. **Localized Recycling Resources:**
   - Allow users to select their province or territory and display a link to the respective recycling website after classification.

---

## Live Demo

Try the app live here: [Waste Classification Web App](https://huggingface.co/spaces/zhangzi0902/CS5330_Waste_Sorting)

**Note:** If you're using Safari, the UI may be displayed incorrectly. For the best experience, we recommend using Google Chrome.

---

## Getting Started

### Prerequisites

- Libraries: `gradio`, `numpy`, `torch`, `torchvision`, `pillow`, `opencv`. Please check `requirements.txt` for details.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/kevinbjl/garbage-classification.git
   cd garbage-classification

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
3. Run the application locally:
   ```
   python app.py
4. Open the app in your browser at [Localhost](http://localhost:7860)

---

## Instructions

1. Open the app on [Hugging Face Spaces](https://huggingface.co/spaces/zhangzi0902/CS5330_Waste_Sorting) or run it locally.
2. Upload an image of waste by selecting it from your local folder or capturing it using your phone camera (if you're opening the web app on your phone).
3. Select your province or territory from the dropdown menu, then click `Submit` to submit the image.
4. View the classification result and its confidence score.
5. Review the segmentation mask and its confidence score to identify the highlighted waste area.
6. Check the provided tips for proper waste handling and disposal.
7. Access the local recycling website for more information.

---

## Code Structure
    ├── gradio_interface/          
    │   └── app.py              # Main file for running the app
    ├── requirements.txt        # Python dependencies
    ├── mobilenet/              # Files relevant to the MobileNet model 
    │   ├── data/               # Dataset used for training the model
    │   ├── dataset_link.txt    # A link to the dataset
    │   ├── classifier.pth      # The pre-trained MobileNet model
    │   ├── run_model.ipynb     # Demo on loading and using the model
    │   └── train_model.ipynb   # MobileNet model implementation
    ├── deeplab/                # Files relevant to the DeepLab model
    │   ├── data/               # Dataset used for training the model
    │   ├── dataset_link.txt    # A link to the dataset
    │   ├── dataset.py          # PyTorch Dataset class used for training
    │   ├── download.py         # Script that downloads the dataset
    │   ├── LICENSE.txt         # License information from the dataset
    │   ├── run_model.ipynb     # Demo on loading and using the model
    │   ├── segmentation.pth    # The pre-trained DeepLab model
    │   └── train_model.ipynb   # DeepLab model implementation
    └── README.md               # Project documentation

---
## Acknowledgements

This project includes code from [TACO](https://github.com/pedropro/TACO) which is licensed under the [MIT License](https://opensource.org/licenses/MIT).