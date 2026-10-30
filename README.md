# multimodal-breast-ultrasound-vlm
This project develops an end-to-end multimodal breast ultrasound decision-support system that combines breast ultrasound images with structured clinical data to automatically segment breast lesions, generate a diagnostic prediction, and produce a structured diagnostic report.

<img width="940" height="394" alt="image" src="https://github.com/user-attachments/assets/c3fe4bbb-a0f2-4d1c-8bda-bf396672ccbf" />

The system takes in two main inputs, a breast ultrasound image and clinical information. 

The ultrasound image is first passed through the segmentation component, where a U-Net model generates a lesion mask. This predicted mask is then used to identify the lesion region without the need for expert-provided segmentation.

The ultrasound image is processed using EfficientNet to extract visual features, while the structured clinical information is converted into a suitable textual representation and encoded using ClinicalBERT. 

The two features are then combined using a cross-attention-based multimodal fusion mechanism.
The fused features are fed to two components. A classification head which predicts whether the lesion is benign or malignant and a GPT-2 based report generation component which produces a structured report

The final system produces three outputs:

•	predicted lesion segmentation mask

•	benign or malignant classification 

•	structured diagnostic report

----------------------------------------------------------------------------------------------
Directory Overview

notebooks: The Main development and implementation of the system can be found in this folder

src: Contains the reusable Python modules for preprocessing, segmentation, encoding, fusion, classification, report generation, and inference.

models: Stores the trained model configuration files and the trained model checkpoints.

outputs: Contains final evaluation results, predictions, and generated report metrics


App.py is a streamlit based web interface used to demonstrate the system


SetUp:
'''
Run these commands in the terminal to set up the virtual environment and install dependencies:
1. Create a virtual environment:
python -m venv .venv

2. Activate the virtual environment:
.venv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

4. select the interpreter below:
.venv\Scripts\python.exe

5.Once all notebooks are executed, run the command below to run the streamlit website:
python -m streamlit run app.py        

''' 
