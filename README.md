```markdown
# License Plate Detection and OCR Project

This project leverages YOLO (You Only Look Once) for object detection and TrOCR (Transformer-based OCR) for recognizing text on vehicle license plates. The processed results are stored in an Excel file, including the cropped license plate images and their recognized text.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Running the Project](#running-the-project)
  - [Step 1: License Plate Detection with YOLO](#step-1-license-plate-detection-with-yolo)
  - [Step 2: Text Recognition with TrOCR](#step-2-text-recognition-with-trocr)
  - [Step 3: Saving Results to Excel](#step-3-saving-results-to-excel)
- [Output](#output)
- [Notes](#notes)
- [Acknowledgments](#acknowledgments)

## Overview

The goal of this project is to automate the detection of vehicle license plates and recognize their text content. The project is divided into three main parts:
1. **License Plate Detection**: Using YOLO to detect the location of the license plate on each vehicle image.
2. **Optical Character Recognition (OCR)**: Using TrOCR to read and recognize text on the detected license plate.
3. **Exporting Results**: Saving the image ID, cropped license plate image, and recognized text into an Excel file.

This project can be useful for scenarios such as automated toll collection, parking systems, and vehicle tracking.

## Getting Started

Follow the steps below to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python 3.8 or later installed. This project requires several libraries, including:

- [YOLO](https://docs.ultralytics.com/) for license plate detection.
- [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) for OCR from the Hugging Face Transformers library.
- `pandas`, `Pillow`, and `OpenCV` for image processing and Excel file creation.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/license-plate-detection-ocr.git
   cd license-plate-detection-ocr
   ```

2. Install the required Python packages:
   ```bash
   pip install pandas pillow opencv-python torch torchvision transformers openpyxl ultralytics
   ```

3. Download or train a YOLO model for license plate detection and save it in the specified path, e.g., `best.pt`. You can also use a pre-trained YOLO model and adapt it as needed.

4. Download or fine-tune a TrOCR model. Save it in the `output_dir` you’ll use in the script.

## Project Structure

- `01_Plate_detection_Model_training.ipynb`: Notebook for training the YOLO model.
- `02_TrOCR_model_training.ipynb`: Notebook for training the TrOCR model.
- `03_Using_trained_model.ipynb`: Notebook for running the detection and OCR process, then exporting results.

## Data Preparation

1. **YOLO Model**: Download or train a YOLO model for detecting license plates.
   - Save the model weights (e.g., `best.pt`) in a known location for loading in the code.
   
2. **TrOCR Model**: Download or fine-tune a TrOCR model for recognizing license plate text.
   - Save the fine-tuned model and processor configuration in a directory (e.g., `output_dir`).

3. **Test Images**: Place images to be processed in a `test` folder in the root directory. Ensure each image contains a vehicle with a visible license plate for best results.

## Running the Project

Each step below corresponds to a notebook. Ensure you run them in the order provided:

### Step 1: License Plate Detection with YOLO

In `03_Using_trained_model.ipynb`, the first part loads a pre-trained YOLO model and uses it to detect license plates in the images located in the `test` folder. It performs the following steps:

- **Load YOLO Model**: Load the pre-trained YOLO weights for license plate detection.
- **Process Each Image**: For each image in the test folder, YOLO will:
  - Detect the license plate in the image.
  - Extract the bounding box coordinates of the license plate with the highest confidence.
  - Crop the image to isolate the license plate.

### Step 2: Text Recognition with TrOCR

The cropped image of the license plate is passed to a TrOCR model for text recognition.

- **Load TrOCR Model**: Load the fine-tuned TrOCR model and processor for OCR.
- **Image Preprocessing**: The cropped license plate image is resized and transformed to the input format expected by the TrOCR model.
- **Text Prediction**: The model generates text based on the cropped license plate image.

### Step 3: Saving Results to Excel

The results are saved to an Excel file named `license_plate_results.xlsx`, with each row containing:
- The **Image ID** (original filename).
- The **Cropped License Plate Image**.
- The **Recognized Text** from the license plate.

The code in the notebook will:

1. Create an Excel file using `openpyxl`.
2. Insert each cropped license plate image and its recognized text in separate columns for each image.
3. Save the results with properly adjusted row heights and column widths for clear display.

## Output

The final output is an Excel file named `license_plate_results.xlsx` with the following columns:

1. **Image ID**: Filename of the image processed.
2. **Cropped Image**: A thumbnail of the detected license plate.
3. **Recognized Text**: The OCR-detected text from the license plate.

This Excel file can be found in the root directory after the code execution completes.

## Notes

- **Model Fine-tuning**: For best results, fine-tune YOLO and TrOCR models on your specific dataset, as license plate formats and text styles may vary.
- **Image Quality**: Ensure high-quality images with clear license plates for accurate detection and OCR.
- **Parameter Tuning**: Adjust YOLO’s confidence threshold, bounding box dimensions, and TrOCR’s max length to optimize performance based on your dataset.

## Acknowledgments

- **YOLO**: We used YOLO for efficient object detection on vehicle images.
- **TrOCR**: Hugging Face’s TrOCR model was used for OCR, taking advantage of Transformer-based models for robust text recognition.

For further information on the models and libraries used, see:
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Hugging Face TrOCR Documentation](https://huggingface.co/docs/transformers/model_doc/trocr)

Feel free to reach out with questions or feedback by submitting an issue on this repository.
```
