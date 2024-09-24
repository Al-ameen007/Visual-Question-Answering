# Visual Question Answering with BLIP Model

This project demonstrates how to use the BLIP (Bootstrapping Language-Image Pre-training) model for Visual Question Answering (VQA), with a focus on comparing a base pre-trained model with a fine-tuned model. The goal is to evaluate the performance of the BLIP model on the VizWiz dataset by generating predictions from both models and visualizing the results to assess improvements from task-specific fine-tuning.

## Project Overview

The purpose of this project is to evaluate how fine-tuning the BLIP model improves Visual Question Answering performance on the VizWiz dataset. By comparing a base model pre-trained on general VQA tasks with a model fine-tuned on the VizWiz dataset, the notebook demonstrates the benefits of task-specific training.

### Workflow:
1. **Loading the dataset** (VizWiz).
2. **Preparing the models** (Base and Fine-Tuned BLIP).
3. **Generating predictions** for image-question pairs.
4. **Visualizing and comparing** the performance of both models.

---

## Key Sections

### 1. Environment Setup
The environment setup includes installing necessary libraries such as Hugging Face’s `transformers` and `datasets` for model loading and dataset processing. We also install `rouge` for evaluation purposes. This ensures that all dependencies are in place to process data and run models.

### 2. Dataset Loading
The VizWiz dataset is used for the Visual Question Answering task, consisting of images paired with questions. The notebook loads both the validation and test sets, where the validation set is used for model predictions, and the test set can be used for future generalization performance assessment.

The dataset consists of image-question-answer triples that are fed into the model for inference.

### 3. Model Preparation
We load two versions of the BLIP model:
- **Base Model:** Pre-trained on general VQA tasks, but not specifically fine-tuned on VizWiz.
- **Fine-Tuned Model:** Fine-tuned on VizWiz data, optimizing it for better performance in this specific domain.

Both models are prepared for inference by loading them onto a suitable device (e.g., GPU) and utilizing a processor to handle image-question pair preprocessing.

### 4. Inference Process
During the inference step, image-question pairs from the validation dataset are fed into both models. For each pair, two predictions are generated:
- One from the base model.
- One from the fine-tuned model.

The notebook processes these pairs by converting them into a format the models can understand, generating predictions, and then converting those predictions into human-readable text.

### 5. Results Comparison
This section compares the predictions from both models for each image-question pair. The comparison includes:
- The original question.
- The correct answer from the dataset.
- The predicted answer from the base model.
- The predicted answer from the fine-tuned model.

This comparison helps illustrate how fine-tuning on VizWiz data improves model accuracy and adaptation to task-specific requirements.

### 6. Visualization of Results
To make the comparison more intuitive, the notebook visualizes the image, the question, and the predicted answers from both models. Each visualization includes:
- The dataset image.
- The question related to the image.
- The correct answer.
- Predictions from the base and fine-tuned models.

These graphical comparisons highlight the strengths and weaknesses of each model.

### 7. Evaluation Metrics
Quantitative evaluation metrics are used to compare the models’ performance, with metrics like ROUGE being used to measure the similarity between the predicted answers and the correct answers. These metrics provide insights into how well the models are performing in terms of answer accuracy.

### 8. Conclusion
The project concludes with a comparison of the base and fine-tuned models, demonstrating that fine-tuning on the VizWiz dataset leads to better VQA performance. Both the visual and quantitative comparisons indicate that fine-tuning improves the model’s ability to understand context in image-based questions.

---

## How to Run
1. Set up the environment by installing the necessary dependencies.
2. Load the VizWiz dataset and prepare the models.
3. Run the inference to generate predictions.
4. Compare and visualize the results.

