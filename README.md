# PriceVision 🔮
### Multi-Modal Product Price Prediction

---

## 📘 Overview

PriceVision is a multi-modal machine learning project developed as part of the Amazon ML Challenge 2025. The goal of this project was to predict product prices by combining insights from both textual descriptions and product images — leveraging the power of multi-modal learning.

This challenge involved processing and understanding 150K+ product data points, consisting of product titles, descriptions, and corresponding images, to build an efficient and accurate predictive model.

---

## 🧩 Key Features

* 🧾 Text Data Extraction and Processing – Performed extensive text cleaning, tokenization, and feature extraction from product metadata using NLP techniques.
* 🖼️ Image Feature Extraction – Used deep learning models (transfer learning) to extract meaningful visual features from product images.
* 🔗 Multi-Modal Fusion – Combined textual and visual embeddings for holistic price prediction.
* ⚙️ Data Preprocessing & Feature Engineering – Implemented optimized data handling using Kaggle Notebooks for large-scale computation and GPU acceleration.
* 📊 Evaluation & Submission – Generated two outputs:
    * test_out_lgb.csv → Output from Text Extraction Notebook (text-based predictions).
    * submission_final.csv → Final combined predictions integrating both image and text features.

---

## 💻 Project Structure


.
├── Image Extraction Code.ipynb     # Handles image data processing and feature extraction
├── Text Extraction code.ipynb    # Handles text preprocessing and model training on textual data
├── train.csv                     # Training dataset
├── test.csv                      # Testing dataset
├── test_out_lgb.csv              # Predictions from text model
└── submission_final.csv          # Final combined multi-modal predictions

---

## 🧠 Tech Stack

* Core: Python
* ML / DL: TensorFlow / Keras, Scikit-learn
* Data Handling: Pandas, NumPy
* Image Processing: OpenCV
* Visualization: Matplotlib / Seaborn
* Environment: Kaggle Notebooks (for dataset management, GPU acceleration, and model training)

---

## 🚀 Approach Summary

1.  Data Collection & Exploration
    * Analyzed textual and image data provided in the challenge dataset.

2.  Text Processing (Text Extraction Notebook)
    * Cleaned and vectorized text data using TF-IDF/Word Embeddings.
    * Trained a regression model for preliminary price prediction.

3.  Image Processing (Image Extraction Notebook)
    * Extracted features from product images using CNN-based architectures.
    * Predicted prices based on visual cues.

4.  Multi-Modal Fusion
    * Combined predictions from both models for final price estimation.

5.  Evaluation & Submission
    * Generated the final submission file (submission_final.csv) for leaderboard evaluation.

---

## 🏆 Results

Successfully implemented a multi-modal deep learning pipeline using both text and image features.

---

## 🌐 Environment

Developed entirely on Kaggle Notebooks for seamless dataset integration, GPU computation, and collaborative model experimentation.
