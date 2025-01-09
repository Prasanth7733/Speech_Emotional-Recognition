# Speech Emotional Recognition Project 🎤📈

Welcome to the **Speech Emotional Recognition Project**! This repository focuses on building a deep learning model to classify emotions from audio recordings using Long Short-Term Memory (LSTM) networks. The project includes robust feature extraction, data visualizations, and an efficient model architecture to achieve accurate emotion detection.

## Project Overview
Speech Emotion Recognition (SER) is a challenging yet essential task in the field of human-computer interaction. By leveraging the power of LSTMs, we aim to analyze temporal patterns in audio data and classify emotions such as happiness, sadness, anger, and more. The dataset used for this project contains labeled audio samples with various emotional expressions.

## Key Features
- **Data Visualization**: Includes visual analysis of audio signals, spectrograms, and feature distributions.
- **Feature Extraction**: Utilizes Mel-frequency cepstral coefficients (MFCCs) and other audio features.
- **LSTM-based Architecture**: Leverages the sequence modeling capabilities of LSTM for accurate predictions.

## Dataset
The dataset contains audio recordings labeled with different emotions. Each sample is processed to extract features that capture the emotional content of the speech.

### Dataset Attributes
- Audio samples with varying durations.
- Emotion labels: Happiness, Sadness, Anger, Neutral, etc.
- Features extracted include:
  - **MFCCs**: Represent spectral properties of audio signals.
  - **Chroma Features**: Capture pitch class profiles.
  - **Spectral Contrast**: Measure brightness and harmonics.

## Tools and Libraries
This project utilizes:
- **Python**: Programming language for implementation.
- **Librosa**: For audio processing and feature extraction.
- **Matplotlib & Seaborn**: For visualizing audio features and model performance.
- **TensorFlow/Keras**: Framework for building the LSTM model.
- **NumPy & Pandas**: For numerical computations and data manipulation.

## Project Workflow
1. **Data Exploration**:
   - Load and explore audio samples.
   - Visualize waveforms, spectrograms, and feature distributions.
2. **Feature Extraction**:
   - Extract MFCCs, chroma features, and spectral contrast.
   - Normalize features for consistent scaling.
3. **Model Development**:
   - Design an LSTM-based architecture to capture temporal patterns.
   - Add Dense layers for classification.
4. **Training and Evaluation**:
   - Train the model on extracted features.
   - Evaluate performance using metrics like accuracy and F1-score.
5. **Visualization**:
   - Plot loss and accuracy curves.
   - Display confusion matrix for classification results.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Prasanth7733/speech-emotion-recognition.git
   ```
2. Navigate to the project folder:
   ```bash
   cd speech-emotion-recognition
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the feature extraction script:
   ```bash
   python feature_extraction.py
   ```
5. Train the model:
   ```bash
   python train_model.py
   ```
6. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

## Results
- **Training Accuracy**: ~95%
- **Test Accuracy**: ~90%
- **Visualization Highlights**:
  - Waveforms and spectrograms of audio samples.
  - Confusion matrix for emotion classification.

## Acknowledgments
- **Dataset**: Acknowledgment to publicly available datasets used in this project.
- **Librosa**: For simplifying audio processing tasks.
- **TensorFlow/Keras**: For making deep learning implementation accessible.

---

Feel free to contribute or raise issues in this repository. Let’s make strides in understanding and interpreting human emotions through speech!

