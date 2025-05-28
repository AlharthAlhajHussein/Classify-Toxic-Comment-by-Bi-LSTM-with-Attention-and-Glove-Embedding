# 💬 Toxic Comment Classification with Bi-LSTM, Attention & GloVe Embeddings 🔬

[![Python Version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11_|_3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras Badge](https://img.shields.io/badge/Keras-Deep_Learning-red.svg)](https://keras.io/)
<!--[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) <!-- You can create a LICENSE file -->

This project presents a robust system for detecting and classifying toxic comments into six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. It employs a Bidirectional LSTM (Bi-LSTM) model enhanced with an Attention mechanism and pre-trained GloVe word embeddings. The model demonstrates strong performance, achieving a ROC AUC score of approximately 0.977. A key feature of this project is a user-friendly Graphical User Interface (GUI) built with CustomTkinter, allowing for easy, real-time analysis of comments.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Exploratory Data Analysis (EDA) Insights](#exploratory-data-analysis-eda-insights)
- [Model Architecture & Details](#model-architecture--details)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 📝 Overview

Addressing online toxicity is a critical challenge in maintaining healthy digital communities. This project offers a sophisticated solution by leveraging state-of-the-art deep learning techniques. The core model utilizes:
-   **Bidirectional LSTMs (Bi-LSTMs)** to process text sequences, capturing contextual information from both past and future words.
-   An **Attention Mechanism** (custom or TensorFlow built-in) to allow the model to dynamically focus on words that are most indicative of toxicity.
-   **Pre-trained GloVe Word Embeddings** to initialize the model with rich semantic representations of words, significantly boosting performance and reducing training time.

The project also features an interactive GUI where users can input text and receive immediate feedback on its toxicity profile, visualized through dynamically colored progress bars for each category.

## 🌟 Features

-   **Multi-Label Toxic Comment Classification**: Identifies and assigns probabilities for six distinct types of toxicity.
-   **Advanced Deep Learning Model**:
    -   Bidirectional LSTM layers for comprehensive sequence understanding.
    -   Flexible Attention Mechanism: Includes a custom `AttentionLayer`, with options to use `tf.keras.layers.Attention` or `tf.keras.layers.MultiHeadAttention` for more complex attention patterns.
    -   Utilizes GloVe embeddings (configurable, e.g., 6B or 840B parameters) for superior semantic capture.
-   **Robust Text Preprocessing**: A dedicated `TextCleaner` class (`clean_utilities.py`) handles:
    -   HTML and XML tag removal.
    -   Non-ASCII character removal.
    -   Expansion of contractions (e.g., "don't" to "do not").
    -   Replacement of special entities like URLs, emails, dates, and times with placeholders.
    -   Punctuation and number removal.
    -   Optional stopword removal and stemming.
-   **Class Imbalance Management**: Incorporates `FocalLoss` (custom implementation in `model_utilities.py`) as an option to effectively train on imbalanced datasets where toxic comments are a minority.
-   **Interactive and Intuitive GUI**:
    -   Developed with `CustomTkinter` for a modern and responsive user experience.
    -   Real-time comment analysis: input text and get instant toxicity scores.
    -   Clear visualization: Probabilities for each category are shown with numerical scores and dynamically colored progress bars (Green for low, Orange for medium, Red for high toxicity).
-   **Modular and Well-Organized Code**:
    -   `model_utilities.py`: Contains functions for data loading, model building (multiple variants), training, evaluation, prediction, and model saving/loading.
    -   `clean_utilities.py`: Houses the `TextCleaner` class.
    -   Jupyter Notebooks for EDA (`EDAv1.2.ipynb`) and model development/experimentation (`bi-lstm-with-attention-and-glove-embedding-97-7.ipynb`).

## 🎬 Demo

**Example Placeholder:**

![GUI Screenshot Placeholder](https://github.com/AlharthAlhajHussein/Classify-Toxic-Comment-by-Bi-LSTM-with-Attention-and-Glove-Embedding/blob/main/images/app_screenshot.png)

*Caption: The Toxic Comment Analyzer GUI displaying probability scores for a sample comment.* 

## 📂 Project Structure

```
📦 ToxicCommentClassification
 ┣ 📂 LSTM                         # Main project directory
 ┃ ┣ 📜 app_gui.py                 # The CustomTkinter GUI application
 ┃ ┣ 📜 model_utilities.py         # Core utilities for model, data, building, training, etc.
 ┃ ┣ 📜 clean_utilities.py         # Text cleaning and preprocessing logic
 ┃ ┣ 📜 bi-lstm-with-attention-and-glove-embedding-97-7.ipynb  # Notebook for model building training
 ┃ ┣ 📜 EDAv1.2.ipynb              # Notebook for Exploratory Data Analysis on Toxic comments data
 ┃ ┣ 📂 model and tokenizer        # Directory for storing trained model and tokenizer 
 ┃ ┃ ┣ 📜 toxic_model.keras        # Trained Keras model file
 ┃ ┃ ┗ 📜 preprocessing_data.pkl   # Pickled tokenizer & other preprocessing objects
 ┃ ┣ 📜 README.md                  # This file
 ┃ ┗ 📜 requirements.txt           # Python package dependencies
 ┣ 📜 train.csv                    # training dataset (e.g., from Kaggle)
 ┣ 📜 test.csv                     # test dataset (e.g., from Kaggle)
 ┣ 📜 test_labels.csv              # test labels predictions 
 ┗ 📂 Images                       # Optional: For storing images used in README
   ┗ 📜 gui_screenshot.png         # (Example: actual GUI screenshot)
```

## 🛠️ Technology Stack

-   **Programming Language**: Python 3.8+
-   **Deep Learning Framework**: TensorFlow 2.x, Keras
-   **Natural Language Processing**: NLTK (for tokenization, stopwords), Contractions, BeautifulSoup4 (for HTML parsing)
-   **Data Manipulation & Analysis**: Pandas, NumPy
-   **Graphical User Interface (GUI)**: CustomTkinter
-   **Data Visualization**: Matplotlib, Seaborn (primarily for EDA in notebooks)
-   **Development Environment**: Jupyter Notebooks, Standard Python IDEs

## 📊 Exploratory Data Analysis (EDA) Insights

The `EDAv1.2.ipynb` notebook provides crucial insights into the dataset:
-   **Label Distribution**: Visualized the counts and percentages of comments for each toxic category, revealing significant class imbalance.
-   **Multi-Label Analysis**: Investigated the co-occurrence of multiple toxic labels within single comments, highlighting the complexity of toxic language.
-   **Correlation Matrix**: A heatmap illustrated the correlations between different toxicity types (e.g., 'obscene' and 'insult' often appear together).
-   **Comment Length vs. Toxicity**: Explored the distribution of comment character lengths and its potential relationship with the presence of toxicity.
These findings directly informed the data preprocessing strategies and the decision to include options like Focal Loss for handling class imbalance.


**Show class imbalance:**
![Screenshot](images\EDA1.png)
**Show correlation:**
![Screenshot](images\EDA2.png)
**Show length distribution:**
![Screenshot](images\EDA3.png)

## 📈 Model Architecture & Details

The primary model architecture, detailed in `bi-lstm-with-attention-and-glove-embedding-97-7.ipynb` and `model_utilities.py`, consists of:
-   **Input Layer**: Takes sequences of tokenized word indices.
-   **Embedding Layer**: Uses pre-trained GloVe word embeddings. This layer converts word indices into dense vectors. The embeddings can be kept non-trainable initially or fine-tuned.
-   **Bidirectional LSTM (Bi-LSTM) Layers**: One or more Bi-LSTM layers process the embedded sequences to capture contextual information from both directions (forward and backward).
-   **Attention Layer**: A custom `AttentionLayer` (or `tf.keras.layers.Attention`/`MultiHeadAttention`) is applied to the output of the Bi-LSTM to weigh the importance of different parts of the sequence.
-   **Dense Layers**: Fully connected layers for further feature transformation before classification.
-   **Batch Normalization & Dropout**: Used between layers to stabilize training and prevent overfitting.
-   **Output Layer**: A Dense layer with 6 units (one for each toxicity category) and a `sigmoid` activation function, producing independent probabilities for each label.

**Key Hyperparameters & Choices**:
-   **Optimizer**: Adam optimizer (learning rate is configurable).
-   **Loss Function**: `binary_crossentropy` (standard for multi-label classification). `FocalLoss` is available as an alternative for imbalanced data.
-   **Metrics**: `accuracy` and `AUC` (Area Under the ROC Curve, specifically `tf.keras.metrics.AUC(name='roc_auc')`).
-   **Performance**: The model has achieved a validation ROC AUC of approximately **0.977**, indicating strong discriminative ability.

## 📚 Dataset

The model is designed for datasets typically used in toxic comment classification, such as the one from the **Jigsaw Toxic Comment Classification Challenge** hosted on [Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview). This dataset contains a large corpus of Wikipedia comments, each labeled by human raters for six types of toxicity.
-   The `train.csv` file (or your specific dataset file) should ideally be placed in the project root or a designated data folder, and paths in the notebooks adjusted accordingly if necessary.

## ⚙️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AlharthAlhajHussein/Classify-Toxic-Comment-by-Bi-LSTM-with-Attention-and-Glove-Embedding.git
    cd Classify-Toxic-Comment-by-Bi-LSTM-with-Attention-and-Glove-Embedding/src # Navigate into the src directory
    ```

2.  **Create and Activate a Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Ensure you are in the `src` directory where `requirements.txt` is located.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model and Preprocessing Files:**
    Ensure your trained model (`toxic_model.keras`) and preprocessing data (`preprocessing_data.pkl`) are located in the `LSTM/Submissions/97649/` directory. If your model version or path differs, update the `MODEL_PATH` and `DATA_PATH` variables in `app_gui.py`.

## 🚀 Usage

**1. Running the Toxic Comment Analyzer GUI:**

   -   Navigate to the `LSTM` directory in your terminal.
   -   Execute the GUI application:
       ```bash
       python app_gui.py
       ```
   -   Enter any text comment into the input box and click "Analyze Toxicity". The GUI will display the probability scores for each of the six toxicity categories.

**2. Model Training and Exploration (Jupyter Notebooks):**

   -   To explore the data or retrain/experiment with the model, open and run the Jupyter Notebooks:
       -   `EDAv1.2.ipynb`: For detailed exploratory data analysis.
       -   `bi-lstm-with-attention-and-glove-embedding-97-7.ipynb`: For the model training pipeline, experimentation with different architectures (like self-attention, multi-head attention), and evaluation.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](<your-github-repo-url>/issues) if you want to contribute.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

Distributed under the Aleppo License. See `LICENSE` for more information.

## 👨‍💻 Author

**Alharth Alhaj Hussein**

Connect with me:
-   [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alharth-alhaj-hussein-023417241)
-   [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlharthAlhajHussein)
-   [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/alharthalhajhussein)
-   [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Alharth.Alhaj.Hussein)

---

If you find this project insightful or useful, please consider giving it a ⭐ on GitHub!
