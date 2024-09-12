

# Sentiment Analysis of Movie Reviews using Word2Vec and LSTM

This project explores sentiment analysis on movie reviews using Long Short-Term Memory (LSTM) networks combined with Word2Vec embeddings. The model aims to improve sentiment classification precision and contextual understanding, achieving a comparative accuracy of 85%. This study builds on previous research on sentiment analysis of Indonesian hotel reviews, applying and extending these techniques to a novel dataset of movie reviews. 

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Dataset Description](#dataset-description)
4. [Project Description](#project-description)
5. [Comparison with References](#comparison-with-references)
6. [Experimental Results](#experimental-results)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)
9. [References](#references)
10. [Installation and Usage](#installation-and-usage)
11. [File Structure](#file-structure)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)

## Abstract

This project investigates sentiment analysis in the context of movie reviews using LSTM networks and Word2Vec embeddings. We enhance precision and contextual understanding compared to previous work focused on Indonesian hotel reviews. We present a comparison with GloVe embeddings and introduce a custom learning rate in the Adam optimizer for model fine-tuning. The project contributes to understanding sentiment analysis robustness and adaptability across different domains.

## Introduction

Sentiment analysis is essential in NLP for determining the emotional tone of text. This project adapts LSTM with Word2Vec to movie reviews, extending previous research on Indonesian hotel reviews. We aim to offer insights into sentiment analysis in a new context, emphasizing the adaptation of techniques to different datasets.

## Dataset Description

The dataset used consists of movie reviews, differing from hotel reviews in sentiment distribution and content. It was pre-processed with case folding, tokenization, stopword removal, stemming, and padding to prepare it for analysis.

## Project Description

The project integrates Word2Vec for word embeddings and LSTM for sequence modeling. Key aspects include:
- **Word2Vec Model:** Trained with a skip-gram architecture, hierarchical softmax, vector dimension of 300, window size of 5, and minimum word count of 1.
- **LSTM Model:** Uses Word2Vec embeddings, includes a dropout layer, employs a learning rate of 0.001 with the Adam optimizer, and applies global average pooling and binary cross-entropy loss for classification.

## Comparison with References

This project adapts and extends techniques from previous research:
- **Data Source:** Shifted from Indonesian hotel reviews to movie reviews.
- **Word2Vec Dimensionality:** Set at 300 dimensions.
- **Custom Learning Rate:** Introduced in the Adam optimizer.

## Experimental Results

- **GloVe Accuracy:** 78.3%, with balanced precision-recall.
- **Word2Vec Accuracy:** 66.49%, with noted precision-recall imbalance.

## Conclusion

The project successfully applies sentiment analysis techniques to a new domain, comparing LSTM with Word2Vec to GloVe. The custom learning rate in Adam shows promise for model improvement. The study highlights areas for future refinement and contributes to sentiment analysis research.

## Future Work

- **Hyperparameter Tuning:** Experiment with dropout rates, LSTM configurations, and other hyperparameters.
- **Alternative Embeddings:** Explore techniques like FastText or ELMo.
- **Genre-specific Features:** Analyze sentiment variations based on movie genres.

## References

1. **Reference Paper on Sentiment Analysis Using Word2Vec And LSTM for Indonesian Hotel Reviews:**
   - Provides the foundational framework for sentiment analysis using Word2Vec and LSTM techniques.

2. **GloVe Embedding Technique Documentation:**
   - Offers insights into GloVe, an alternative word embedding method for comparative analysis.

## Installation and Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-lstm.git
   cd sentiment-analysis-lstm
   ```

2. **Install Dependencies:**

   Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data:**

   Place the dataset (e.g., `movie_reviews.csv`) in the `data` directory.

4. **Train and run the Model:**

   ```bash
   python sentiment.py
   ```

   ```

## File Structure

- `data/`: Contains dataset file.
- `models/`: Directory for saving models and embeddings.
- `train_model.py`: Script for training the LSTM model.
- `evaluate_model.py`: Script for model evaluation.
- `predict_sentiment.py`: Script for sentiment prediction.
- `requirements.txt`: List of dependencies.

## Troubleshooting

- **Script Errors:** Ensure all dependencies are installed and the dataset is correctly placed.
- **Accuracy Issues:** Consider tuning hyperparameters or exploring alternative embeddings.

## Contributing

Contributions are welcome. Please fork the repository, make changes, and submit a pull request.

1. Fork the Project.
2. Create a Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit Your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
