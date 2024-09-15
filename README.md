# Question Duplicate Detection using Random Forest and Bag of Words (BoW)

### Project Overview
This project aims to predict whether two questions have the same meaning using a Bag of Words (BoW) approach and a RandomForestClassifier. The dataset consists of question pairs and their corresponding labels (1 if the questions have the same meaning and 0 otherwise). The goal is to identify duplicate questions effectively using a machine learning pipeline with feature engineering and model training.

### Dataset Description
The dataset contains pairs of questions, and the task is to determine whether the two questions in each pair are duplicates. The labels are generated based on human judgment, which may sometimes be subjective or noisy. However, the overall dataset represents a reasonable consensus on what constitutes duplicate questions.

#### Data Fields
- **id**: The id of a training set question pair.
- **qid1**: Unique id for the first question.
- **qid2**: Unique id for the second question.
- **question1**: The full text of the first question.
- **question2**: The full text of the second question.
- **is_duplicate**: The target variable, set to `1` if `question1` and `question2` have the same meaning, and `0` otherwise.

### Bag of Words (BoW) Model
The BoW model is a simple yet powerful method for representing text data. It converts text into numerical vectors based on word frequency while disregarding word order and grammar.

#### BoW Steps:
1. **Tokenization**: Split text into individual words (tokens).
2. **Vocabulary Creation**: Create a list of all unique words in the dataset.
3. **Vectorization**: Each document is represented as a vector where each element is the frequency of a word from the vocabulary.

#### Example:
For two sentences:
- Sentence 1: "The cat sat on the mat."
- Sentence 2: "The dog sat on the log."

Vocabulary: `["The", "cat", "sat", "on", "the", "mat", "dog", "log"]`

BoW Representation:
- Sentence 1: `[2, 1, 1, 1, 1, 1, 0, 0]`
- Sentence 2: `[2, 0, 1, 1, 1, 0, 1, 1]`

### Vectorization Techniques
- **Count Vectorization (BoW)**: Represents text by counting word frequencies.
- **TF-IDF**: Adjusts word frequency based on importance across all documents.
- **Word Embeddings**: Dense vectors capturing the semantic meaning of words.
- **Transformers**: Contextual embeddings using pre-trained models like BERT or GPT.

### Feature Engineering
Feature engineering is an essential step to extract more relevant features from the raw text. For this project, the following features were engineered:

- `q1_len` → Length of `question1` in characters.
- `q2_len` → Length of `question2` in characters.
- `q1_words` → Number of words in `question1`.
- `q2_words` → Number of words in `question2`.
- `words_common` → Number of unique words common between `question1` and `question2`.
- `words_total` → Total number of words in both questions combined.
- `words_share` → Ratio of common words to the total number of words.

### Modeling
The dataset is transformed using the BoW vectorizer, and a **RandomForestClassifier** is trained using the extracted features. For every 30,000 rows, 3,000 BoW features for `question1` and 3,000 BoW features for `question2` were created, resulting in a total of 6,007 features (6,000 from BoW and 7 from feature engineering).

### Installation
To set up and run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/duplicate-question-pairs.git
