# SMS Spam Detection using NLP & Naive Bayes
An end-to-end text classification pipeline to detect SMS spam using 
NLP preprocessing (NLTK) and a Multinomial Naive Bayes classifier.

## Results

| Metric | Score |
|---|---|
| Train Accuracy | 99.19% |
| Test Accuracy | 98.21% |
| Cross-Validation (5-fold) | 97.68% |

Minimal gap between train/test/CV scores confirms no overfitting.

## Pipeline
Raw SMS → Remove Punctuation → Lowercase → Remove Stopwords →
PorterStemmer → CountVectorizer (6,296 features) → MultinomialNB
## Dataset

UCI SMS Spam Collection — 5,572 messages (4,825 Ham / 747 Spam)

## Tech Stack

Python · NLTK · Scikit-learn · Pandas · CountVectorizer · Multinomial Naive Bayes

## Setup

```bash
pip install pandas scikit-learn nltk

python
import nltk
nltk.download('stopwords')
```

Update the dataset path in the notebook:
python
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["class", "message"])
```

Then run: `Text Classification using NLP-checkpoint.ipynb`
