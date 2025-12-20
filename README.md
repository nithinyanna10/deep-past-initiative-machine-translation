# Deep Past Initiative - Machine Translation Challenge

## Overview
This project aims to build neural machine translation models for Old Assyrian cuneiform tablets, translating transliterated Akkadian into English.

## Dataset
- **train.csv**: ~1,500 transliterations with English translations
- **test.csv**: ~4,000 sentences to translate (sentence-level alignment)
- **published_texts.csv**: ~8,000 transliterations without translations
- **publications.csv**: ~880 scholarly publications (OCR text) with potential translations
- **OA_Lexicon_eBL.csv**: ~39,000 word lexicon with dictionary links
- **bibliography.csv**: Bibliographic data for publications

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
```

## Project Structure
```
.
├── data/              # Data exploration and preprocessing scripts
├── models/            # Model training and evaluation scripts
├── notebooks/         # Jupyter notebooks for analysis
├── utils/             # Utility functions
└── submissions/       # Generated submission files
```

## Evaluation Metric
Geometric Mean of BLEU and chrF++ scores (micro-averaged)

## References
- Competition: [Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-challenge)
- Dataset Instructions: See competition page
- OARE Database: https://oare.byu.edu/
- CDLI: https://cdli.earth/

