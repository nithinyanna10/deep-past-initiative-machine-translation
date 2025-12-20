# Project Setup Summary

## ✅ Completed Setup

1. **Virtual Environment**: Created and activated Python 3.13.4 virtual environment
2. **Project Structure**: Created organized directory structure
3. **Dependencies**: Installed core data processing libraries (pandas, numpy, matplotlib, seaborn, tqdm)
4. **Utilities**: Created preprocessing and metrics utilities
5. **Exploration Script**: Data exploration script ready to use
6. **Baseline Model**: Placeholder baseline model structure

## 📁 Project Structure

```
deep-past-initiative-machine-translation/
├── data/              # Processed data files
├── models/            # Model implementations
│   ├── __init__.py
│   └── baseline_model.py
├── notebooks/         # Jupyter notebooks for experimentation
├── submissions/       # Generated submission files
├── utils/             # Utility functions
│   ├── __init__.py
│   ├── metrics.py     # Evaluation metrics (BLEU, chrF++)
│   └── preprocessing.py  # Data cleaning utilities
├── venv/              # Virtual environment
├── explore_data.py    # Data exploration script
├── quick_start.py     # Quick setup verification
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file
```

## 📊 Dataset Overview

- **Training Data**: 1,561 transliterations with English translations
  - Average transliteration: ~427 chars, ~58 words
  - Average translation: ~500 chars, ~91 words
  - Translation/Transliteration ratio: ~1.09

- **Test Data**: 4 examples (dummy data for development)
  - Real test set will have ~4,000 sentences from ~400 documents
  - Sentence-level alignment (vs document-level in training)

- **Additional Resources**:
  - ~8,000 published texts (without translations)
  - ~880 scholarly publications (OCR text, potential translations)
  - ~39,000 word lexicon

## 🚀 Next Steps

### 1. Install Full Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

This will install:
- PyTorch and Transformers (for neural models)
- SacreBLEU (for evaluation)
- NLTK (for text processing)
- Jupyter (for notebooks)

### 2. Explore the Data
```bash
python explore_data.py
```

### 3. Start Model Development

**Option A: Neural Machine Translation (Recommended)**
- Use Transformers library (mBART, mT5, or custom seq2seq)
- Fine-tune on the training data
- Consider data augmentation from published texts

**Option B: Statistical/Classical Approaches**
- Rule-based translation using lexicon
- Statistical MT
- Hybrid approaches

### 4. Extract Additional Training Data

The `publications.csv` file contains ~880 scholarly publications with potential translations. You can:
- Extract translations from OCR text
- Align them with transliterations in `published_texts.csv`
- Convert to English if needed
- Create sentence-level pairs

### 5. Evaluation

Use the `utils/metrics.py` module to evaluate your models:
```python
from utils.metrics import compute_geometric_mean_bleu_chrf

# Evaluate predictions
scores = compute_geometric_mean_bleu_chrf(references, predictions)
print(f"BLEU: {scores['bleu']:.4f}")
print(f"chrF++: {scores['chrf']:.4f}")
print(f"Geometric Mean: {scores['geometric_mean']:.4f}")
```

## 💡 Key Challenges

1. **Low-resource language**: Only ~1,500 training examples
2. **Morphologically complex**: Single Akkadian words encode multiple English words
3. **Domain-specific**: Ancient business/legal texts
4. **Sentence alignment**: Training is document-level, test is sentence-level

## 📚 Resources

- Competition: [Deep Past Challenge on Kaggle](https://www.kaggle.com/competitions/deep-past-challenge)
- OARE Database: https://oare.byu.edu/
- CDLI: https://cdli.earth/
- eBL Dictionary: Links in OA_Lexicon_eBL.csv

## 🔧 Development Tips

1. **Start Simple**: Build a baseline model first
2. **Data Augmentation**: Extract more training data from publications
3. **Cross-validation**: Use document-level splits (by text_id) for validation
4. **Preprocessing**: Clean transliterations carefully (preserve special characters)
5. **Evaluation**: Use the exact metric (Geometric Mean of BLEU and chrF++)

## 📝 Notes

- Test data is dummy data - real test set will be provided during submission
- Submissions must be in CSV format: `id,translation`
- Each translation should be a single sentence
- Evaluation uses micro-averaging across the entire corpus

