"""
Preprocessing utilities for Akkadian transliterations and English translations
"""
import re
import pandas as pd
from typing import List, Tuple

def clean_transliteration(text: str) -> str:
    """
    Clean and normalize Akkadian transliteration.
    
    Args:
        text: Raw transliteration text
    
    Returns:
        Cleaned transliteration
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def clean_translation(text: str) -> str:
    """
    Clean and normalize English translation.
    
    Args:
        text: Raw translation text
    
    Returns:
        Cleaned translation
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure single sentence (as per competition requirements)
    # Remove multiple periods/spaces
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences (for English translations).
    
    Args:
        text: Text to split
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting on periods, exclamation, question marks
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def create_sentence_pairs(transliterations: List[str], translations: List[str]) -> List[Tuple[str, str]]:
    """
    Create sentence-level aligned pairs from document-level pairs.
    
    Args:
        transliterations: List of transliteration texts
        translations: List of translation texts
    
    Returns:
        List of (transliteration, translation) sentence pairs
    """
    pairs = []
    for trans, trans_en in zip(transliterations, translations):
        # For now, treat each document as a single sentence pair
        # More sophisticated alignment can be added later
        trans_clean = clean_transliteration(trans)
        trans_en_clean = clean_translation(trans_en)
        
        if trans_clean and trans_en_clean:
            pairs.append((trans_clean, trans_en_clean))
    
    return pairs

def prepare_training_data(train_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Prepare training data for model training.
    
    Args:
        train_file: Path to train.csv
        output_file: Optional path to save processed data
    
    Returns:
        DataFrame with cleaned data
    """
    df = pd.read_csv(train_file)
    
    # Clean transliterations and translations
    df['transliteration_clean'] = df['transliteration'].apply(clean_transliteration)
    df['translation_clean'] = df['translation'].apply(clean_translation)
    
    # Remove empty pairs
    df = df[
        (df['transliteration_clean'].str.len() > 0) & 
        (df['translation_clean'].str.len() > 0)
    ].copy()
    
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df

