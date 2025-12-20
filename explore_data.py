"""
Data exploration script for Deep Past Initiative Machine Translation Challenge
"""
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load all CSV files"""
    print("Loading datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    published_texts = pd.read_csv('published_texts.csv')
    lexicon = pd.read_csv('OA_Lexicon_eBL.csv')
    publications = pd.read_csv('publications.csv')
    bibliography = pd.read_csv('bibliography.csv')
    
    return {
        'train': train,
        'test': test,
        'published_texts': published_texts,
        'lexicon': lexicon,
        'publications': publications,
        'bibliography': bibliography
    }

def explore_train_data(train_df):
    """Explore training data statistics"""
    print("\n" + "="*60)
    print("TRAINING DATA EXPLORATION")
    print("="*60)
    
    print(f"\nTotal training examples: {len(train_df)}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Basic statistics
    train_df['transliteration_length'] = train_df['transliteration'].str.len()
    train_df['translation_length'] = train_df['translation'].str.len()
    train_df['transliteration_word_count'] = train_df['transliteration'].str.split().str.len()
    train_df['translation_word_count'] = train_df['translation'].str.split().str.len()
    
    print("\n--- Transliteration Statistics ---")
    print(f"Mean length: {train_df['transliteration_length'].mean():.1f} chars")
    print(f"Median length: {train_df['transliteration_length'].median():.1f} chars")
    print(f"Min length: {train_df['transliteration_length'].min()} chars")
    print(f"Max length: {train_df['transliteration_length'].max()} chars")
    print(f"Mean word count: {train_df['transliteration_word_count'].mean():.1f} words")
    
    print("\n--- Translation Statistics ---")
    print(f"Mean length: {train_df['translation_length'].mean():.1f} chars")
    print(f"Median length: {train_df['translation_length'].median():.1f} chars")
    print(f"Min length: {train_df['translation_length'].min()} chars")
    print(f"Max length: {train_df['translation_length'].max()} chars")
    print(f"Mean word count: {train_df['translation_word_count'].mean():.1f} words")
    
    # Length ratio
    train_df['length_ratio'] = train_df['translation_length'] / train_df['transliteration_length']
    print(f"\n--- Length Ratio (Translation/Transliteration) ---")
    print(f"Mean ratio: {train_df['length_ratio'].mean():.2f}")
    print(f"Median ratio: {train_df['length_ratio'].median():.2f}")
    
    # Sample examples
    print("\n--- Sample Examples ---")
    for idx in [0, 1, 2]:
        print(f"\nExample {idx}:")
        print(f"Transliteration: {train_df.iloc[idx]['transliteration'][:200]}...")
        print(f"Translation: {train_df.iloc[idx]['translation'][:200]}...")
    
    return train_df

def explore_test_data(test_df):
    """Explore test data statistics"""
    print("\n" + "="*60)
    print("TEST DATA EXPLORATION")
    print("="*60)
    
    print(f"\nTotal test examples: {len(test_df)}")
    print(f"Unique documents (text_id): {test_df['text_id'].nunique()}")
    
    # Statistics per document
    doc_stats = test_df.groupby('text_id').agg({
        'id': 'count',
        'transliteration': lambda x: x.str.len().mean()
    }).rename(columns={'id': 'sentence_count', 'transliteration': 'avg_length'})
    
    print(f"\n--- Document Statistics ---")
    print(f"Mean sentences per document: {doc_stats['sentence_count'].mean():.1f}")
    print(f"Median sentences per document: {doc_stats['sentence_count'].median():.1f}")
    print(f"Min sentences per document: {doc_stats['sentence_count'].min()}")
    print(f"Max sentences per document: {doc_stats['sentence_count'].max()}")
    
    # Transliteration statistics
    test_df['transliteration_length'] = test_df['transliteration'].str.len()
    test_df['transliteration_word_count'] = test_df['transliteration'].str.split().str.len()
    
    print(f"\n--- Transliteration Statistics ---")
    print(f"Mean length: {test_df['transliteration_length'].mean():.1f} chars")
    print(f"Median length: {test_df['transliteration_length'].median():.1f} chars")
    print(f"Mean word count: {test_df['transliteration_word_count'].mean():.1f} words")
    
    # Sample examples
    print("\n--- Sample Examples ---")
    for idx in [0, 1, 2]:
        print(f"\nExample {idx}:")
        print(f"Text ID: {test_df.iloc[idx]['text_id']}")
        print(f"Lines: {test_df.iloc[idx]['line_start']}-{test_df.iloc[idx]['line_end']}")
        print(f"Transliteration: {test_df.iloc[idx]['transliteration'][:200]}...")
    
    return test_df

def explore_lexicon(lexicon_df):
    """Explore lexicon data"""
    print("\n" + "="*60)
    print("LEXICON EXPLORATION")
    print("="*60)
    
    print(f"\nTotal lexicon entries: {len(lexicon_df)}")
    print(f"Columns: {list(lexicon_df.columns)}")
    
    # Word type distribution
    if 'type' in lexicon_df.columns:
        type_counts = lexicon_df['type'].value_counts()
        print(f"\n--- Word Type Distribution ---")
        for word_type, count in type_counts.head(10).items():
            print(f"{word_type}: {count}")
    
    # Unique lexemes
    if 'lexeme' in lexicon_df.columns:
        unique_lexemes = lexicon_df['lexeme'].nunique()
        print(f"\nUnique lexemes: {unique_lexemes}")
    
    return lexicon_df

def explore_published_texts(published_df):
    """Explore published texts data"""
    print("\n" + "="*60)
    print("PUBLISHED TEXTS EXPLORATION")
    print("="*60)
    
    print(f"\nTotal published texts: {len(published_df)}")
    
    # Genre distribution
    if 'genre_label' in published_df.columns:
        genre_counts = published_df['genre_label'].value_counts()
        print(f"\n--- Genre Distribution ---")
        for genre, count in genre_counts.head(10).items():
            if pd.notna(genre):
                print(f"{genre}: {count}")
    
    # Transliteration statistics
    if 'transliteration' in published_df.columns:
        published_df['transliteration_length'] = published_df['transliteration'].str.len()
        print(f"\n--- Transliteration Statistics ---")
        print(f"Mean length: {published_df['transliteration_length'].mean():.1f} chars")
        print(f"Median length: {published_df['transliteration_length'].median():.1f} chars")
    
    return published_df

def main():
    """Main exploration function"""
    print("="*60)
    print("DEEP PAST INITIATIVE - DATA EXPLORATION")
    print("="*60)
    
    # Load all data
    data = load_data()
    
    # Explore each dataset
    explore_train_data(data['train'])
    explore_test_data(data['test'])
    explore_lexicon(data['lexicon'])
    explore_published_texts(data['published_texts'])
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

