"""
Quick start script to get familiar with the dataset and create initial setup
"""
import pandas as pd
import os
from utils.preprocessing import prepare_training_data, clean_transliteration, clean_translation

def main():
    print("="*60)
    print("DEEP PAST INITIATIVE - QUICK START")
    print("="*60)
    
    # Check if data files exist
    required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Please ensure all data files are in the current directory.")
        return
    
    print("\n✓ All required data files found")
    
    # Load and explore training data
    print("\n" + "-"*60)
    print("Loading training data...")
    train_df = pd.read_csv('train.csv')
    print(f"✓ Loaded {len(train_df)} training examples")
    
    # Prepare cleaned training data
    print("\n" + "-"*60)
    print("Preparing cleaned training data...")
    train_clean = prepare_training_data('train.csv', 'data/train_clean.csv')
    print(f"✓ Cleaned data: {len(train_clean)} examples")
    print(f"  Average transliteration length: {train_clean['transliteration_clean'].str.len().mean():.1f} chars")
    print(f"  Average translation length: {train_clean['translation_clean'].str.len().mean():.1f} chars")
    
    # Load test data
    print("\n" + "-"*60)
    print("Loading test data...")
    test_df = pd.read_csv('test.csv')
    print(f"✓ Loaded {len(test_df)} test examples")
    print(f"  Unique documents: {test_df['text_id'].nunique()}")
    
    # Show sample examples
    print("\n" + "-"*60)
    print("Sample Training Examples:")
    print("-"*60)
    for idx in [0, 1, 2]:
        row = train_clean.iloc[idx]
        print(f"\nExample {idx + 1}:")
        print(f"Transliteration: {row['transliteration_clean'][:150]}...")
        print(f"Translation: {row['translation_clean'][:150]}...")
    
    print("\n" + "-"*60)
    print("Sample Test Examples:")
    print("-"*60)
    for idx in [0, 1]:
        row = test_df.iloc[idx]
        print(f"\nTest {idx + 1} (ID: {row['id']}):")
        print(f"Text ID: {row['text_id']}")
        print(f"Lines: {row['line_start']}-{row['line_end']}")
        print(f"Transliteration: {row['transliteration'][:150]}...")
    
    # Create data directory structure
    os.makedirs('data', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Explore the data: python explore_data.py")
    print("2. Start building models in the models/ directory")
    print("3. Use notebooks/ for experimentation")
    print("4. Check utils/ for preprocessing and evaluation utilities")

if __name__ == "__main__":
    main()

