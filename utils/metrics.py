"""
Evaluation metrics for the Deep Past Initiative Challenge
Uses Geometric Mean of BLEU and chrF++ scores
"""
import sacrebleu
from typing import List

def compute_geometric_mean_bleu_chrf(references: List[str], predictions: List[str]) -> float:
    """
    Compute the geometric mean of BLEU and chrF++ scores.
    
    Args:
        references: List of reference translations
        predictions: List of predicted translations
    
    Returns:
        Geometric mean score
    """
    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    bleu_score = bleu.score / 100.0  # Convert to 0-1 scale
    
    # Compute chrF++ score
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrf_score = chrf.score / 100.0  # Convert to 0-1 scale
    
    # Geometric mean
    geometric_mean = (bleu_score * chrf_score) ** 0.5
    
    return {
        'bleu': bleu_score,
        'chrf': chrf_score,
        'geometric_mean': geometric_mean,
        'bleu_raw': bleu.score,
        'chrf_raw': chrf.score
    }

def evaluate_submission(reference_file: str, prediction_file: str) -> dict:
    """
    Evaluate a submission file against reference translations.
    
    Args:
        reference_file: Path to CSV file with reference translations
        prediction_file: Path to CSV file with predictions
    
    Returns:
        Dictionary with evaluation metrics
    """
    import pandas as pd
    
    # Load reference and predictions
    ref_df = pd.read_csv(reference_file)
    pred_df = pd.read_csv(prediction_file)
    
    # Merge on id
    merged = ref_df.merge(pred_df, on='id', suffixes=('_ref', '_pred'))
    
    references = merged['translation_ref'].tolist()
    predictions = merged['translation_pred'].tolist()
    
    return compute_geometric_mean_bleu_chrf(references, predictions)

