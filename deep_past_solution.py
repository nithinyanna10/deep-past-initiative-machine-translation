#!/usr/bin/env python3
"""
Deep Past Initiative - End-to-End Machine Translation Solution
Single-file implementation for Old Assyrian to English translation
"""

import argparse
import logging
import os
import platform
import random
import re
import sys
from collections import Counter
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu import corpus_bleu, corpus_chrf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Suppress Metal/GPU errors on macOS
if platform.system() == "Darwin":
    # Force CPU usage to avoid Metal memory issues
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    # Suppress Metal error messages
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    # Suppress Metal backend errors
    os.environ['PYTORCH_MPS_LOG_LEVEL'] = '0'

@contextmanager
def suppress_metal_errors():
    """Context manager to suppress Metal GPU error messages on macOS"""
    if platform.system() == "Darwin":
        original_stderr = sys.stderr
        filtered_stderr = StringIO()
        
        class MetalFilter:
            def __init__(self, original):
                self.original = original
                self.buffer = []
            
            def write(self, text):
                # Filter out Metal-related error messages
                if any(keyword in text for keyword in [
                    'Metal', 'AGXG', 'command buffer', 'Insufficient Memory',
                    'kIOGPUCommandBufferCallbackErrorOutOfMemory',
                    'Metal Performance Shaders'
                ]):
                    return
                self.original.write(text)
            
            def flush(self):
                self.original.flush()
        
        sys.stderr = MetalFilter(original_stderr)
        try:
            yield
        finally:
            sys.stderr = original_stderr
    else:
        yield

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_translit(text: str, strict: bool = False) -> str:
    """
    Clean and normalize Akkadian transliteration.
    
    Args:
        text: Raw transliteration text
        strict: If True, remove editorial uncertainty markers
    
    Returns:
        Cleaned transliteration
    """
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove obvious noise (repeated weird patterns)
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to triple
    text = re.sub(r'\s*-\s*-\s*', ' - ', text)  # Multiple hyphens
    
    if strict:
        # Remove editorial uncertainty markers (conservative)
        # Only remove standalone bracketed uncertainty like [gap], [broken], etc.
        text = re.sub(r'\[(?:gap|broken|missing|illegible|unclear)\s*\]', '', text, flags=re.IGNORECASE)
        # Remove empty brackets
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\(\s*\)', '', text)
    
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def chunk_document(text: str, max_chars: int = 240, overlap: int = 30) -> List[str]:
    """
    Split document into chunks with priority on natural boundaries.
    
    Args:
        text: Document text to chunk
        max_chars: Maximum characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of non-empty chunks in original order
    """
    if not text or len(text) <= max_chars:
        return [text] if text.strip() else []
    
    chunks = []
    text = text.strip()
    
    # Priority 1: Split on newlines first
    if '\n' in text:
        lines = text.split('\n')
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if len(current_chunk) + len(line) + 1 <= max_chars:
                current_chunk = f"{current_chunk} {line}".strip() if current_chunk else line
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # If chunks are still too long, split further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                # Priority 2: Split on sentence boundaries
                final_chunks.extend(_split_on_punctuation(chunk, max_chars, overlap))
        
        return [c for c in final_chunks if c.strip()]
    
    # Priority 2: Split on punctuation boundaries
    return _split_on_punctuation(text, max_chars, overlap)


def _split_on_punctuation(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split text on punctuation boundaries, then by max_chars with overlap"""
    # Split on sentence boundaries
    sentences = re.split(r'([.!?;:]+)', text)
    # Recombine punctuation with preceding text
    combined = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined.append(sentences[i] + sentences[i + 1])
        else:
            combined.append(sentences[i])
    
    chunks = []
    current_chunk = ""
    
    for sent in combined:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(current_chunk) + len(sent) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sent}".strip() if current_chunk else sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # If still too long, split by max_chars with overlap
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split with overlap
            start = 0
            while start < len(chunk):
                end = start + max_chars
                if end >= len(chunk):
                    final_chunks.append(chunk[start:].strip())
                    break
                else:
                    # Try to break at word boundary
                    break_point = chunk.rfind(' ', start, end)
                    if break_point > start:
                        end = break_point
                    final_chunks.append(chunk[start:end].strip())
                    start = max(start + 1, end - overlap)
    
    return [c for c in final_chunks if c.strip()]


def align_chunks(src_chunks: List[str], tgt_chunks: List[str]) -> List[Tuple[str, str]]:
    """
    Align source and target chunks using monotonic greedy alignment with robustness improvements.
    
    Args:
        src_chunks: Source transliteration chunks
        tgt_chunks: Target translation chunks
    
    Returns:
        List of (src_chunk, tgt_chunk) pairs
    """
    if not src_chunks or not tgt_chunks:
        return []
    
    # Pre-process: merge tiny target sentences (<20 chars) if counts differ greatly
    if abs(len(src_chunks) - len(tgt_chunks)) > max(len(src_chunks), len(tgt_chunks)) * 0.5:
        merged_tgt = []
        i = 0
        while i < len(tgt_chunks):
            current = tgt_chunks[i]
            # Merge tiny chunks with neighbors
            while i + 1 < len(tgt_chunks) and len(current) < 20:
                current = f"{current} {tgt_chunks[i + 1]}"
                i += 1
            merged_tgt.append(current)
            i += 1
        tgt_chunks = merged_tgt
    
    # If counts match, pair in order
    if len(src_chunks) == len(tgt_chunks):
        aligned = list(zip(src_chunks, tgt_chunks))
        # Check for extreme ratios
        ratios = [len(t) / len(s) if len(s) > 0 else 0 for s, t in aligned]
        if any(r > 50 or r < 0.02 for r in ratios if r > 0):
            # Fallback to fixed window alignment
            return _align_by_fixed_windows(src_chunks, tgt_chunks)
        return aligned
    
    # Monotonic greedy alignment
    aligned = []
    src_idx = 0
    tgt_idx = 0
    
    # Calculate length ratios for guidance
    src_total = sum(len(c) for c in src_chunks)
    tgt_total = sum(len(c) for c in tgt_chunks)
    length_ratio = tgt_total / src_total if src_total > 0 else 1.0
    
    while src_idx < len(src_chunks) and tgt_idx < len(tgt_chunks):
        src_chunk = src_chunks[src_idx]
        tgt_chunk = tgt_chunks[tgt_idx]
        
        # Check if lengths roughly match
        src_len = len(src_chunk)
        expected_tgt_len = src_len * length_ratio
        
        if abs(len(tgt_chunk) - expected_tgt_len) < expected_tgt_len * 0.5 or tgt_idx == len(tgt_chunks) - 1:
            # Good match or last target chunk
            aligned.append((src_chunk, tgt_chunk))
            src_idx += 1
            tgt_idx += 1
        elif len(tgt_chunk) < expected_tgt_len * 0.7:
            # Target chunk too short, try merging with next
            if tgt_idx + 1 < len(tgt_chunks):
                merged_tgt = f"{tgt_chunk} {tgt_chunks[tgt_idx + 1]}"
                aligned.append((src_chunk, merged_tgt))
                src_idx += 1
                tgt_idx += 2
            else:
                aligned.append((src_chunk, tgt_chunk))
                src_idx += 1
                tgt_idx += 1
        else:
            # Target chunk too long, use as is
            aligned.append((src_chunk, tgt_chunk))
            src_idx += 1
            tgt_idx += 1
    
    # Handle remaining chunks
    while src_idx < len(src_chunks) and aligned:
        # Attach to last target
        last_src, last_tgt = aligned[-1]
        aligned[-1] = (f"{last_src} {src_chunks[src_idx]}", last_tgt)
        src_idx += 1
    
    while tgt_idx < len(tgt_chunks) and aligned:
        # Attach to last target
        last_src, last_tgt = aligned[-1]
        aligned[-1] = (last_src, f"{last_tgt} {tgt_chunks[tgt_idx]}")
        tgt_idx += 1
    
    # Validate alignment quality
    ratios = [len(t) / len(s) if len(s) > 0 else 0 for s, t in aligned]
    if any(r > 50 or r < 0.02 for r in ratios if r > 0):
        # Fallback to fixed window alignment
        return _align_by_fixed_windows(src_chunks, tgt_chunks)
    
    return aligned


def _align_by_fixed_windows(src_chunks: List[str], tgt_chunks: List[str]) -> List[Tuple[str, str]]:
    """Fallback alignment by grouping into fixed windows by character length"""
    src_total = sum(len(c) for c in src_chunks)
    tgt_total = sum(len(c) for c in tgt_chunks)
    
    if src_total == 0 or tgt_total == 0:
        return list(zip(src_chunks, tgt_chunks)) if len(src_chunks) == len(tgt_chunks) else []
    
    # Create windows based on cumulative character length
    aligned = []
    src_chars = 0
    tgt_chars = 0
    src_window = []
    tgt_window = []
    src_idx = 0
    tgt_idx = 0
    
    target_ratio = tgt_total / src_total
    
    while src_idx < len(src_chunks) or tgt_idx < len(tgt_chunks):
        # Fill source window
        if src_idx < len(src_chunks):
            src_window.append(src_chunks[src_idx])
            src_chars += len(src_chunks[src_idx])
            src_idx += 1
        
        # Fill target window to match ratio
        target_chars = src_chars * target_ratio
        while tgt_idx < len(tgt_chunks) and tgt_chars < target_chars * 1.1:
            tgt_window.append(tgt_chunks[tgt_idx])
            tgt_chars += len(tgt_chunks[tgt_idx])
            tgt_idx += 1
        
        # Create pair
        if src_window and tgt_window:
            aligned.append((' '.join(src_window), ' '.join(tgt_window)))
            src_window = []
            tgt_window = []
            src_chars = 0
            tgt_chars = 0
    
    # Handle remaining
    if src_window or tgt_window:
        aligned.append((' '.join(src_window) if src_window else '', ' '.join(tgt_window) if tgt_window else ''))
    
    return aligned


def anti_loop_generate(
    model,
    inputs,
    tokenizer,
    max_new_tokens: int = 80,
    min_new_tokens: int = 4,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 4,
    repetition_penalty: float = 1.2,
    length_penalty: float = 0.8,
    early_stopping: bool = True,
    max_length: Optional[int] = None,
):
    """
    Generate with anti-looping parameters to prevent repetition.
    
    Args:
        model: The model to generate from
        inputs: Tokenized inputs
        tokenizer: Tokenizer for decoding
        max_new_tokens: Maximum new tokens to generate
        min_new_tokens: Minimum new tokens to generate
        num_beams: Beam search width
        no_repeat_ngram_size: N-gram size to prevent repetition
        repetition_penalty: Penalty for repetition
        length_penalty: Length penalty for beam search
        early_stopping: Whether to stop early
        max_length: Maximum total length (if None, uses max_new_tokens)
    
    Returns:
        Generated outputs
    """
    generate_kwargs = {
        'num_beams': num_beams,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'repetition_penalty': repetition_penalty,
        'length_penalty': length_penalty,
        'early_stopping': early_stopping,
    }
    
    if max_length is not None:
        generate_kwargs['max_length'] = max_length
    else:
        generate_kwargs['max_new_tokens'] = max_new_tokens
        if min_new_tokens > 0:
            generate_kwargs['min_length'] = inputs['input_ids'].shape[1] + min_new_tokens
    
    return model.generate(**inputs, **generate_kwargs)


def parse_line_key(line_str: str) -> Tuple[int, int]:
    """
    Parse line_start/line_end into sortable key.
    
    Args:
        line_str: Line string like "1", "1'", "1''", etc.
    
    Returns:
        (base_number, prime_count) for sorting
    """
    if pd.isna(line_str):
        return (0, 0)
    
    line_str = str(line_str).strip()
    
    # Extract leading number
    match = re.match(r'^(\d+)', line_str)
    if match:
        base_num = int(match.group(1))
    else:
        base_num = 0
    
    # Count prime characters
    prime_count = len(re.findall(r'[\'ʹ′]', line_str))
    
    return (base_num, prime_count)


def build_dataset(
    train_file: str,
    max_source_len: int = 512,
    max_target_len: int = 256,
    val_ratio: float = 0.1,
    use_context: bool = True,
    strict_clean: bool = False,
    max_train_pairs: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dict]:
    """
    Build training dataset from document-level pairs.
    
    Returns:
        train_dataset, val_dataset, stats_dict
    """
    logger.info(f"Loading training data from {train_file}")
    df = pd.read_csv(train_file)
    
    logger.info(f"Loaded {len(df)} documents")
    
    # Build chunk-level pairs
    all_pairs = []
    doc_chunks = {}  # Store chunks per doc for context
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
        doc_id = row.get('oare_id', f'doc_{idx}')
        translit = clean_translit(row['transliteration'], strict=strict_clean)
        translation = clean_translit(row['translation'], strict=strict_clean)
        
        if not translit or not translation:
            continue
        
        # Chunk transliteration
        src_chunks = chunk_document(translit, max_chars=max_source_len // 2)
        
        # Chunk translation (split on sentence boundaries)
        tgt_sentences = re.split(r'([.!?;]+)', translation)
        tgt_combined = []
        for i in range(0, len(tgt_sentences) - 1, 2):
            if i + 1 < len(tgt_sentences):
                tgt_combined.append(tgt_sentences[i] + tgt_sentences[i + 1])
            else:
                tgt_combined.append(tgt_sentences[i])
        tgt_chunks = [s.strip() for s in tgt_combined if s.strip()]
        
        # Align chunks
        aligned = align_chunks(src_chunks, tgt_chunks)
        
        # Store chunks per doc
        doc_chunks[doc_id] = {
            'src_chunks': src_chunks,
            'tgt_chunks': tgt_chunks,
        }
        
        # Create pairs
        for chunk_idx, (src, tgt) in enumerate(aligned):
            if src.strip() and tgt.strip():
                all_pairs.append({
                    'doc_id': doc_id,
                    'chunk_idx': chunk_idx,
                    'src': src.strip(),
                    'tgt': tgt.strip(),
                })
    
    logger.info(f"Created {len(all_pairs)} chunk-level pairs from {len(df)} documents")
    
    # Group split by doc_id
    unique_docs = df['oare_id'].unique() if 'oare_id' in df.columns else list(range(len(df)))
    random.shuffle(unique_docs)
    split_idx = int(len(unique_docs) * (1 - val_ratio))
    train_docs = set(unique_docs[:split_idx])
    val_docs = set(unique_docs[split_idx:])
    
    train_pairs = [p for p in all_pairs if p['doc_id'] in train_docs]
    val_pairs = [p for p in all_pairs if p['doc_id'] in val_docs]
    
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # Dataset sanity diagnostics
    logger.info("\n" + "="*60)
    logger.info("DATASET SANITY DIAGNOSTICS")
    logger.info("="*60)
    
    # Print 30 random pairs
    logger.info("\n30 Random (src, tgt) pairs:")
    sample_pairs = random.sample(train_pairs, min(30, len(train_pairs)))
    for i, pair in enumerate(sample_pairs, 1):
        logger.info(f"\nPair {i}:")
        logger.info(f"  Src: {pair['src'][:200]}")
        logger.info(f"  Tgt: {pair['tgt'][:200]}")
    
    # Percentage containing "silver"
    silver_count = sum(1 for p in train_pairs if 'silver' in p['tgt'].lower())
    silver_pct = (silver_count / len(train_pairs) * 100) if train_pairs else 0
    logger.info(f"\nPercentage of targets containing 'silver': {silver_pct:.2f}% ({silver_count}/{len(train_pairs)})")
    
    # Top-20 most frequent words in targets
    all_words = []
    for pair in train_pairs:
        all_words.extend(pair['tgt'].split())
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(20)
    logger.info("\nTop-20 most frequent words in targets:")
    for word, count in top_words:
        logger.info(f"  '{word}': {count}")
    
    # Length distribution
    src_lens = [len(p['src']) for p in train_pairs]
    tgt_lens = [len(p['tgt']) for p in train_pairs]
    ratios = [t / s if s > 0 else 0 for s, t in zip(src_lens, tgt_lens)]
    
    logger.info("\nSource length (chars) distribution:")
    logger.info(f"  Min: {min(src_lens)}, Median: {np.median(src_lens):.1f}, P95: {np.percentile(src_lens, 95):.1f}, Max: {max(src_lens)}")
    
    logger.info("\nTarget length (chars) distribution:")
    logger.info(f"  Min: {min(tgt_lens)}, Median: {np.median(tgt_lens):.1f}, P95: {np.percentile(tgt_lens, 95):.1f}, Max: {max(tgt_lens)}")
    
    logger.info("\nSource/Target length ratio distribution:")
    valid_ratios = [r for r in ratios if r > 0]
    if valid_ratios:
        logger.info(f"  Min: {min(valid_ratios):.4f}, Median: {np.median(valid_ratios):.4f}, P95: {np.percentile(valid_ratios, 95):.4f}, Max: {max(valid_ratios):.4f}")
    
    # Warnings for problematic pairs
    empty_tgt_count = sum(1 for p in train_pairs if not p['tgt'].strip())
    if empty_tgt_count > 0:
        logger.warning(f"\n⚠️  Found {empty_tgt_count} pairs with empty targets!")
        for p in train_pairs[:5]:  # Show first 5 examples
            if not p['tgt'].strip():
                logger.warning(f"  Example: src='{p['src'][:100]}...', tgt='{p['tgt']}'")
    
    extreme_ratios = [(i, r) for i, r in enumerate(ratios) if r > 50 or (r < 0.02 and r > 0)]
    if extreme_ratios:
        logger.warning(f"\n⚠️  Found {len(extreme_ratios)} pairs with extreme length ratios (>50 or <0.02)!")
        for idx, ratio in extreme_ratios[:5]:  # Show first 5 examples
            pair = train_pairs[idx]
            logger.warning(f"  Example {idx}: ratio={ratio:.4f}, src_len={len(pair['src'])}, tgt_len={len(pair['tgt'])}")
            logger.warning(f"    Src: {pair['src'][:150]}...")
            logger.warning(f"    Tgt: {pair['tgt'][:150]}...")
    
    logger.info("="*60 + "\n")
    
    # Cap training pairs if specified
    if max_train_pairs and len(train_pairs) > max_train_pairs:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:max_train_pairs]
        logger.info(f"Capped training pairs to {max_train_pairs}")
    
    # Add context if requested
    if use_context:
        train_pairs = _add_context(train_pairs, doc_chunks)
        val_pairs = _add_context(val_pairs, doc_chunks)
    
    # Convert to datasets
    train_df = pd.DataFrame(train_pairs)
    val_df = pd.DataFrame(val_pairs)
    
    # Stats
    stats = {
        'num_docs': len(df),
        'num_train_pairs': len(train_pairs),
        'num_val_pairs': len(val_pairs),
        'avg_src_len': train_df['src'].str.len().mean() if len(train_df) > 0 else 0,
        'avg_tgt_len': train_df['tgt'].str.len().mean() if len(train_df) > 0 else 0,
    }
    
    # Quality checks
    assert len(train_pairs) > 0, "No training pairs created!"
    assert len(val_pairs) > 0, "No validation pairs created!"
    assert all(p['src'].strip() and p['tgt'].strip() for p in train_pairs), "Empty pairs found!"
    
    logger.info("Dataset stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Show examples
    logger.info("\nSample training examples:")
    for i, pair in enumerate(random.sample(train_pairs, min(3, len(train_pairs)))):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Source: {pair['src'][:150]}...")
        logger.info(f"  Target: {pair['tgt'][:150]}...")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset, stats


def _add_context(pairs: List[Dict], doc_chunks: Dict) -> List[Dict]:
    """Add context windows to pairs"""
    contexted = []
    
    for pair in pairs:
        doc_id = pair['doc_id']
        chunk_idx = pair['chunk_idx']
        
        if doc_id in doc_chunks:
            src_chunks = doc_chunks[doc_id]['src_chunks']
            prev_chunk = src_chunks[chunk_idx - 1] if chunk_idx > 0 else ""
            next_chunk = src_chunks[chunk_idx + 1] if chunk_idx + 1 < len(src_chunks) else ""
            cur_chunk = pair['src']
            
            if prev_chunk or next_chunk:
                contexted_src = f"<prev> {prev_chunk} </prev> <cur> {cur_chunk} </cur> <next> {next_chunk} </next>"
            else:
                contexted_src = cur_chunk
        else:
            contexted_src = pair['src']
        
        contexted.append({
            'doc_id': pair['doc_id'],
            'chunk_idx': pair['chunk_idx'],
            'src': contexted_src,
            'tgt': pair['tgt'],
        })
    
    return contexted


def compute_metrics(eval_pred, tokenizer):
    """Compute BLEU and chrF++ metrics"""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU
    bleu = corpus_bleu(decoded_preds, [decoded_labels])
    bleu_score = bleu.score
    
    # Compute chrF++ (word_order=2)
    chrf = corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
    chrf_score = chrf.score
    
    # Geometric mean
    final_score = np.sqrt(bleu_score * chrf_score)
    
    return {
        'bleu': bleu_score,
        'chrf++': chrf_score,
        'final': final_score,
    }


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_name: str,
    out_dir: str,
    max_source_len: int = 512,
    max_target_len: int = 256,
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 8,
    grad_accum: int = 2,
    num_beams: int = 5,
    save_best: bool = True,
    smoke_mode: bool = False,
):
    """Train the translation model"""
    logger.info(f"Loading model: {model_name}")
    
    # Force CPU on macOS to avoid Metal memory issues
    if platform.system() == "Darwin":
        device = "cpu"
        logger.info("macOS detected: Using CPU to avoid Metal memory issues")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize datasets
    def preprocess_function(examples):
        inputs = [ex for ex in examples['src']]
        targets = [ex for ex in examples['tgt']]
        
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_len,
            truncation=True,
            padding='max_length',
        )
        
        labels = tokenizer(
            targets,
            max_length=max_target_len,
            truncation=True,
            padding='max_length',
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Training arguments
    # Use fp16 only for CUDA, not for CPU/Metal
    use_fp16 = torch.cuda.is_available() and platform.system() != "Darwin"
    
    # Reduce batch size for smoke mode or macOS
    effective_batch_size = batch_size
    if smoke_mode:
        effective_batch_size = min(batch_size, 2)
    elif platform.system() == "Darwin":
        effective_batch_size = min(batch_size, 4)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=save_best,
        metric_for_best_model="final",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=max_target_len,
        generation_num_beams=num_beams,
        generation_max_new_tokens=80,  # Default, can be overridden
        logging_steps=50,
        report_to=None,
        dataloader_pin_memory=False,  # Disable pinning on macOS
    )
    
    if smoke_mode:
        training_args.max_steps = 50
        training_args.eval_strategy = "steps"
        training_args.eval_steps = 25
        training_args.save_steps = 25
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Metrics function
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if not smoke_mode else [],
    )
    
    logger.info("Starting training...")
    with suppress_metal_errors():
        trainer.train()
    logger.info("Training complete!")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    logger.info("Validation Results:")
    logger.info(f"  BLEU: {eval_results.get('eval_bleu', 0):.4f}")
    logger.info(f"  chrF++: {eval_results.get('eval_chrf++', 0):.4f}")
    logger.info(f"  FINAL: {eval_results.get('eval_final', 0):.4f}")
    
    # Save final model
    trainer.save_model(os.path.join(out_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final_model"))
    
    return model, tokenizer


def run_inference(
    test_file: str,
    model,
    tokenizer,
    out_dir: str,
    max_source_len: int = 512,
    max_target_len: int = 256,
    num_beams: int = 4,
    max_new_tokens: int = 80,
    min_new_tokens: int = 4,
    no_repeat_ngram_size: int = 4,
    repetition_penalty: float = 1.2,
    length_penalty: float = 0.8,
    use_context: bool = True,
    smoke_mode: bool = False,
):
    """Run inference on test set"""
    logger.info(f"Loading test data from {test_file}")
    test_df = pd.read_csv(test_file)
    
    if smoke_mode:
        test_df = test_df.head(50)
        logger.info(f"Smoke mode: using first 50 test examples")
    
    logger.info(f"Loaded {len(test_df)} test examples")
    
    # Force CPU on macOS to avoid Metal memory issues
    if platform.system() == "Darwin":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    # Group by text_id for context
    if use_context and 'text_id' in test_df.columns:
        # Sort by text_id, then by line numbers (convert tuple to sortable value)
        test_df = test_df.copy()
        test_df['_line_start_base'] = test_df['line_start'].apply(lambda x: parse_line_key(x)[0] if pd.notna(x) else 0)
        test_df['_line_start_prime'] = test_df['line_start'].apply(lambda x: parse_line_key(x)[1] if pd.notna(x) else 0)
        test_df = test_df.sort_values(['text_id', '_line_start_base', '_line_start_prime'])
        test_df = test_df.drop(columns=['_line_start_base', '_line_start_prime'])
        
        # Build context
        predictions = []
        for text_id, group in tqdm(test_df.groupby('text_id'), desc="Processing documents"):
            group_chunks = []
            for idx, row in group.iterrows():
                translit = clean_translit(row['transliteration'])
                group_chunks.append(translit)
            
            # Generate with context
            for i, (idx, row) in enumerate(group.iterrows()):
                translit = clean_translit(row['transliteration'])
                
                if use_context and len(group_chunks) > 1:
                    prev_chunk = group_chunks[i - 1] if i > 0 else ""
                    next_chunk = group_chunks[i + 1] if i + 1 < len(group_chunks) else ""
                    contexted_src = f"<prev> {prev_chunk} </prev> <cur> {translit} </cur> <next> {next_chunk} </next>"
                else:
                    contexted_src = translit
                
                # Tokenize and generate
                inputs = tokenizer(
                    contexted_src,
                    max_length=max_source_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                ).to(device)
                
                with torch.no_grad(), suppress_metal_errors():
                    outputs = anti_loop_generate(
                        model=model,
                        inputs=inputs,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        early_stopping=True,
                        max_length=max_target_len,
                    )
                
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Postprocess
                pred_text = re.sub(r'\s+', ' ', pred_text).strip()
                predictions.append(pred_text)
    else:
        # No context
        predictions = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
            translit = clean_translit(row['transliteration'])
            
            inputs = tokenizer(
                translit,
                max_length=max_source_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            ).to(device)
            
            with torch.no_grad(), suppress_metal_errors():
                outputs = model.generate(
                    **inputs,
                    max_length=max_target_len,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_text = re.sub(r'\s+', ' ', pred_text).strip()
            predictions.append(pred_text)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'translation': predictions,
    })
    
    # Quality checks
    assert len(submission_df) == len(test_df), "Mismatch in submission length!"
    assert all(submission_df['translation'].str.len() > 0), "Empty predictions found!"
    
    # Show samples
    logger.info("\nSample predictions:")
    for i in range(min(5, len(submission_df))):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Source: {test_df.iloc[i]['transliteration'][:150]}...")
        logger.info(f"  Prediction: {submission_df.iloc[i]['translation'][:200]}...")
    
    # Save submission
    submission_file = os.path.join(out_dir, "submission.csv")
    submission_df.to_csv(submission_file, index=False)
    logger.info(f"\nSaved submission to {submission_file}")
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Deep Past Initiative MT Solution")
    
    # Required arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train.csv and test.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for models and submissions")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/byt5-small", help="HuggingFace model name")
    parser.add_argument("--mode", type=str, choices=["smoke", "train", "predict", "train_predict"], 
                       default="train_predict", help="Execution mode")
    
    # Data arguments
    parser.add_argument("--max_source_len", type=int, default=512, help="Max source sequence length")
    parser.add_argument("--max_target_len", type=int, default=256, help="Max target sequence length")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--use_context", type=int, default=1, choices=[0, 1], help="Use context windows")
    parser.add_argument("--disable_context", type=int, default=1, choices=[0, 1], help="Force disable context (overrides use_context)")
    parser.add_argument("--strict_clean", type=int, default=0, choices=[0, 1], help="Strict cleaning mode")
    parser.add_argument("--max_train_pairs", type=int, default=None, help="Cap on training pairs")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam search width")
    parser.add_argument("--save_best", type=int, default=1, choices=[0, 1], help="Save best model")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Maximum new tokens to generate")
    parser.add_argument("--min_new_tokens", type=int, default=4, help="Minimum new tokens to generate")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4, help="N-gram size to prevent repetition")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repetition")
    parser.add_argument("--length_penalty", type=float, default=0.8, help="Length penalty for beam search")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Paths
    train_file = os.path.join(args.data_dir, "train.csv")
    test_file = os.path.join(args.data_dir, "test.csv")
    
    # Validate files
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if args.mode in ["predict", "train_predict"] and not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Handle context flag: disable_context overrides use_context
    if args.disable_context == 1:
        use_context = False
    else:
        use_context = args.use_context == 1
    
    strict_clean = args.strict_clean == 1
    save_best = args.save_best == 1
    smoke_mode = args.mode == "smoke"
    
    # Build dataset
    if args.mode in ["smoke", "train", "train_predict"]:
        train_dataset, val_dataset, stats = build_dataset(
            train_file=train_file,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            val_ratio=args.val_ratio,
            use_context=use_context,
            strict_clean=strict_clean,
            max_train_pairs=args.max_train_pairs,
        )
    
    # Train model
    if args.mode in ["smoke", "train", "train_predict"]:
        model, tokenizer = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=args.model_name,
            out_dir=args.out_dir,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_beams=args.num_beams,
            save_best=save_best,
            smoke_mode=smoke_mode,
        )
    else:
        # Load model for prediction only
        model_path = os.path.join(args.out_dir, "final_model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        logger.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Run inference
    if args.mode in ["smoke", "predict", "train_predict"]:
        run_inference(
            test_file=test_file,
            model=model,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            use_context=use_context,
            smoke_mode=smoke_mode,
        )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

