import pandas as pd
import numpy as np
from typing import Tuple, List
import sys

def load_gesture_data(filepath: str) -> pd.DataFrame:
    """
    Load gesture emotion data from CSV file.
    Expected columns: valence, arousal, dominance
    """
    try:
        df = pd.read_csv(filepath)
        required_cols = ['valence', 'arousal', 'dominance']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def compute_coverage(df: pd.DataFrame, n_bins: int = 5) -> float:
    """
    Compute the coverage of gestures in the VAD (Valence-Arousal-Dominance) space.
    
    Coverage measures how well the gestures span the emotional space.
    Higher coverage means gestures are more diverse and cover more of the emotional spectrum.
    
    Args:
        df: DataFrame with valence, arousal, dominance columns
        n_bins: Number of bins to divide each dimension into
    
    Returns:
        Coverage score (0-1), where 1 means complete coverage
    """
    # Normalize values to 0-1 range if not already
    vad_data = df[['valence', 'arousal', 'dominance']].copy()
    
    for col in vad_data.columns:
        min_val = vad_data[col].min()
        max_val = vad_data[col].max()
        if max_val > min_val:
            vad_data[col] = (vad_data[col] - min_val) / (max_val - min_val)
    
    # Discretize into bins
    binned_data = (vad_data * (n_bins - 0.001)).astype(int)
    
    # Count unique cells occupied in 3D space
    unique_cells = binned_data.drop_duplicates().shape[0]
    total_cells = n_bins ** 3
    
    coverage = unique_cells / total_cells
    
    return coverage

def compute_agreement(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Compute agreement scores for gestures based on clustering in VAD space.
    
    Agreement measures how consistent or distinct each gesture is in emotional space.
    Higher agreement means gestures are more clustered (consistent emotional expression).
    
    Args:
        df: DataFrame with valence, arousal, dominance columns
        threshold: Distance threshold for considering gestures as similar
    
    Returns:
        DataFrame with agreement scores for each gesture
    """
    vad_cols = ['valence', 'arousal', 'dominance']
    vad_data = df[vad_cols].values
    
    # Normalize the data
    mean = vad_data.mean(axis=0)
    std = vad_data.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    vad_normalized = (vad_data - mean) / std
    
    # Compute pairwise Euclidean distances
    n_gestures = len(df)
    distances = np.zeros((n_gestures, n_gestures))
    
    for i in range(n_gestures):
        for j in range(n_gestures):
            distances[i, j] = np.linalg.norm(vad_normalized[i] - vad_normalized[j])
    
    # Compute agreement: proportion of gestures within threshold distance
    agreement_scores = []
    for i in range(n_gestures):
        similar_gestures = np.sum(distances[i] <= threshold) - 1  # Exclude self
        agreement = similar_gestures / (n_gestures - 1) if n_gestures > 1 else 0
        agreement_scores.append(agreement)
    
    result_df = df.copy()
    result_df['agreement_score'] = agreement_scores
    
    return result_df

def rank_gestures(df: pd.DataFrame, coverage: float) -> pd.DataFrame:
    """
    Rank gestures based on agreement scores and overall coverage.
    
    Args:
        df: DataFrame with agreement scores
        coverage: Overall coverage score
    
    Returns:
        DataFrame sorted by ranking
    """
    # Add coverage info
    df_ranked = df.copy()
    df_ranked['coverage'] = coverage
    
    # Rank by agreement score (higher is better for consistency)
    df_ranked['rank'] = df_ranked['agreement_score'].rank(ascending=False, method='dense').astype(int)
    
    # Sort by rank
    df_ranked = df_ranked.sort_values('rank')
    
    return df_ranked

def main():
    if len(sys.argv) < 2:
        print("Usage: python gesture_analysis.py <csv_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load data
    print(f"Loading data from {filepath}...")
    df = load_gesture_data(filepath)
    print(f"Loaded {len(df)} gestures\n")
    
    # Compute coverage
    coverage = compute_coverage(df)
    print(f"Coverage Score: {coverage:.4f}")
    print(f"  (Measures diversity of gestures in emotional space, range: 0-1)\n")
    
    # Compute agreement
    df_with_agreement = compute_agreement(df)
    print("Agreement Scores computed")
    print(f"  (Measures consistency of gestures, higher = more clustered)\n")
    
    # Rank gestures
    df_ranked = rank_gestures(df_with_agreement, coverage)
    
    # Display results
    print("=" * 80)
    print("RANKED GESTURES")
    print("=" * 80)
    print(df_ranked.to_string(index=True))
    
    # Save results
    output_file = filepath.replace('.csv', '_ranked.csv')
    df_ranked.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Gestures: {len(df_ranked)}")
    print(f"Coverage Score: {coverage:.4f}")
    print(f"Mean Agreement: {df_ranked['agreement_score'].mean():.4f}")
    print(f"Std Agreement: {df_ranked['agreement_score'].std():.4f}")

if __name__ == "__main__":
    main()