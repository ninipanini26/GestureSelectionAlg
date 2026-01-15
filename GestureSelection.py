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

def compute_coverage(df, n_bins= 5):
    """
    Compute the coverage of gestures in the VAD (Valence-Arousal-Dominance) space.
    
    Coverage measures how well the gestures span the emotional space.
    Higher coverage means gestures are more diverse and cover more of the emotional spectrum.
    
    Args:
        df: DataFrame with valence, arousal, dominance columns
        n_bins: Number of bins to divide each dimension into
            use "cut" functions
        For each cut (need to put the cuts in own column)
    
    Returns:
        Coverage score (0-1), where 1 means complete coverage
    """
    # Normalize values to 0-1 range if not already
    #vad_data = df[['valence', 'arousal', 'dominance']].copy()
    
    labs = range (n_bins)
    
    df['bin_v'] = pd.cut(df['valence'], bins=n_bins, labels=labs)
    df['bin_a'] = pd.cut(df['arousal'], bins=n_bins, labels=labs)
    df['bin_d'] = pd.cut(df['dominance'], bins=n_bins, labels=labs)
    
    def combine(v, a, d):
        return f'{v}, {a}, {d}'
    
    df['combine']=df.apply(lambda x: combine(x['bin_v'], x["bin_a"], x['bin_d']), axis=1)
    
    coverage = len(df['combine'].unique())/len(df)
    
    return coverage
    


#FINISH THIS
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