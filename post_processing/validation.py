from pathlib import Path
import pandas as pd
import numpy as np

CUR_DIR = Path(__file__).parent
OUTPUT_DIR = CUR_DIR / "ndw_output"

def analyze_flow_columns(csv_path: Path) -> None:
    """Check if flow_0 is the sum of other flow columns."""
    print(f"\n{'='*80}")
    print(f"ANALYZING FLOW COLUMNS: {csv_path.name}")
    print(f"{'='*80}\n")
    
    # Load data with error handling for malformed CSV
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
        print(f"✓ Loaded CSV (skipped malformed lines)")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    print(f"Total rows loaded: {len(df):,}\n")
    
    # Find all flow columns
    flow_cols = [col for col in df.columns if col.startswith('flow_')]
    flow_cols_sorted = sorted(flow_cols, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
    
    print(f"Flow columns found: {flow_cols_sorted}\n")
    
    if 'flow_0' not in flow_cols:
        print("❌ flow_0 column not found")
        return
    
    if len(flow_cols) < 2:
        print("⚠️  Only one flow column found, cannot compare")
        return
    
    # Get other flow columns (excluding flow_0)
    other_flow_cols = [col for col in flow_cols_sorted if col != 'flow_0']
    
    print(f"Testing if flow_0 = sum({', '.join(other_flow_cols)})\n")
    print(f"Note: Missing values treated as zero when summing\n")
    print("="*80)
    
    # Calculate sum of other flows (treat NaN as 0)
    df['flow_sum_others'] = df[other_flow_cols].fillna(0).sum(axis=1)
    
    # Only analyze rows where flow_0 is present (not NaN)
    df_valid = df[df['flow_0'].notna()].copy()
    
    print(f"Rows with flow_0 present: {len(df_valid):,}\n")
    
    if len(df_valid) == 0:
        print("❌ No rows with flow_0 data")
        return
    
    # Calculate difference
    df_valid['difference'] = df_valid['flow_0'] - df_valid['flow_sum_others']
    
    # Avoid division by zero
    df_valid['ratio'] = np.where(
        df_valid['flow_sum_others'] != 0,
        df_valid['flow_0'] / df_valid['flow_sum_others'],
        np.nan
    )
    
    # Statistics
    print("COMPARISON STATISTICS:")
    print("-"*80)
    print(f"flow_0 mean: {df_valid['flow_0'].mean():.2f}")
    print(f"Sum of others mean: {df_valid['flow_sum_others'].mean():.2f}")
    print(f"\nDifference (flow_0 - sum_others):")
    print(f"  Mean: {df_valid['difference'].mean():.2f}")
    print(f"  Std:  {df_valid['difference'].std():.2f}")
    print(f"  Min:  {df_valid['difference'].min():.2f}")
    print(f"  Max:  {df_valid['difference'].max():.2f}")
    print(f"\nRatio (flow_0 / sum_others):")
    valid_ratios = df_valid['ratio'].dropna()
    if len(valid_ratios) > 0:
        print(f"  Mean: {valid_ratios.mean():.3f}")
        print(f"  Std:  {valid_ratios.std():.3f}")
        print(f"  Min:  {valid_ratios.min():.3f}")
        print(f"  Max:  {valid_ratios.max():.3f}")
    else:
        print("  No valid ratios (sum_others always zero)")
    
    # Check if they're equal (within tolerance)
    tolerance = 50  # Allow 50 vehicle/hour difference
    df_valid['is_equal'] = np.abs(df_valid['difference']) < tolerance
    
    equal_count = df_valid['is_equal'].sum()
    equal_pct = (equal_count / len(df_valid)) * 100
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print("="*80)
    print(f"Rows where flow_0 ≈ sum of others (within {tolerance}): {equal_count:,} ({equal_pct:.1f}%)")
    
    if equal_pct > 95:
        print("\n✅ YES - flow_0 is the SUM of other flow columns (aggregate)")
    elif equal_pct < 5:
        print("\n❌ NO - flow_0 is a SEPARATE lane measurement")
    else:
        print("\n⚠️  MIXED - Some sites use flow_0 as aggregate, others as separate lane")
    
    # Show examples
    print(f"\n{'='*80}")
    print("SAMPLE ROWS (first 10):")
    print("="*80)
    display_cols = ['site_id'] + flow_cols_sorted + ['flow_sum_others', 'difference', 'ratio']
    print(df_valid[display_cols].head(10).to_string(index=False))
    
    # Show cases where they differ significantly
    df_diff = df_valid[~df_valid['is_equal']].copy()
    if len(df_diff) > 0:
        print(f"\n{'='*80}")
        print(f"EXAMPLES WHERE flow_0 ≠ sum (showing first 5 of {len(df_diff):,}):")
        print("="*80)
        print(df_diff[display_cols].head(5).to_string(index=False))
    
    # Group by site to see if pattern is consistent per site
    if 'site_id' in df_valid.columns:
        print(f"\n{'='*80}")
        print("PATTERN BY SITE (first 10 sites):")
        print("="*80)
        site_analysis = df_valid.groupby('site_id').agg({
            'is_equal': ['sum', 'count', lambda x: (x.sum() / len(x) * 100)]
        })
        site_analysis.columns = ['equal_count', 'total_count', 'equal_pct']
        site_analysis['pattern'] = site_analysis['equal_pct'].apply(
            lambda x: 'AGGREGATE' if x > 95 else ('SEPARATE' if x < 5 else 'MIXED')
        )
        print(site_analysis.head(10).to_string())
        
        # Summary by pattern
        print(f"\n{'='*80}")
        print("SUMMARY BY PATTERN:")
        print("="*80)
        pattern_counts = site_analysis['pattern'].value_counts()
        print(pattern_counts)
        print()


if __name__ == "__main__":
    csv_file = OUTPUT_DIR / "trafficspeed.csv"
    
    if not csv_file.exists():
        print(f"❌ File not found: {csv_file}")
        exit(1)
    
    analyze_flow_columns(csv_file)