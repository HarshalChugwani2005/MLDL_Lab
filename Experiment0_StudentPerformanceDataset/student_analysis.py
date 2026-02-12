import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations folder if it doesn't exist
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv('student_performance.csv')

print("="*70)
print("STUDENT PERFORMANCE ANALYSIS - COMPREHENSIVE DATA SCIENCE EXERCISES")
print("="*70)

# ============================================================================
# Exercise 1: NumPy Basics
# ============================================================================

print("\n" + "="*70)
print("Exercise 1: NumPy Basics")
print("="*70)

# 1. Load Final_Score as NumPy array
final_score = data['Final_Score'].to_numpy()
print("\n1. Loaded Final_Score as NumPy array")
print(f"   Array shape: {final_score.shape}")
print(f"   Array dtype: {final_score.dtype}")

# 2. Compute mean, median, and standard deviation
mean_score = np.mean(final_score)
median_score = np.median(final_score)
std_dev_score = np.std(final_score)
var_score = np.var(final_score)
min_score_val = np.min(final_score)
max_score_val = np.max(final_score)

print("\n2. Statistical Measures of Final_Score:")
print(f"   Mean:              {mean_score:.2f}")
print(f"   Median:            {median_score:.2f}")
print(f"   Standard Deviation: {std_dev_score:.2f}")
print(f"   Variance:          {var_score:.2f}")
print(f"   Min Value:         {min_score_val:.2f}")
print(f"   Max Value:         {max_score_val:.2f}")
print(f"   Range:             {max_score_val - min_score_val:.2f}")

# 3. Perform Min-Max normalization
normalized_score = (final_score - min_score_val) / (max_score_val - min_score_val)
data['Normalized_Final_Score'] = normalized_score

print("\n3. Min-Max Normalization:")
print(f"   Formula: (X - Min) / (Max - Min)")
print(f"   Normalized Range: [0, 1]")
print(f"   Original Score Range: [{min_score_val}, {max_score_val}]")
print(f"   Sample Normalization:")
for i in range(min(5, len(final_score))):
    print(f"      Original: {final_score[i]:.2f} → Normalized: {normalized_score[i]:.4f}")

# ============================================================================
# Exercise 2: Pandas Data Handling
# ============================================================================

print("\n" + "="*70)
print("Exercise 2: Pandas Data Handling")
print("="*70)

# 1. Load CSV file using Pandas (already done above)
print("\n1. CSV file loaded successfully using Pandas")

# 2. Check shape, columns, and missing values
print("\n2. Dataset Information:")
print(f"   Shape (rows, columns): {data.shape}")
print(f"\n   Column Names and Data Types:")
for col in data.columns:
    print(f"      - {col}: {data[col].dtype}")

print(f"\n   Missing Values Summary:")
missing_values = data.isnull().sum()
if missing_values.sum() == 0:
    print(f"      No missing values found!")
else:
    print(missing_values)

print(f"\n   Dataset Statistics:")
print(data.describe())

# 3. Create Performance label based on Final_Score
def performance_label(score):
    if score >= 80:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 60:
        return 'Average'
    else:
        return 'Poor'

data['Performance'] = data['Final_Score'].apply(performance_label)

print("\n3. Performance Labels Created Based on Final_Score:")
print(f"   Excellent: Final_Score >= 80")
print(f"   Good:      Final_Score >= 70 and < 80")
print(f"   Average:   Final_Score >= 60 and < 70")
print(f"   Poor:      Final_Score < 60")
print(f"\n   Performance Distribution:")
print(data['Performance'].value_counts())

# ============================================================================
# Exercise 3: Matplotlib Visualization
# ============================================================================

print("\n" + "="*70)
print("Exercise 3: Matplotlib Visualization")
print("="*70)

# 1. Line plot: Hours_Studied vs Final_Score
print("\n1. Creating Line Plot: Hours_Studied vs Final_Score...")
plt.figure(figsize=(10, 6))
plt.plot(data['Hours_Studied'], data['Final_Score'], 
         marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
plt.title('Hours Studied vs Final Score', fontsize=14, fontweight='bold')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hours_studied_vs_final_score_lineplot.png'), dpi=300)
plt.close()
print("   ✓ Saved: hours_studied_vs_final_score_lineplot.png")

# 2. Histogram of Final_Score
print("\n2. Creating Histogram of Final_Score...")
plt.figure(figsize=(10, 6))
plt.hist(data['Final_Score'], bins=8, color='g', alpha=0.7, edgecolor='black')
plt.title('Distribution of Final Score', fontsize=14, fontweight='bold')
plt.xlabel('Final Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_final_score.png'), dpi=300)
plt.close()
print("   ✓ Saved: histogram_final_score.png")

# ============================================================================
# Exercise 4: Seaborn Visualization
# ============================================================================

print("\n" + "="*70)
print("Exercise 4: Seaborn Visualization")
print("="*70)

# 1. Scatter plot using seaborn
print("\n1. Creating Scatter Plot with Seaborn...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours_Studied', y='Final_Score', hue='Performance', 
                data=data, s=100, palette='Set2')
plt.title('Hours Studied vs Final Score (Colored by Performance)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.legend(title='Performance', title_fontsize=11, fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter_plot_performance.png'), dpi=300)
plt.close()
print("   ✓ Saved: scatter_plot_performance.png")

# 2. Heatmap for correlation analysis
print("\n2. Creating Heatmap for Correlation Analysis...")
plt.figure(figsize=(10, 8))
correlation_matrix = data[['Hours_Studied', 'Attendance', 'Assignment_Score', 
                            'Midterm_Score', 'Final_Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', 
            square=True, cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=0.5, linecolor='gray')
plt.title('Correlation Heatmap - Student Performance Metrics', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_correlation.png'), dpi=300)
plt.close()
print("   ✓ Saved: heatmap_correlation.png")

print("\n   Correlation Insights:")
print(correlation_matrix)

# 3. Boxplot for categorical analysis
print("\n3. Creating Boxplot for Categorical Analysis...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Performance', y='Final_Score', data=data, palette='Set2')
plt.title('Final Score Distribution by Performance Category', 
          fontsize=14, fontweight='bold')
plt.xlabel('Performance Category', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_performance_category.png'), dpi=300)
plt.close()
print("   ✓ Saved: boxplot_performance_category.png")

# Additional visualization: Distribution across multiple metrics
print("\n4. Creating Additional Visualization: Multi-metric Boxplots...")
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics = ['Hours_Studied', 'Attendance', 'Assignment_Score', 'Midterm_Score']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']

for idx, metric in enumerate(metrics):
    sns.boxplot(y=metric, data=data, ax=axes[idx], color=colors[idx])
    axes[idx].set_title(f'{metric} Distribution', fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_all_metrics.png'), dpi=300)
plt.close()
print("   ✓ Saved: boxplot_all_metrics.png")

# ============================================================================
# Summary and Key Insights
# ============================================================================

print("\n" + "="*70)
print("SUMMARY AND KEY INSIGHTS")
print("="*70)

print("\nDataset Overview:")
print(f"  • Total Students: {len(data)}")
print(f"  • Features: {list(data.columns)}")

print("\nFinal Score Statistics:")
print(f"  • Average Score: {mean_score:.2f}")
print(f"  • Median Score: {median_score:.2f}")
print(f"  • Standard Deviation: {std_dev_score:.2f}")
print(f"  • Score Range: {min_score_val} - {max_score_val}")

print("\nPerformance Distribution:")
for performance in ['Excellent', 'Good', 'Average', 'Poor']:
    count = (data['Performance'] == performance).sum()
    percentage = (count / len(data)) * 100
    print(f"  • {performance}: {count} students ({percentage:.1f}%)")

print("\nCorrelation Insights:")
print(f"  • Hours Studied ↔ Final Score: {data['Hours_Studied'].corr(data['Final_Score']):.3f}")
print(f"  • Attendance ↔ Final Score: {data['Attendance'].corr(data['Final_Score']):.3f}")
print(f"  • Assignment Score ↔ Final Score: {data['Assignment_Score'].corr(data['Final_Score']):.3f}")
print(f"  • Midterm Score ↔ Final Score: {data['Midterm_Score'].corr(data['Final_Score']):.3f}")

print("\nVisualizations Generated:")
print(f"  • Line plot showing Hours Studied vs Final Score trend")
print(f"  • Histogram showing Final Score distribution")
print(f"  • Scatter plot with performance categories")
print(f"  • Correlation heatmap for all metrics")
print(f"  • Boxplots for categorical analysis")
print(f"  • Multi-metric distribution visualization")

print("\n" + "="*70)
print("All visualizations saved to: visualizations/")
print("="*70)
print("\nExercise Completion Status: ✓ ALL EXERCISES COMPLETED SUCCESSFULLY")
print("="*70 + "\n")
