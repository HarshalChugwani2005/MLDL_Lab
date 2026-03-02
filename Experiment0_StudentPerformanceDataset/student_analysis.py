"""
Experiment 0: Student Performance Analysis
Dataset: Student Performance Dataset
Exercises: NumPy, Pandas, Matplotlib, Seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

# Load dataset
data = pd.read_csv(os.path.join(script_dir, 'student_performance.csv'))

print("EXPERIMENT 0: STUDENT PERFORMANCE ANALYSIS")

# =============================================
# EXERCISE 1: NumPy Basics
# =============================================
print("\n1. NUMPY BASICS")

# Load Final_Score as NumPy array
final_score = data['Final_Score'].to_numpy()
print(f"Final_Score array shape: {final_score.shape}")

# Statistical measures
mean_score = np.mean(final_score)
median_score = np.median(final_score)
std_score = np.std(final_score)
min_score = np.min(final_score)
max_score = np.max(final_score)

print(f"Mean: {mean_score:.2f}, Median: {median_score:.2f}, Std Dev: {std_score:.2f}")
print(f"Min: {min_score:.2f}, Max: {max_score:.2f}")

# Min-Max Normalization
normalized_score = (final_score - min_score) / (max_score - min_score)
data['Normalized_Score'] = normalized_score
print(f"Normalized scores range: [{normalized_score.min():.2f}, {normalized_score.max():.2f}]")

# =============================================
# EXERCISE 2: Pandas Data Handling
# =============================================
print("\n2. PANDAS DATA HANDLING")

print(f"Dataset Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Missing Values: {data.isnull().sum().sum()}")

# Create Performance labels
def get_performance(score):
    if score >= 80: return 'Excellent'
    elif score >= 70: return 'Good'
    elif score >= 60: return 'Average'
    else: return 'Poor'

data['Performance'] = data['Final_Score'].apply(get_performance)
print(f"\nPerformance Distribution:\n{data['Performance'].value_counts()}")

# =============================================
# EXERCISE 3 & 4: Matplotlib & Seaborn Visualization
# =============================================
print("\n3. GENERATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Line Plot - Hours Studied vs Final Score
ax1 = axes[0, 0]
ax1.plot(data['Hours_Studied'], data['Final_Score'], 'bo-', markersize=6)
ax1.set_xlabel('Hours Studied')
ax1.set_ylabel('Final Score')
ax1.set_title('Hours Studied vs Final Score')
ax1.grid(True, alpha=0.3)

# Plot 2: Histogram of Final Score
ax2 = axes[0, 1]
ax2.hist(data['Final_Score'], bins=8, color='green', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Final Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Final Score')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Scatter Plot (Seaborn) with Performance colors
ax3 = axes[1, 0]
sns.scatterplot(x='Hours_Studied', y='Final_Score', hue='Performance', 
                data=data, s=100, palette='Set2', ax=ax3)
ax3.set_xlabel('Hours Studied')
ax3.set_ylabel('Final Score')
ax3.set_title('Hours Studied vs Final Score (by Performance)')
ax3.grid(True, alpha=0.3)

# Plot 4: Correlation Heatmap
ax4 = axes[1, 1]
numeric_cols = ['Hours_Studied', 'Attendance', 'Assignment_Score', 'Midterm_Score', 'Final_Score']
corr_matrix = data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax4)
ax4.set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'visualizations/student_analysis_results.png'), dpi=150)
plt.close()
print("Saved: visualizations/student_analysis_results.png")

# =============================================
# SUMMARY
# =============================================
print("\nSUMMARY")
print(f"""
Dataset: Student Performance ({len(data)} students)

Final Score Statistics:
- Mean: {mean_score:.2f}
- Median: {median_score:.2f}  
- Std Dev: {std_score:.2f}
- Range: {min_score:.2f} - {max_score:.2f}

Performance Distribution:
- Excellent (>=80): {(data['Performance'] == 'Excellent').sum()} students
- Good (70-79): {(data['Performance'] == 'Good').sum()} students
- Average (60-69): {(data['Performance'] == 'Average').sum()} students
- Poor (<60): {(data['Performance'] == 'Poor').sum()} students

Key Correlations with Final Score:
- Hours Studied: {data['Hours_Studied'].corr(data['Final_Score']):.3f}
- Attendance: {data['Attendance'].corr(data['Final_Score']):.3f}
- Assignment Score: {data['Assignment_Score'].corr(data['Final_Score']):.3f}
- Midterm Score: {data['Midterm_Score'].corr(data['Final_Score']):.3f}
""")
print("EXPERIMENT COMPLETED!")
