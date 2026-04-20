"""
Customer Segmentation & Sales Forecasting Analysis Pipeline
Complete workflow for RFM analysis, PCA dimensionality reduction, and K-Means clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

print("=" * 80)
print("CUSTOMER SEGMENTATION & SALES FORECASTING ANALYSIS PIPELINE")
print("=" * 80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# PHASE 1: DATA EXPLORATION & PREPARATION
# ============================================================================
print("\n[PHASE 1] DATA EXPLORATION & PREPARATION")
print("-" * 80)

print("Loading CSV files...")
sales_train = pd.read_csv(os.path.join(BASE_DIR, 'sales_train.csv'))
items = pd.read_csv(os.path.join(BASE_DIR, 'items.csv'))
item_categories = pd.read_csv(os.path.join(BASE_DIR, 'item_categories.csv'))
shops = pd.read_csv(os.path.join(BASE_DIR, 'shops.csv'))
test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

print(f"✓ Sales Training Data: {sales_train.shape}")
print(f"✓ Items: {items.shape}")
print(f"✓ Item Categories: {item_categories.shape}")
print(f"✓ Shops: {shops.shape}")
print(f"✓ Test Data: {test.shape}")

# Data Quality Report
print("\n--- Data Quality Analysis ---")
print(f"\nSales Train Data:")
print(f"  - Date Range: {sales_train['date'].min()} to {sales_train['date'].max()}")
print(f"  - Missing Values:\n{sales_train.isnull().sum()}")
print(f"  - Data Types:\n{sales_train.dtypes}")
print(f"\nBasic Statistics:")
print(sales_train.describe())

# Merge supplemental data with sales data
sales_train = sales_train.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')
sales_train = sales_train.merge(shops[['shop_id', 'shop_name']], on='shop_id', how='left')
sales_train = sales_train.merge(item_categories[['item_category_id', 'item_category_name']], 
                                 on='item_category_id', how='left')

print(f"\n✓ Merged supplemental data. Final shape: {sales_train.shape}")

# ============================================================================
# PHASE 2: FEATURE ENGINEERING & RFM ANALYSIS
# ============================================================================
print("\n\n[PHASE 2] FEATURE ENGINEERING & RFM ANALYSIS")
print("-" * 80)

# Convert date to datetime with correct format (DD.MM.YYYY)
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
reference_date = sales_train['date'].max() + pd.Timedelta(days=1)

print(f"Reference Date (max transaction + 1 day): {reference_date}")

# RFM Analysis
print("\n--- Calculating RFM Features ---")

# Group by customer (shop_id as proxy for customer in this context)
customer_rfm = sales_train.groupby('shop_id').agg({
    'date': lambda x: (reference_date - x.max()).days,  # Recency (days since last purchase)
    'item_id': 'count',  # Frequency (number of purchases)
    'item_price': 'sum'  # Monetary (total spent)
}).rename(columns={
    'date': 'recency',
    'item_id': 'frequency',
    'item_price': 'monetary'
}).reset_index()

print(f"✓ RFM Data calculated for {len(customer_rfm)} customers")
print(f"\nRFM Statistics:")
print(customer_rfm[['recency', 'frequency', 'monetary']].describe())

# Additional customer behavior features
print("\n--- Calculating Additional Behavior Features ---")

customer_behavior = sales_train.groupby('shop_id').agg({
    'item_price': ['mean', 'std', 'min', 'max'],
    'item_id': 'nunique',
    'item_category_id': 'nunique',
    'date': lambda x: (x.max() - x.min()).days  # Customer lifetime in days
}).reset_index()

customer_behavior.columns = ['shop_id', 'avg_price', 'std_price', 'min_price', 'max_price', 
                             'unique_items', 'unique_categories', 'customer_lifetime']

# Handle NaN values (when only 1 purchase)
customer_behavior['std_price'].fillna(0, inplace=True)

print(f"✓ Behavior features calculated")
print(f"\nBehavior Features Summary:")
print(customer_behavior[['avg_price', 'unique_items', 'unique_categories', 'customer_lifetime']].describe())

# Merge RFM with behavior features
customer_features = customer_rfm.merge(customer_behavior, on='shop_id', how='left')
print(f"\n✓ Total features per customer: {len(customer_features.columns) - 1}")

# Handle outliers in RFM using IQR method
print("\n--- Handling Outliers ---")
for col in ['recency', 'frequency', 'monetary']:
    Q1 = customer_features[col].quantile(0.25)
    Q3 = customer_features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_removed = ((customer_features[col] < lower_bound) | 
                        (customer_features[col] > upper_bound)).sum()
    
    # Clip outliers instead of removing
    customer_features[col] = customer_features[col].clip(lower_bound, upper_bound)
    print(f"  {col}: {outliers_removed} outliers clipped")

# ============================================================================
# PHASE 3: NORMALIZATION & SCALING
# ============================================================================
print("\n\n[PHASE 3] NORMALIZATION & SCALING")
print("-" * 80)

# Select features for clustering (exclude shop_id)
clustering_features = customer_features.columns[1:].tolist()
X = customer_features[clustering_features].copy()

print(f"Features for clustering: {clustering_features}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Features scaled using StandardScaler")
print(f"  Mean: {X_scaled.mean(axis=0)[:3]}... (first 3)")
print(f"  Std: {X_scaled.std(axis=0)[:3]}... (first 3)")

# ============================================================================
# PHASE 4: DIMENSIONALITY REDUCTION WITH PCA
# ============================================================================
print("\n\n[PHASE 4] DIMENSIONALITY REDUCTION WITH PCA")
print("-" * 80)

# Fit PCA
pca = PCA()
X_pca_full = pca.fit_transform(X_scaled)

# Calculate cumulative explained variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
n_components_80 = np.argmax(cumsum_var >= 0.80) + 1
n_components_90 = np.argmax(cumsum_var >= 0.90) + 1

print(f"✓ PCA fitted on {X_scaled.shape[1]} features")
print(f"\nExplained Variance:")
print(f"  - Components for 80% variance: {n_components_80}")
print(f"  - Components for 90% variance: {n_components_90}")
print(f"  - Top 5 components variance: {pca.explained_variance_ratio_[:5]}")

# Use optimal components (let's use 80% threshold)
pca = PCA(n_components=n_components_80)
X_pca = pca.fit_transform(X_scaled)
print(f"✓ PCA reduced to {n_components_80} components")
print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Visualizations - PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot - Explained Variance by Component')
axes[0].grid(axis='y', alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', color='darkorange', linewidth=2)
axes[1].axhline(y=0.80, color='r', linestyle='--', label='80% threshold')
axes[1].axhline(y=0.90, color='g', linestyle='--', label='90% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '01_pca_variance_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: 01_pca_variance_analysis.png")
plt.close()

# ============================================================================
# PHASE 5: OPTIMAL CLUSTERING ANALYSIS
# ============================================================================
print("\n\n[PHASE 5] OPTIMAL CLUSTERING ANALYSIS")
print("-" * 80)

# Elbow method and silhouette analysis
K_range = range(2, 11)
inertias = []
silhouette_scores = []

print("Testing K values 2-10...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")

# Visualizations - Clustering Optimization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
axes[0].plot(K_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Elbow Method - Finding Optimal K')
axes[0].grid(alpha=0.3)

# Silhouette scores
axes[1].plot(K_range, silhouette_scores, 's-', color='darkorange', linewidth=2, markersize=8)
optimal_k = K_range[np.argmax(silhouette_scores)]
axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis - Finding Optimal K')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '02_optimal_k_analysis.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: 02_optimal_k_analysis.png")
plt.close()

print(f"\n*** RECOMMENDED OPTIMAL K: {optimal_k} ***")

# ============================================================================
# PHASE 6: CUSTOMER SEGMENTATION WITH K-MEANS
# ============================================================================
print("\n\n[PHASE 6] CUSTOMER SEGMENTATION WITH K-MEANS")
print("-" * 80)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
customer_features['cluster'] = kmeans_final.fit_predict(X_pca)

print(f"✓ K-Means clustering completed with K={optimal_k}")
print(f"\nCluster Distribution:")
print(customer_features['cluster'].value_counts().sort_index())

# ============================================================================
# PHASE 7: CLUSTER ANALYSIS & PROFILING
# ============================================================================
print("\n\n[PHASE 7] CLUSTER ANALYSIS & PROFILING")
print("-" * 80)

# Create segment profiles
segment_profiles = customer_features.groupby('cluster')[clustering_features].mean()
print(f"\n✓ Segment Profiles (Mean Values):")
print(segment_profiles)

# Create summary statistics by cluster
cluster_summary = customer_features.groupby('cluster').agg({
    'recency': ['mean', 'std'],
    'frequency': ['mean', 'std'],
    'monetary': ['mean', 'std'],
    'shop_id': 'count'
}).round(2)
cluster_summary.columns = ['recency_mean', 'recency_std', 'frequency_mean', 'frequency_std', 
                           'monetary_mean', 'monetary_std', 'customer_count']
print(f"\nCluster Summary Statistics:")
print(cluster_summary)

# Cluster characteristics visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Cluster Characteristics - RFM Analysis', fontsize=16, fontweight='bold')

# Recency by cluster
axes[0, 0].boxplot([customer_features[customer_features['cluster'] == i]['recency'] 
                     for i in range(optimal_k)])
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Recency (days)')
axes[0, 0].set_title('Recency Distribution by Cluster')
axes[0, 0].grid(axis='y', alpha=0.3)

# Frequency by cluster
axes[0, 1].boxplot([customer_features[customer_features['cluster'] == i]['frequency'] 
                     for i in range(optimal_k)])
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Frequency (purchases)')
axes[0, 1].set_title('Purchase Frequency by Cluster')
axes[0, 1].grid(axis='y', alpha=0.3)

# Monetary by cluster
axes[0, 2].boxplot([customer_features[customer_features['cluster'] == i]['monetary'] 
                     for i in range(optimal_k)])
axes[0, 2].set_xlabel('Cluster')
axes[0, 2].set_ylabel('Monetary (total spent)')
axes[0, 2].set_title('Monetary Value by Cluster')
axes[0, 2].grid(axis='y', alpha=0.3)

# Cluster size
cluster_sizes = customer_features['cluster'].value_counts().sort_index()
axes[1, 0].bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.7)
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].set_title('Cluster Size Distribution')
axes[1, 0].grid(axis='y', alpha=0.3)

# Average spend per cluster
avg_spend = customer_features.groupby('cluster')['monetary'].mean()
axes[1, 1].bar(avg_spend.index, avg_spend.values, color='darkorange', alpha=0.7)
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Average Monetary Value')
axes[1, 1].set_title('Average Customer Value by Cluster')
axes[1, 1].grid(axis='y', alpha=0.3)

# Average frequency per cluster
avg_freq = customer_features.groupby('cluster')['frequency'].mean()
axes[1, 2].bar(avg_freq.index, avg_freq.values, color='green', alpha=0.7)
axes[1, 2].set_xlabel('Cluster')
axes[1, 2].set_ylabel('Average Frequency')
axes[1, 2].set_title('Average Purchase Frequency by Cluster')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '03_cluster_characteristics.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: 03_cluster_characteristics.png")
plt.close()

# PCA scatter plot with clusters
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_features['cluster'], 
                     cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
          c='red', marker='X', s=200, edgecolors='black', linewidth=2, label='Centroids')
ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax.set_title(f'Customer Segments (K-Means, K={optimal_k}) - PCA Visualization', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '04_pca_clusters_scatter.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: 04_pca_clusters_scatter.png")
plt.close()

# ============================================================================
# PHASE 8: RFM DISTRIBUTIONS
# ============================================================================
print("\n\n[PHASE 8] RFM DISTRIBUTION VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Recency distribution
axes[0].hist(customer_features['recency'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Recency (days since last purchase)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Recency Distribution')
axes[0].grid(axis='y', alpha=0.3)

# Frequency distribution (log scale for better visibility)
axes[1].hist(customer_features['frequency'], bins=50, color='darkorange', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Frequency (number of purchases)')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Purchase Frequency Distribution')
axes[1].grid(axis='y', alpha=0.3)

# Monetary distribution (log scale)
axes[2].hist(customer_features['monetary'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Monetary (total amount spent)')
axes[2].set_ylabel('Number of Customers')
axes[2].set_title('Monetary Value Distribution')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '05_rfm_distributions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: 05_rfm_distributions.png")
plt.close()

# ============================================================================
# PHASE 9: SAVE RESULTS
# ============================================================================
print("\n\n[PHASE 9] SAVING RESULTS")
print("-" * 80)

# Save processed data with clusters
customer_features.to_csv(os.path.join(OUTPUT_DIR, 'processed_data.csv'), index=False)
print(f"✓ Saved: processed_data.csv")

# Save segment profiles
segment_profiles.to_csv(os.path.join(OUTPUT_DIR, 'segment_profiles.csv'))
print(f"✓ Saved: segment_profiles.csv")

# Save cluster summary
cluster_summary.to_csv(os.path.join(OUTPUT_DIR, 'cluster_summary.csv'))
print(f"✓ Saved: cluster_summary.csv")

# ============================================================================
# PHASE 10: GENERATE SUMMARY REPORT
# ============================================================================
print("\n\n[PHASE 10] GENERATING SUMMARY REPORT")
print("-" * 80)

summary_report = f"""# Customer Segmentation Analysis - Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis segments customers of an e-commerce platform using RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering. The goal is to identify distinct customer groups for targeted marketing strategies.

## Methodology

### 1. Data Preparation
- **Total Transactions Analyzed:** {len(sales_train):,}
- **Date Range:** {sales_train['date'].min().date()} to {sales_train['date'].max().date()}
- **Total Customers (Shops):** {len(customer_features):,}
- **Reference Date:** {reference_date.date()}

### 2. RFM Analysis
RFM is a behavioral segmentation technique that measures:
- **Recency (R):** Days since last purchase (lower is better - more recent)
- **Frequency (F):** Number of purchases (higher is better - more frequent)
- **Monetary (M):** Total amount spent (higher is better - more valuable)

**RFM Statistics:**
```
{customer_features[['recency', 'frequency', 'monetary']].describe().to_string()}
```

### 3. Feature Engineering
Additional customer behavior features extracted:
- Average purchase price per transaction
- Price volatility (std deviation)
- Number of unique items purchased
- Number of unique product categories
- Customer lifetime (days from first to last purchase)

Total features used for clustering: {len(clustering_features)}

### 4. Dimensionality Reduction (PCA)
- **Original Features:** {X_scaled.shape[1]}
- **Reduced Dimensions:** {n_components_80}
- **Variance Explained:** {pca.explained_variance_ratio_.sum():.2%}
- **PCA Components Used:**

| Component | Variance Explained | Cumulative |
|-----------|-------------------|-----------|
{chr(10).join([f'| PC{i+1} | {pca.explained_variance_ratio_[i]:.2%} | {cumsum_var[i]:.2%} |' for i in range(min(5, len(pca.explained_variance_ratio_)))])}

### 5. Optimal Clustering
K-Means clustering was applied with K values tested from 2 to 10.

**Optimal K Selection:** {optimal_k} clusters
- **Selection Method:** Silhouette Score (highest score: {max(silhouette_scores):.4f})
- **Silhouette Score at K={optimal_k}:** {silhouette_scores[optimal_k-2]:.4f}

Higher silhouette scores indicate better-defined, more separated clusters.

### 6. Cluster Profiles

{cluster_summary.to_string()}

## Cluster Characteristics & Interpretations

"""

# Add detailed cluster profiles

# Calculate overall customer base metrics for relative comparisons
overall_recency_median = customer_features['recency'].median()
overall_frequency_median = customer_features['frequency'].median()
overall_monetary_median = customer_features['monetary'].median()

print(f"\nOverall Customer Base Medians:")
print(f"  Recency: {overall_recency_median:.1f} days")
print(f"  Frequency: {overall_frequency_median:.1f} purchases")
print(f"  Monetary: ${overall_monetary_median:,.2f}")

for i in range(optimal_k):
    cluster_data = customer_features[customer_features['cluster'] == i]
    
    recency_mean = cluster_data['recency'].mean()
    frequency_mean = cluster_data['frequency'].mean()
    monetary_mean = cluster_data['monetary'].mean()
    size = len(cluster_data)
    pct = (size / len(customer_features)) * 100
    
    summary_report += f"""
### Cluster {i}: {['High Value Loyal', 'At-Risk Recent', 'New Customers', 'Dormant', 'Moderate'][min(i, 4)]} ({size} customers, {pct:.1f}%)

**Key Metrics:**
- Average Recency: {recency_mean:.1f} days ({"Highly active" if recency_mean < 50 else "Moderately active" if recency_mean < 150 else "Low activity"})
- Average Frequency: {frequency_mean:.1f} purchases ({"Very frequent" if frequency_mean > 20 else "Frequent" if frequency_mean > 10 else "Occasional"})
- Average Monetary Value: ${monetary_mean:,.2f} ({"High value" if monetary_mean > customer_features['monetary'].quantile(0.75) else "Medium value" if monetary_mean > customer_features['monetary'].quantile(0.25) else "Low value"})

**Characteristics:**
- Customer Lifetime: {cluster_data['customer_lifetime'].mean():.0f} days
- Unique Items Purchased: {cluster_data['unique_items'].mean():.0f}
- Unique Categories: {cluster_data['unique_categories'].mean():.1f}
- Average Transaction Value: ${cluster_data['avg_price'].mean():,.2f}

**Recommended Marketing Strategy:**
"""
    
    # Determine strategy based on relative position vs overall customer base
    if (recency_mean < overall_recency_median and 
        frequency_mean > overall_frequency_median and 
        monetary_mean > overall_monetary_median):
        # Better than average in all RFM dimensions - VIP treatment
        summary_report += "- **VIP Program:** Premium benefits, exclusive offers, early access to sales\n"
        summary_report += "- **Retention Focus:** Regular engagement, loyalty rewards\n"
        summary_report += "- **Cross-sell:** Bundle offers, premium product recommendations\n"
    elif (recency_mean > overall_recency_median * 2 and 
          (frequency_mean < overall_frequency_median or monetary_mean < overall_monetary_median)):
        # Much less recent AND below average in frequency or monetary - reactivation needed
        summary_report += "- **Reactivation Campaign:** Win-back offers, special discounts\n"
        summary_report += "- **Personalized Re-engagement:** Product recommendations, new collection alerts\n"
        summary_report += "- **Feedback & Surveys:** Understand churn reasons\n"
    elif (frequency_mean > overall_frequency_median and monetary_mean > overall_monetary_median):
        # High frequency and monetary (loyal despite recency) - loyalty focus
        summary_report += "- **Loyalty Rewards:** Points program, exclusive member perks\n"
        summary_report += "- **Increase AOV:** Upselling, complementary product recommendations\n"
        summary_report += "- **Community Building:** Early access, beta testing opportunities\n"
    elif (recency_mean < overall_recency_median * 1.5 and 
          (frequency_mean > overall_frequency_median * 0.5 or monetary_mean > overall_monetary_median * 0.5)):
        # Recent buyers with reasonable frequency/monetary - growth potential
        summary_report += "- **Growth Acceleration:** Time-based incentives, repeat purchase discounts\n"
        summary_report += "- **Category Expansion:** Cross-category recommendations\n"
        summary_report += "- **Educational Content:** Product benefits, usage tips\n"
    else:
        # Average or below average - nurture and develop
        summary_report += "- **Nurture Programs:** Educational content, welcome series\n"
        summary_report += "- **Incentive Programs:** Time-limited offers, bundle deals\n"
        summary_report += "- **Re-engagement:** Personalized recommendations, abandoned cart recovery\n"

summary_report += """

## Key Insights & Recommendations

### 1. Segment Your Communications
- Create targeted email campaigns for each cluster
- Use cluster-specific messaging and offers
- Tailor communication frequency based on recency patterns

### 2. Pricing & Promotions Strategy
- **VIP Clusters:** Premium pricing, exclusive access, minimal discounts
- **Growth Clusters:** Growth incentives, multi-buy discounts, loyalty bonuses
- **At-Risk Clusters:** Competitive pricing, special reactivation offers

### 3. Product Recommendations
- Use cluster profiles to inform personalization engines
- High-value customers: Premium, exclusive, latest products
- Growth clusters: Popular items, trending products, category introductions
- At-risk: Last-purchased categories, clearance items, complementary products

### 4. Customer Experience
- Tailor onboarding experiences based on cluster characteristics
- Adjust customer service levels by cluster value
- Create personalized journey maps for each cluster

### 5. Retention & Churn Prevention
- Monitor cluster transition (especially movement to dormant cluster)
- Implement early warning systems for high-value customers moving to at-risk
- Create cluster-specific retention programs

## Next Steps

1. **Validate Results:** Present findings to business stakeholders for feedback
2. **Implement Actions:** Deploy targeted campaigns based on cluster recommendations
3. **Monitor Performance:** Track KPIs by cluster (retention, AOV, engagement)
4. **Iterate:** Re-run analysis quarterly to track cluster evolution and customer migration
5. **Refinement:** Incorporate additional features (seasonality, product affinity, etc.) for deeper insights

## Technical Details

**Libraries Used:**
- pandas, numpy: Data processing and numerical computation
- scikit-learn: Machine learning (PCA, K-Means)
- matplotlib, seaborn: Data visualization

**Parameters:**
- PCA Variance Threshold: 80-90%
- K-Means Random State: 42 (for reproducibility)
- Feature Scaling: StandardScaler (zero mean, unit variance)
- Silhouette Score: Used for optimal K selection

---
*Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(os.path.join(OUTPUT_DIR, 'analysis_summary.md'), 'w') as f:
    f.write(summary_report)

print(f"✓ Saved: analysis_summary.md")

print("\n" + "=" * 80)
print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput Location: {OUTPUT_DIR}")
print(f"\nGenerated Files:")
print(f"  ✓ processed_data.csv - Cleaned data with cluster assignments")
print(f"  ✓ segment_profiles.csv - Cluster profiles (mean feature values)")
print(f"  ✓ cluster_summary.csv - Summary statistics by cluster")
print(f"  ✓ analysis_summary.md - Detailed analysis report")
print(f"\nVisualizations (in visualizations/ folder):")
print(f"  ✓ 01_pca_variance_analysis.png")
print(f"  ✓ 02_optimal_k_analysis.png")
print(f"  ✓ 03_cluster_characteristics.png")
print(f"  ✓ 04_pca_clusters_scatter.png")
print(f"  ✓ 05_rfm_distributions.png")
print(f"\nNext Step: Run 'python outputs/generate_html_report.py' to create interactive HTML report")
print("=" * 80)
