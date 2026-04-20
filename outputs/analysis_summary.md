# Customer Segmentation Analysis - Summary Report

**Generated:** 2026-04-20 19:57:37

## Executive Summary

This analysis segments customers of an e-commerce platform using RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering. The goal is to identify distinct customer groups for targeted marketing strategies.

## Methodology

### 1. Data Preparation
- **Total Transactions Analyzed:** 2,935,849
- **Date Range:** 2013-01-01 to 2015-10-31
- **Total Customers (Shops):** 60
- **Reference Date:** 2015-11-01

### 2. RFM Analysis
RFM is a behavioral segmentation technique that measures:
- **Recency (R):** Days since last purchase (lower is better - more recent)
- **Frequency (F):** Number of purchases (higher is better - more frequent)
- **Monetary (M):** Total amount spent (higher is better - more valuable)

**RFM Statistics:**
```
         recency      frequency      monetary
count  60.000000      60.000000  6.000000e+01
mean   21.191667   44747.106250  4.173966e+07
std    33.301059   32537.058838  3.121942e+07
min     1.000000     306.000000  3.568190e+05
25%     1.000000   20503.750000  1.786439e+07
50%     1.000000   42037.500000  3.796641e+07
75%    32.500000   58211.000000  5.612990e+07
max    79.750000  114771.875000  1.135282e+08
```

### 3. Feature Engineering
Additional customer behavior features extracted:
- Average purchase price per transaction
- Price volatility (std deviation)
- Number of unique items purchased
- Number of unique product categories
- Customer lifetime (days from first to last purchase)

Total features used for clustering: 10

### 4. Dimensionality Reduction (PCA)
- **Original Features:** 10
- **Reduced Dimensions:** 3
- **Variance Explained:** 80.43%
- **PCA Components Used:**

| Component | Variance Explained | Cumulative |
|-----------|-------------------|-----------|
| PC1 | 46.73% | 46.73% |
| PC2 | 22.36% | 69.09% |
| PC3 | 11.34% | 80.43% |

### 5. Optimal Clustering
K-Means clustering was applied with K values tested from 2 to 10.

**Optimal K Selection:** 2 clusters
- **Selection Method:** Silhouette Score (highest score: 0.5240)
- **Silhouette Score at K=2:** 0.5240

Higher silhouette scores indicate better-defined, more separated clusters.

### 6. Cluster Profiles

         recency_mean  recency_std  frequency_mean  frequency_std  monetary_mean  monetary_std  customer_count
cluster                                                                                                       
0               56.43        34.79         8718.93        9485.95     6408612.73    6779334.92              15
1                9.44        23.18        56756.50       28300.58    53516670.77   26923874.22              45

## Cluster Characteristics & Interpretations


### Cluster 0: High Value Loyal (15 customers, 25.0%)

**Key Metrics:**
- Average Recency: 56.4 days (Moderately active)
- Average Frequency: 8718.9 purchases (Very frequent)
- Average Monetary Value: $6,408,612.73 (Low value)

**Characteristics:**
- Customer Lifetime: 318 days
- Unique Items Purchased: 2243
- Unique Categories: 38.1
- Average Transaction Value: $876.88

**Recommended Marketing Strategy:**
- **Reactivation Campaign:** Win-back offers, special discounts
- **Personalized Re-engagement:** Product recommendations, new collection alerts
- **Feedback & Surveys:** Understand churn reasons

### Cluster 1: At-Risk Recent (45 customers, 75.0%)

**Key Metrics:**
- Average Recency: 9.4 days (Highly active)
- Average Frequency: 56756.5 purchases (Very frequent)
- Average Monetary Value: $53,516,670.77 (Medium value)

**Characteristics:**
- Customer Lifetime: 980 days
- Unique Items Purchased: 8677
- Unique Categories: 60.0
- Average Transaction Value: $941.94

**Recommended Marketing Strategy:**
- **Loyalty Rewards:** Points program, exclusive member perks
- **Increase AOV:** Upselling, complementary product recommendations
- **Community Building:** Early access, beta testing opportunities


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
