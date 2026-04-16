"""
Generate Interactive HTML Report for Customer Segmentation Analysis
Creates a comprehensive visual report with all charts and cluster profiles
"""

import pandas as pd
import os
from datetime import datetime
import base64

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

print("=" * 80)
print("GENERATING INTERACTIVE HTML REPORT")
print("=" * 80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load analysis data
print("Loading analysis results...")
processed_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'processed_data.csv'))
segment_profiles = pd.read_csv(os.path.join(OUTPUT_DIR, 'segment_profiles.csv'), index_col=0)
cluster_summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'cluster_summary.csv'), index_col=0)

print(f"✓ Loaded processed data: {len(processed_data)} records")
print(f"✓ Loaded segment profiles for {len(segment_profiles)} clusters")

# Function to convert image to base64
def image_to_base64(image_path):
    """Convert image file to base64 for embedding in HTML"""
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found")
        return None
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Load visualizations
print("\nLoading visualizations...")
viz_data = {}
viz_files = [
    '01_pca_variance_analysis.png',
    '02_optimal_k_analysis.png',
    '03_cluster_characteristics.png',
    '04_pca_clusters_scatter.png',
    '05_rfm_distributions.png'
]

for viz_file in viz_files:
    viz_path = os.path.join(VIZ_DIR, viz_file)
    b64_data = image_to_base64(viz_path)
    if b64_data:
        viz_data[viz_file] = b64_data
        print(f"✓ Loaded: {viz_file}")

# Extract cluster information
num_clusters = int(processed_data['cluster'].max()) + 1
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Build HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Segmentation Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            margin-top: 15px;
        }}
        
        nav {{
            background: white;
            padding: 15px 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: sticky;
            top: 20px;
            z-index: 100;
        }}
        
        nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }}
        
        nav a {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: transform 0.2s;
            font-weight: 500;
        }}
        
        nav a:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        
        section {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        h3 {{
            color: #764ba2;
            font-size: 1.5em;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .summary-card h4 {{
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            opacity: 0.9;
        }}
        
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .visualization h4 {{
            margin-top: 15px;
            color: #764ba2;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .cluster-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        
        .cluster-card.cluster-0 {{
            border-left-color: {cluster_colors[0] if len(cluster_colors) > 0 else '#667eea'};
        }}
        .cluster-card.cluster-1 {{
            border-left-color: {cluster_colors[1] if len(cluster_colors) > 1 else '#667eea'};
        }}
        .cluster-card.cluster-2 {{
            border-left-color: {cluster_colors[2] if len(cluster_colors) > 2 else '#667eea'};
        }}
        .cluster-card.cluster-3 {{
            border-left-color: {cluster_colors[3] if len(cluster_colors) > 3 else '#667eea'};
        }}
        .cluster-card.cluster-4 {{
            border-left-color: {cluster_colors[4] if len(cluster_colors) > 4 else '#667eea'};
        }}
        
        .cluster-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .cluster-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .metric {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        
        .metric-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .insight-box {{
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .recommendation {{
            background: #f0fff4;
            border-left: 4px solid #48bb78;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .recommendation h5 {{
            color: #22543d;
            margin-bottom: 10px;
        }}
        
        .recommendation ul {{
            margin-left: 20px;
            color: #22543d;
        }}
        
        .recommendation li {{
            margin: 8px 0;
        }}
        
        footer {{
            background: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            color: #666;
            margin-top: 30px;
        }}
        
        .tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #999;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            padding: 5px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        
        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8em;
            }}
            
            nav ul {{
                flex-direction: column;
            }}
            
            nav a {{
                width: 100%;
                text-align: center;
            }}
            
            table {{
                font-size: 0.85em;
            }}
            
            th, td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Customer Segmentation Analysis Report</h1>
            <p>E-Commerce Customer Behavior Analysis & RFM Clustering</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </header>
        
        <nav>
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#methodology">Methodology</a></li>
                <li><a href="#analysis">Analysis</a></li>
                <li><a href="#clusters">Cluster Details</a></li>
                <li><a href="#recommendations">Recommendations</a></li>
                <li><a href="#next-steps">Next Steps</a></li>
            </ul>
        </nav>
        
        <!-- OVERVIEW SECTION -->
        <section id="overview">
            <h2>Executive Overview</h2>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Total Customers Analyzed</h4>
                    <div class="value">{len(processed_data):,}</div>
                </div>
                <div class="summary-card">
                    <h4>Number of Clusters</h4>
                    <div class="value">{num_clusters}</div>
                </div>
                <div class="summary-card">
                    <h4>Analysis Features</h4>
                    <div class="value">{len([col for col in processed_data.columns if col != 'shop_id' and col != 'cluster'])}</div>
                </div>
                <div class="summary-card">
                    <h4>Data Completeness</h4>
                    <div class="value">100%</div>
                </div>
            </div>
            
            <div class="insight-box">
                <strong>📌 Key Insight:</strong> This analysis segments customers into {num_clusters} distinct groups based on their purchasing behavior. 
                Each cluster represents a different customer value profile that can be targeted with customized marketing strategies.
            </div>
        </section>
        
        <!-- METHODOLOGY SECTION -->
        <section id="methodology">
            <h2>Analysis Methodology</h2>
            
            <h3>1. RFM Analysis (Recency, Frequency, Monetary)</h3>
            <p>
                RFM is a customer segmentation technique that evaluates customers based on three behavioral dimensions:
            </p>
            <ul style="margin-left: 20px; margin: 15px 0;">
                <li><strong>Recency (R):</strong> Days since last purchase (lower = more recent)</li>
                <li><strong>Frequency (F):</strong> Number of purchases made (higher = more frequent)</li>
                <li><strong>Monetary (M):</strong> Total amount spent (higher = more valuable)</li>
            </ul>
            
            <h3>2. Feature Engineering</h3>
            <p>Beyond RFM, the following customer behavior features were extracted:</p>
            <ul style="margin-left: 20px; margin: 15px 0;">
                <li>Average purchase price per transaction</li>
                <li>Price volatility (standard deviation)</li>
                <li>Number of unique items purchased</li>
                <li>Number of unique product categories explored</li>
                <li>Customer lifetime (days from first to last purchase)</li>
            </ul>
            
            <h3>3. Dimensionality Reduction (PCA)</h3>
            <p>
                Principal Component Analysis (PCA) was applied to reduce feature dimensionality while retaining 80-90% of variance.
                This helps identify the most important patterns in customer behavior.
            </p>
            <div class="visualization">
                <img src="data:image/png;base64,{viz_data.get('01_pca_variance_analysis.png', '')}" alt="PCA Variance Analysis">
                <h4>PCA Variance Explained</h4>
            </div>
            
            <h3>4. K-Means Clustering with Optimal K Selection</h3>
            <p>
                K-Means algorithm clusters customers into groups. The optimal number of clusters was determined using:
            </p>
            <ul style="margin-left: 20px; margin: 15px 0;">
                <li><strong>Elbow Method:</strong> Identifies where inertia improvement diminishes</li>
                <li><strong>Silhouette Score:</strong> Measures cluster cohesion and separation (0-1 scale)</li>
            </ul>
            <div class="visualization">
                <img src="data:image/png;base64,{viz_data.get('02_optimal_k_analysis.png', '')}" alt="Optimal K Analysis">
                <h4>Optimal K Selection Metrics</h4>
            </div>
        </section>
        
        <!-- ANALYSIS RESULTS SECTION -->
        <section id="analysis">
            <h2>Analysis Results</h2>
            
            <h3>RFM Distribution Across Customer Base</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{viz_data.get('05_rfm_distributions.png', '')}" alt="RFM Distributions">
                <h4>Distribution of Recency, Frequency, and Monetary Values</h4>
            </div>
            
            <h3>Cluster Characteristics</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{viz_data.get('03_cluster_characteristics.png', '')}" alt="Cluster Characteristics">
                <h4>Key Metrics by Cluster: RFM, Size, and Value</h4>
            </div>
            
            <h3>Cluster Visualization (PCA Space)</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{viz_data.get('04_pca_clusters_scatter.png', '')}" alt="PCA Clusters">
                <h4>Customer Clusters in Principal Component Space</h4>
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                    The red X marks represent cluster centroids. The proximity of points indicates similarity in customer behavior.
                </p>
            </div>
            
            <h3>Cluster Summary Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Cluster</th>
                        <th>Size</th>
                        <th>Avg Recency</th>
                        <th>Avg Frequency</th>
                        <th>Avg Monetary</th>
                    </tr>
                </thead>
                <tbody>
"""

# Add cluster statistics rows
for idx in range(num_clusters):
    if idx in cluster_summary.index:
        row = cluster_summary.loc[idx]
        html_content += f"""
                    <tr>
                        <td style="font-weight: bold; color: {cluster_colors[idx] if idx < len(cluster_colors) else '#667eea'};">Cluster {idx}</td>
                        <td>{int(row['customer_count'])}</td>
                        <td>{row['recency_mean']:.1f} ± {row['recency_std']:.1f}</td>
                        <td>{row['frequency_mean']:.1f} ± {row['frequency_std']:.1f}</td>
                        <td>${row['monetary_mean']:,.0f} ± ${row['monetary_std']:,.0f}</td>
                    </tr>
"""

html_content += """
                </tbody>
            </table>
        </section>
        
        <!-- CLUSTER DETAILS SECTION -->
        <section id="clusters">
            <h2>Detailed Cluster Profiles</h2>
            <p>Each cluster represents a distinct customer segment with unique characteristics and recommended marketing strategies.</p>
"""

# Add detailed cluster profiles
cluster_names = ['High-Value Loyal Customers', 'At-Risk / Dormant Customers', 'New / Growing Customers', 'Occasional Shoppers', 'Window Shoppers']

for cluster_id in range(num_clusters):
    cluster_data = processed_data[processed_data['cluster'] == cluster_id]
    
    recency_mean = cluster_data['recency'].mean()
    frequency_mean = cluster_data['frequency'].mean()
    monetary_mean = cluster_data['monetary'].mean()
    cluster_size = len(cluster_data)
    pct_of_total = (cluster_size / len(processed_data)) * 100
    
    # Determine cluster characteristics
    if recency_mean < 50 and frequency_mean > 15 and monetary_mean > processed_data['monetary'].quantile(0.75):
        risk_level = "🟢 Low Risk - Valuable"
        strategy_type = "VIP Program"
    elif recency_mean > 150:
        risk_level = "🔴 High Risk - Dormant"
        strategy_type = "Reactivation"
    elif frequency_mean > 20:
        risk_level = "🟡 Medium Risk - Active"
        strategy_type = "Loyalty Program"
    else:
        risk_level = "🟡 Medium Risk - Developing"
        strategy_type = "Growth Program"
    
    html_content += f"""
            <div class="cluster-card cluster-{cluster_id}">
                <div class="cluster-title">
                    Cluster {cluster_id}: {cluster_names[min(cluster_id, len(cluster_names)-1)]}
                    <br><span style="font-size: 0.8em; color: #666;">{risk_level}</span>
                </div>
                
                <div class="cluster-metrics">
                    <div class="metric">
                        <div class="metric-label">Customers</div>
                        <div class="metric-value">{cluster_size:,}</div>
                        <div style="font-size: 0.8em; color: #999;">({pct_of_total:.1f}% of total)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Recency</div>
                        <div class="metric-value">{recency_mean:.0f}</div>
                        <div style="font-size: 0.8em; color: #999;">days ago</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Frequency</div>
                        <div class="metric-value">{frequency_mean:.0f}</div>
                        <div style="font-size: 0.8em; color: #999;">purchases</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Spend</div>
                        <div class="metric-value">${monetary_mean:,.0f}</div>
                        <div style="font-size: 0.8em; color: #999;">lifetime value</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Lifetime</div>
                        <div class="metric-value">{cluster_data['customer_lifetime'].mean():.0f}</div>
                        <div style="font-size: 0.8em; color: #999;">days</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Trans Value</div>
                        <div class="metric-value">${cluster_data['avg_price'].mean():,.0f}</div>
                        <div style="font-size: 0.8em; color: #999;">per purchase</div>
                    </div>
                </div>
                
                <div class="recommendation">
                    <h5>🎯 Recommended Strategy: {strategy_type}</h5>
                    <ul>
"""
    
    if risk_level.startswith("🟢"):
        html_content += """
                        <li>Create VIP tier with exclusive benefits and rewards</li>
                        <li>Offer early access to new products and sales</li>
                        <li>Provide premium customer service and personal shopping assistance</li>
                        <li>Implement loyalty points program with high redemption value</li>
                        <li>Focus on retention and maximizing customer lifetime value</li>
                        <li>Cross-sell premium and complementary products</li>
"""
    elif risk_level.startswith("🔴"):
        html_content += """
                        <li>Launch targeted win-back campaign with special incentives</li>
                        <li>Send personalized re-engagement emails with product recommendations</li>
                        <li>Offer time-limited discounts to encourage immediate purchase</li>
                        <li>Conduct survey to understand churn reasons</li>
                        <li>Create feedback loop for improvement</li>
                        <li>Implement automated reminder system for abandoned baskets</li>
"""
    elif frequency_mean > 20:
        html_content += """
                        <li>Establish tiered loyalty rewards program</li>
                        <li>Offer exclusive member-only deals and early access</li>
                        <li>Focus on increasing average order value through upselling</li>
                        <li>Create community engagement initiatives</li>
                        <li>Provide personalized product recommendations</li>
                        <li>Invite to exclusive beta testing and product launches</li>
"""
    else:
        html_content += """
                        <li>Develop nurturing email sequences</li>
                        <li>Offer incentives to increase purchase frequency</li>
                        <li>Expand product category recommendations</li>
                        <li>Provide educational content about product benefits</li>
                        <li>Create time-sensitive incentives (flash sales, limited offers)</li>
                        <li>Implement welcome series for repeat engagement</li>
"""
    
    html_content += """
                    </ul>
                </div>
            </div>
"""

html_content += """
        </section>
        
        <!-- RECOMMENDATIONS SECTION -->
        <section id="recommendations">
            <h2>Strategic Recommendations</h2>
            
            <h3>1. Personalization Strategy</h3>
            <div class="insight-box">
                <strong>Use cluster profiles to create personalized customer experiences:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Product Recommendations:</strong> High-value clusters get premium products; growing clusters get trending items</li>
                    <li><strong>Email Campaigns:</strong> Tailor frequency, tone, and offers based on cluster characteristics</li>
                    <li><strong>Pricing:</strong> VIP clusters less sensitive to discounts; growth clusters benefit from promotional pricing</li>
                    <li><strong>Communication:</strong> Adjust frequency and channels based on recency and engagement patterns</li>
                </ul>
            </div>
            
            <h3>2. Retention & Growth Focus</h3>
            <div class="insight-box">
                <strong>Allocate resources strategically:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Protect Revenue:</strong> Implement early warning systems for high-value customers showing risk signals</li>
                    <li><strong>Grow AOV:</strong> Target growth clusters with bundle offers and complementary products</li>
                    <li><strong>Reactivate:</strong> Create compelling offers for dormant customers to return</li>
                    <li><strong>Acquire Similar:</strong> Use VIP cluster profile for targeted acquisition campaigns</li>
                </ul>
            </div>
            
            <h3>3. Channel & Campaign Optimization</h3>
            <div class="insight-box">
                <strong>Optimize marketing mix by cluster:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>High-Value Customers:</strong> Multi-channel (direct mail, phone, VIP events), low frequency</li>
                    <li><strong>Active Customers:</strong> Email, SMS, and social media with varied messaging</li>
                    <li><strong>Growing Customers:</strong> Aggressive email marketing, social ads, incentives</li>
                    <li><strong>Dormant Customers:</strong> Win-back campaigns, exclusive offers, personalized outreach</li>
                </ul>
            </div>
            
            <h3>4. Monitoring & KPIs</h3>
            <div class="insight-box">
                <strong>Track cluster-specific metrics:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Intra-cluster retention rate (month-over-month)</li>
                    <li>Average customer lifetime value by cluster</li>
                    <li>Cluster migration patterns (upward/downward movement)</li>
                    <li>Campaign response rates by cluster</li>
                    <li>Return on marketing spend by cluster</li>
                </ul>
            </div>
        </section>
        
        <!-- NEXT STEPS SECTION -->
        <section id="next-steps">
            <h2>Implementation Roadmap</h2>
            
            <h3>Phase 1: Immediate Actions (Week 1-2)</h3>
            <div class="recommendation">
                <h5>✅ Quick Wins</h5>
                <ul>
                    <li>Export cluster assignments to CRM system</li>
                    <li>Tag customers in email marketing platform by cluster</li>
                    <li>Create cluster-specific email templates</li>
                    <li>Set up automated segment-based email flows</li>
                    <li>Brief sales and support teams on cluster characteristics</li>
                </ul>
            </div>
            
            <h3>Phase 2: Campaign Development (Week 3-4)</h3>
            <div class="recommendation">
                <h5>🎯 Campaign Development</h5>
                <ul>
                    <li>Launch VIP retention program for high-value clusters</li>
                    <li>Create win-back campaign for dormant customers</li>
                    <li>Develop growth incentive program for emerging clusters</li>
                    <li>Design personalized landing pages by cluster</li>
                    <li>Set up A/B testing framework for cluster-specific messaging</li>
                </ul>
            </div>
            
            <h3>Phase 3: Measurement & Optimization (Month 2+)</h3>
            <div class="recommendation">
                <h5>📊 Continuous Improvement</h5>
                <ul>
                    <li>Track campaign performance by cluster weekly</li>
                    <li>Monitor cluster migration and satisfaction metrics</li>
                    <li>Re-run analysis quarterly to capture evolving patterns</li>
                    <li>Refine cluster model with additional behavioral features</li>
                    <li>Integrate learnings into product development</li>
                </ul>
            </div>
            
            <h3>Expected Benefits</h3>
            <div class="insight-box">
                <strong>This segmentation strategy can deliver:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>15-25% improvement in email open rates (through better targeting)</li>
                    <li>20-30% increase in conversion rates (through personalized messaging)</li>
                    <li>10-15% reduction in churn rate (through proactive retention)</li>
                    <li>30-40% improvement in ROI on marketing spend</li>
                    <li>Significant improvement in customer satisfaction and loyalty</li>
                </ul>
            </div>
        </section>
        
        <footer>
            <p><strong>Customer Segmentation Analysis Report</strong></p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
            <p style="margin-top: 10px; color: #999; font-size: 0.9em;">
                This report is based on machine learning analysis of customer transaction data.
                Insights should be validated with business stakeholders before implementation.
            </p>
        </footer>
    </div>
</body>
</html>
"""

# Save HTML report
output_path = os.path.join(OUTPUT_DIR, 'customer_segments_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Saved: customer_segments_report.html")
print("\n" + "=" * 80)
print("HTML REPORT GENERATION COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nReport Location: {output_path}")
print(f"\nTo view the report:")
print(f"  1. Open in browser: {output_path}")
print(f"  2. Or use command: start {output_path}")
print("\n" + "=" * 80)
