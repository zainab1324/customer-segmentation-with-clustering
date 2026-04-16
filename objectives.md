# Sales Forecasting with Time Series

## Overview
- Segment customers of an e-commerce site using purchasing data. 
- This project demonstrates how to analyze customer behavior and group similar customers together for more effective marketing strategies.

## Objectives
- Uncover distinct customer groups for targeted marketing, 
- personalize customer experiences, and 
- optimize product recommendations based on behavioral patterns.

## Methodology
- K-Means clustering algorithm
- Principal Component Analysis (PCA) for dimensionality reduction
- RFM (Recency, Frequency, Monetary) analysis
- Customer behavior feature extraction

## Dataset Description
- You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

### File descriptions
- sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
- test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
- sample_submission.csv - a sample submission file in the correct format.
- items.csv - supplemental information about the items/products.
- item_categories.csv  - supplemental information about the items categories.
- shops.csv- supplemental information about the shops.