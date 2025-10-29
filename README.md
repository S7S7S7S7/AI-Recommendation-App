# 🤖 AI-Driven Recommendation System

![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![ML](https://img.shields.io/badge/Machine%20Learning-Enabled-yellow)

---

###  [Watch Video] :- https://www.loom.com/share/3512bd900f8e4bf7a2c5bb21495d7d87

### 🚀 Overview

The **AI-Driven Recommendation System** is an interactive web app built with **Streamlit** that helps analyze customer purchase patterns, predict next likely items, and perform **sentiment analysis** on customer reviews.

This project demonstrates **multiple recommendation and AI algorithms** integrated into a single unified platform — useful for retail analytics, product recommendations, and customer behavior prediction.

---

### 🎯 Key Features

✅ **Association Rule Mining**
- Implements **Apriori** and **FP-Growth** algorithms to uncover product co-purchase patterns.  
- Generates rules like *“If Milk → Bread”* based on frequent itemsets.

✅ **Sequential Pattern Mining**
- Learns **purchase sequences over time** using n-gram transitions.  
- Predicts the *next likely item* a user will buy, based on purchase history.

✅ **Sentiment Analysis (LSTM)**
- Uses **Deep Learning (LSTM)** to analyze text reviews.  
- Classifies sentiments as *Positive*, *Negative*, or *Neutral* with confidence scores.

✅ **Data Exploration (EDA)**
- Provides **interactive visualizations** including histograms, heatmaps, line charts, bar graphs, and pie charts.

✅ **Smart Preprocessing**
- Automatically handles missing values, encodes categorical data, removes outliers, and supports transaction-based data for Apriori/FP-Growth.

---

### 🧩 Algorithms Implemented

| Algorithm | Purpose | Type |
|------------|----------|------|
| **Apriori** | Discover frequent itemsets | Association Rule Mining |
| **FP-Growth** | Fast frequent itemset generation | Association Rule Mining |
| **Sequential Pattern Matching** | Predict next likely item | Sequence Mining |
| **LSTM** | Analyze text sentiment | Deep Learning |


# 💡 What’s Next
- Fixing dependency compatibility for deployment
- Adding chatbot integration module
