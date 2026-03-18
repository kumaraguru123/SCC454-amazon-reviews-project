# SCC454-amazon-reviews-project
 Amazon Reviews 2023: databases, similarity, clustering and recommendation system.
 
📌 Overview

This project presents a complete data-driven pipeline for analyzing e-commerce product data and building an intelligent recommendation system. The system processes raw data, extracts meaningful features, applies clustering techniques, performs similarity analysis, and generates personalized recommendations.

The project demonstrates the application of machine learning techniques in real-world scenarios such as product grouping, user segmentation, and recommendation systems.

🎯 Objectives

Clean and preprocess raw e-commerce data

Perform feature engineering for users and products

Apply clustering algorithms to identify patterns

Compute similarity between products and users

Build a recommendation system for personalized suggestions

📂 Dataset

The cleaned dataset, intermediate outputs, and results are available on Hugging Face:

👉 https://huggingface.co/datasets/KUMARAGURU0504/SCC_454

Dataset Includes:

Cleaned product data

User interaction data

Feature-engineered datasets

Clustering results

Similarity outputs

⚙️ Project Pipeline
🔹 1. Data Preprocessing

Data cleaning and filtering

Handling missing values

Creating a clean subset

📁 src/data/

🔹 2. Feature Engineering

User behaviour features

Product features (including TF-IDF for text data)

📁 src/features/

🔹 3. Clustering
Product Clustering:

K-Means

DBSCAN

TF-IDF + K-Means

User Clustering:

Agglomerative Clustering

📁 src/clustering/

🔹 4. Similarity Analysis

Product similarity computation

Benchmarking similarity methods

📁 src/similarity/

🔹 5. Recommendation System

Collaborative filtering

Similarity-based recommendation

📁 src/recommendation/

🔹 6. Database Integration

SQLite

DuckDB

MongoDB Atlas

📁 src/database/

🔹 7. Visualization

Graphs and plots for evaluation

📁 src/visualization/

📁 Project Structure
SCC454-amazon-reviews-project/
│
├── README.md
├── src/
│   ├── data/
│   ├── features/
│   ├── clustering/
│   ├── similarity/
│   ├── recommendation/
│   ├── database/
│   ├── visualization/
│   └── utils/
│
├── data/
├── outputs/
├── reports/
🚀 How to Run
1. Clone the Repository
git clone https://github.com/kumaraguru123/SCC454-amazon-reviews-project
cd SCC454-amazon-reviews-project
2. Install Dependencies
pip install -r requirements.txt
3. Run the Pipeline
python src/data/build_clean_subset.py
python src/features/build_user_features.py
python src/features/build_product_features.py
python src/clustering/product_clustering_kmeans.py
python src/similarity/task2_similarity_all.py
python src/recommendation/task4_recommendation.py
📊 Techniques Used
Machine Learning

K-Means Clustering

DBSCAN

Agglomerative Clustering

Text Processing

TF-IDF Vectorization

Recommendation Systems

Collaborative Filtering

Similarity-Based Filtering

Databases

SQLite

DuckDB

MongoDB Atlas

📈 Results

Effective clustering of products and users

Improved recommendation accuracy

Scalable and modular pipeline

All results are available on Hugging Face:
👉 https://huggingface.co/datasets/KUMARAGURU0504/SCC_454

🔮 Future Work

Deep learning-based recommendation models

Real-time recommendation system

Web-based interface for users

Advanced evaluation metrics

👨‍💻 Author

Kumaraguru
B.Sc Computer Technology
Final Year Project

📄 License

This project is developed for academic purposes.
