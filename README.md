# 📈 Stock Data Analysis Web  

## 🔥 Overview  
This project is a **full-stack data analytics web application** that pulls stock data from **TradingView**, processes it using **AWS Glue ETL**, stores it in **Amazon S3**, and then visualizes it through a **Streamlit-based web interface** with interactive **Plotly charts**.  

The app provides traders, analysts, and investors with **clean, labeled, and real-time visualizations** of stock market data.  

---

## ⚙️ Architecture Flow  

1. **TradingView API → Data Ingestion**  
   - Stock market data is pulled via **API calls**.  

2. **AWS Glue ETL → Data Processing**  
   - Data is cleaned, labeled, and transformed.  
   - Processed data is saved into **S3 buckets**.  

3. **Amazon S3 → Data Lake**  
   - Serves as the central storage for all transformed stock data.  

4. **Boto3 → Data Access Layer**  
   - Streamlit app retrieves processed data from S3.  

5. **Streamlit + Plotly → Web App**  
   - Provides a **UI dashboard** with interactive charts.  
   - Allows end-users to explore stock price trends, volume, and custom metrics.  

---

## 📂 Project Structure  

