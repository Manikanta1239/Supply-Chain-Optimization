# **Supply Chain Optimization Dashboard**

This project implements a Supply Chain Optimization Dashboard using Streamlit. It integrates machine learning models, optimization techniques, and data visualization tools to support key supply chain functions. 

## **Features**

- **Synthetic Data Generation:** Simulates supply chain data for products, including demand, inventory levels, costs, and lead times.
- **Demand Forecasting:** LSTM-based time-series model predicts product demand, with performance metrics like MAE and MSE.
- **Inventory Management:** Provides strategies for safety stock, reorder points, and economic order quantities (EOQ).
- **Route Optimization:** Uses linear programming to optimize delivery routes between warehouses and customers.
- **Interactive Visualization:** Dynamic charts and metrics using Plotly to analyze and interpret data patterns.

## **Tech Stack**

- **Streamlit:** For building the interactive web dashboard.
- **PyTorch:** To develop and train the demand forecasting LSTM model.
- **PuLP:** For linear programming-based route optimization.
- **Plotly:** For data visualization and graphical representation.

## **How to Run**

1. **Clone the repository and install the dependencies:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

This project provides a comprehensive toolkit for analyzing, forecasting, and optimizing supply chain operations.
