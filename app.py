import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import pulp
import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class DemandForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class SupplyChainOptimizer:
    def __init__(self):
        self.demand_model = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, num_products=5, num_months=36):
        """
        Generate synthetic supply chain dataset
        
        Columns:
        - Product ID
        - Month
        - Demand
        - Inventory Level
        - Production Cost
        - Shipping Cost
        - Lead Time
        """
        np.random.seed(42)
        data = []
        
        for product in range(num_products):
            base_demand = np.random.randint(100, 1000)
            seasonality = np.sin(np.linspace(0, 4*np.pi, num_months)) * base_demand * 0.2
            noise = np.random.normal(0, base_demand * 0.1, num_months)
            
            for month in range(num_months):
                demand = max(0, base_demand + seasonality[month] + noise[month])
                data.append({
                    'product_id': product,
                    'month': month,
                    'demand': demand,
                    'inventory_level': max(0, demand * np.random.uniform(0.8, 1.2)),
                    'production_cost': np.random.uniform(10, 100),
                    'shipping_cost': np.random.uniform(5, 50),
                    'lead_time': np.random.randint(1, 30)
                })
        
        return pd.DataFrame(data)
    
    def prepare_forecasting_data(self, data):
        """Prepare data for time series forecasting"""
        X, y = [], []
        look_back = 3
        
        for product in data['product_id'].unique():
            product_data = data[data['product_id'] == product].sort_values('month')
            
            for i in range(len(product_data) - look_back):
                X.append(product_data['demand'].values[i:i+look_back])
                y.append(product_data['demand'].values[i+look_back])
        
        X = np.array(X).reshape(-1, look_back, 1)
        y = np.array(y)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_demand_forecast_model(self, X_train, y_train, X_test, y_test):
        """Train LSTM-based demand forecasting model"""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = DemandForecastingModel(input_dim=1, hidden_dim=50, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        for epoch in range(100):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            mae = mean_absolute_error(y_test, test_predictions.numpy())
            mse = mean_squared_error(y_test, test_predictions.numpy())
        
        return model, mae, mse
    
    def route_optimization(self, warehouses, customers):
        """
        Optimize delivery routes using linear programming
        
        Args:
        warehouses (list): List of warehouse locations
        customers (list): List of customer locations
        
        Returns:
        Optimized routes minimizing total transportation cost
        """
        # Create optimization problem
        prob = pulp.LpProblem("Route_Optimization", pulp.LpMinimize)
        
        # Decision variables
        routes = pulp.LpVariable.dicts(
            "route", 
            ((w, c) for w in warehouses for c in customers), 
            cat='Binary'
        )
        
        # Objective function: Minimize total transportation cost
        prob += pulp.lpSum(
            routes[w, c] * np.random.uniform(10, 100) 
            for w in warehouses for c in customers
        )
        
        # Constraints
        # Each customer must be served by exactly one warehouse
        for c in customers:
            prob += pulp.lpSum(routes[w, c] for w in warehouses) == 1
        
        # Warehouse capacity constraints
        warehouse_capacity = {w: np.random.randint(100, 500) for w in warehouses}
        for w in warehouses:
            prob += pulp.lpSum(
                routes[w, c] * np.random.randint(10, 50) 
                for c in customers
            ) <= warehouse_capacity[w]
        
        # Solve the problem
        prob.solve()
        
        # Extract solution
        optimized_routes = [
            (w, c) for (w, c) in routes 
            if routes[w, c].varValue == 1
        ]
        
        return optimized_routes
    
    def inventory_management_strategy(self, data):
        """
        Develop inventory management strategy
        
        Metrics:
        - Safety Stock
        - Reorder Point
        - Economic Order Quantity (EOQ)
        """
        inventory_strategy = {}
        
        for product in data['product_id'].unique():
            product_data = data[data['product_id'] == product]
            
            # Calculate key metrics
            avg_demand = product_data['demand'].mean()
            demand_std = product_data['demand'].std()
            lead_time = product_data['lead_time'].mean()
            
            # Safety Stock (95% confidence interval)
            safety_stock = demand_std * 1.96 * np.sqrt(lead_time)
            
            # Reorder Point
            reorder_point = avg_demand * lead_time + safety_stock
            
            # Economic Order Quantity (EOQ)
            setup_cost = product_data['production_cost'].mean()
            holding_cost_rate = 0.2  # 20% of product cost
            eoq = np.sqrt(
                (2 * avg_demand * setup_cost) / 
                (setup_cost * holding_cost_rate)
            )
            
            inventory_strategy[product] = {
                'avg_demand': avg_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq
            }
        
        return inventory_strategy

class StreamlitSupplyChainDashboard:
    def __init__(self):
        self.optimizer = SupplyChainOptimizer()
        self.supply_chain_data = None
        
    def run(self):
        st.title("Supply Chain Optimization Dashboard")
        
        # Sidebar for navigation
        page = st.sidebar.radio(
            "Navigation", 
            [
                "Data Generation", 
                "Demand Forecasting", 
                "Inventory Management", 
                "Route Optimization"
            ]
        )
        
        if page == "Data Generation":
            self.render_data_generation()
        elif page == "Demand Forecasting":
            self.render_demand_forecasting()
        elif page == "Inventory Management":
            self.render_inventory_management()
        elif page == "Route Optimization":
            self.render_route_optimization()
    
    def render_data_generation(self):
        st.header("Supply Chain Data Generation")
        
        # Slider for data generation parameters
        num_products = st.slider("Number of Products", 1, 10, 5)
        num_months = st.slider("Number of Months", 12, 60, 36)
        
        if st.button("Generate Synthetic Data"):
            self.supply_chain_data = self.optimizer.generate_synthetic_data(
                num_products, num_months
            )
            
            # Display data summary
            st.dataframe(self.supply_chain_data.groupby('product_id').agg({
                'demand': ['mean', 'std'],
                'inventory_level': ['mean', 'min', 'max']
            }))
            
            # Visualize demand patterns
            fig = px.line(
                self.supply_chain_data, 
                x='month', 
                y='demand', 
                color='product_id', 
                title='Demand Patterns by Product'
            )
            st.plotly_chart(fig)
    
    def render_demand_forecasting(self):
        st.header("Demand Forecasting")
        
        if self.supply_chain_data is None:
            st.warning("Please generate data first")
            return
        
        # Prepare data for forecasting
        X_train, X_test, y_train, y_test = self.optimizer.prepare_forecasting_data(
            self.supply_chain_data
        )
        
        # Train model
        model, mae, mse = self.optimizer.train_demand_forecast_model(
            X_train, y_train, X_test, y_test
        )
        
        st.metric("Mean Absolute Error", f"{mae:.2f}")
        st.metric("Mean Squared Error", f"{mse:.2f}")
        
        # Prediction visualization
        st.subheader("Forecasting Performance")
        predictions = model(torch.FloatTensor(X_test)).detach().numpy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=y_test, 
            mode='lines', 
            name='Actual Demand'
        ))
        fig.add_trace(go.Scatter(
            y=predictions.flatten(), 
            mode='lines', 
            name='Predicted Demand'
        ))
        st.plotly_chart(fig)
    
    def render_inventory_management(self):
        st.header("Inventory Management Strategy")
        
        if self.supply_chain_data is None:
            st.warning("Please generate data first")
            return
        
        # Calculate inventory strategy
        inventory_strategy = self.optimizer.inventory_management_strategy(
            self.supply_chain_data
        )
        
        # Display strategies
        for product, strategy in inventory_strategy.items():
            st.subheader(f"Product {product}")
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Avg Demand", f"{strategy['avg_demand']:.2f}")
            
            with cols[1]:
                st.metric("Safety Stock", f"{strategy['safety_stock']:.2f}")
            
            with cols[2]:
                st.metric("Reorder Point", f"{strategy['reorder_point']:.2f}")
            
            with cols[3]:
                st.metric("EOQ", f"{strategy['economic_order_quantity']:.2f}")
    
    def render_route_optimization(self):
        st.header("Route Optimization")
        
        # Generate sample warehouses and customers
        warehouses = [f"WH{i}" for i in range(3)]
        customers = [f"Customer{i}" for i in range(10)]
        
        if st.button("Optimize Routes"):
            optimized_routes = self.optimizer.route_optimization(
                warehouses, customers
            )
            
            # Visualize routes
            route_df = pd.DataFrame(
                optimized_routes, 
                columns=['Warehouse', 'Customer']
            )
            st.dataframe(route_df)
            
            # Create route distribution visualization
            route_counts = route_df['Warehouse'].value_counts()
            fig = px.pie(
                values=route_counts.values, 
                names=route_counts.index, 
                title='Route Distribution Across Warehouses'
            )
            st.plotly_chart(fig)

def main():
    dashboard = StreamlitSupplyChainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
