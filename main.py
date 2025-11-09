"""
Supply Chain Risk Modeling Competition
========================================
A modular, class-based implementation for modeling supply chain risks using
GNN embeddings + Gradient Boosting (LightGBM/CatBoost).

Author: Competition Team
Date: November 2025
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
import lightgbm as lgb
import catboost as cb
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class PathHandler:
    """Central path management for all I/O operations."""
    data: str = "data"
    cleaned: str = "data/cleaned"
    output: str = "output"
    reports: str = "reports"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data, self.cleaned, self.output, self.reports]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    @property
    def risk_raw(self) -> str:
        return os.path.join(self.data, "supply_chain_risk.csv")
    
    @property
    def resilience_raw(self) -> str:
        return os.path.join(self.data, "supply_chain_resilience.csv")
    
    @property
    def risk_cleaned(self) -> str:
        return os.path.join(self.cleaned, "risk_cleaned.csv")
    
    @property
    def resilience_cleaned(self) -> str:
        return os.path.join(self.cleaned, "resilience_cleaned.csv")


@dataclass
class HyperParameters:
    """Centralized hyperparameter management."""
    
    # Data Splitting
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    
    # GNN Architecture
    gnn_hidden_channels: int = 64
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.3
    gnn_heads: int = 4  # For GAT
    
    # GNN Training
    gnn_epochs: int = 100
    gnn_lr: float = 0.001
    gnn_weight_decay: float = 5e-4
    gnn_patience: int = 15
    
    # LightGBM Parameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 500,
        'random_state': 42
    })
    
    lgb_regression_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 500,
        'random_state': 42
    })
    
    # CatBoost Parameters
    catboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': False
    })
    
    catboost_regression_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    })
    
    # Visualization
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataLoader:
    """Handles loading and initial validation of datasets."""
    
    def __init__(self, paths: PathHandler):
        self.paths = paths
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both risk and resilience datasets."""
        print("Loading datasets...")
        
        risk_df = pd.read_csv(self.paths.risk_raw)
        resilience_df = pd.read_csv(self.paths.resilience_raw)
        
        print(f"Risk dataset shape: {risk_df.shape}")
        print(f"Resilience dataset shape: {resilience_df.shape}")
        
        return risk_df, resilience_df


class DataPreprocessor:
    """Comprehensive data cleaning and feature engineering."""
    
    def __init__(self, paths: PathHandler):
        self.paths = paths
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_risk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess risk dataset."""
        df = df.copy()
        
        # Handle missing values
        df['manual_risk_label'].fillna('Medium', inplace=True)
        df['actual_delivery_date'].fillna(df['expected_delivery_date'], inplace=True)
        df['supplier_rating'].fillna(df['supplier_rating'].median(), inplace=True)
        
        # Convert timestamps
        for col in ['timestamp', 'order_placed_date', 'expected_delivery_date', 'actual_delivery_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Feature Engineering
        df['delivery_delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
        df['order_to_delivery_days'] = (df['actual_delivery_date'] - df['order_placed_date']).dt.days
        df['is_delayed'] = (df['delivery_delay_days'] > 0).astype(int)
        
        # Temperature risk zones
        df['temp_risk'] = pd.cut(df['temperature'], bins=[-np.inf, 0, 30, np.inf], 
                                  labels=['Low', 'Medium', 'High'])
        
        # Vibration risk
        df['vibration_risk'] = pd.cut(df['vibration_level'], 
                                       bins=[-np.inf, df['vibration_level'].quantile(0.33), 
                                             df['vibration_level'].quantile(0.66), np.inf],
                                       labels=['Low', 'Medium', 'High'])
        
        # Inventory risk
        df['inventory_risk'] = df['inventory_status'].map({
            'In Stock': 'Low',
            'Low Stock': 'Medium',
            'Out of Stock': 'High'
        })
        
        # Text feature lengths
        df['social_media_length'] = df['social_media_feed'].fillna('').str.len()
        df['news_alert_length'] = df['news_alert'].fillna('').str.len()
        df['system_log_length'] = df['system_log_message'].fillna('').str.len()
        
        # Drop original timestamp columns after feature extraction
        df.drop(columns=['timestamp', 'order_placed_date', 'expected_delivery_date', 
                        'actual_delivery_date', 'social_media_feed', 'news_alert', 
                        'system_log_message'], inplace=True, errors='ignore')
        
        return df
    
    def clean_resilience_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess resilience dataset."""
        df = df.copy()
        
        # Handle missing values
        df['Disruption_Type'].fillna('None', inplace=True)
        # Convert severity to numeric, forcing text values like 'Low' to NaN
        df['Disruption_Severity'] = pd.to_numeric(df['Disruption_Severity'], errors='coerce')

# Now, fill all NaNs (original ones and the new text-based ones) with 0
        df['Disruption_Severity'].fillna(0, inplace=True)
        df['Supplier_Reliability_Score'].fillna(df['Supplier_Reliability_Score'].median(), inplace=True)
        
        # Convert dates
        for col in ['Order_Date', 'Dispatch_Date', 'Delivery_Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Feature Engineering
        df['processing_time_days'] = (df['Dispatch_Date'] - df['Order_Date']).dt.days
        df['transit_time_days'] = (df['Delivery_Date'] - df['Dispatch_Date']).dt.days
        df['total_lead_time'] = (df['Delivery_Date'] - df['Order_Date']).dt.days
        
        # Resilience Score (target variable)
        df['resilience_score'] = (
            (100 - df['Delay_Days'].clip(0, 30) * 2) * 0.3 +
            df['Supplier_Reliability_Score'] * 0.4 +
            (100 - df['Historical_Disruption_Count'].clip(0, 20) * 3) * 0.3
        ).clip(0, 100)
        
        # Resilience Label
        df['resilience_label'] = pd.cut(df['resilience_score'], 
                                        bins=[0, 40, 70, 100],
                                        labels=['Low', 'Medium', 'High'])
        
        # Risk Score (severity-based)
        df['risk_score'] = (
            df['Delay_Days'] * 0.4 +
            df['Disruption_Severity'] * 0.3 +
            df['Historical_Disruption_Count'] * 0.2 +
            (100 - df['Supplier_Reliability_Score']) * 0.1
        )
        
        # Drop date columns after feature extraction
        df.drop(columns=['Order_Date', 'Dispatch_Date', 'Delivery_Date'], 
                inplace=True, errors='ignore')
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in ['manual_risk_label', 'resilience_label']:  # Don't encode targets
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], method='clip') -> pd.DataFrame:
        """Handle outliers using IQR method."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if method == 'clip':
                    df[col] = df[col].clip(lower_bound, upper_bound)
                elif method == 'remove':
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def process_and_save(self, risk_df: pd.DataFrame, resilience_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete preprocessing pipeline."""
        print("\nPreprocessing datasets...")
        
        # Clean datasets
        risk_clean = self.clean_risk_data(risk_df)
        resilience_clean = self.clean_resilience_data(resilience_df)
        
        # Handle outliers
        numeric_cols = risk_clean.select_dtypes(include=[np.number]).columns.tolist()
        risk_clean = self.handle_outliers(risk_clean, numeric_cols)
        
        numeric_cols = resilience_clean.select_dtypes(include=[np.number]).columns.tolist()
        resilience_clean = self.handle_outliers(resilience_clean, numeric_cols)
        
        # Encode categorical
        risk_clean = self.encode_categorical(risk_clean)
        resilience_clean = self.encode_categorical(resilience_clean)
        
        # Save cleaned data
        risk_clean.to_csv(self.paths.risk_cleaned, index=False)
        resilience_clean.to_csv(self.paths.resilience_cleaned, index=False)
        
        print(f"Cleaned risk data saved: {risk_clean.shape}")
        print(f"Cleaned resilience data saved: {resilience_clean.shape}")
        
        return risk_clean, resilience_clean


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

class EDAVisualizer:
    """Generate comprehensive EDA reports and visualizations."""
    
    def __init__(self, paths: PathHandler, params: HyperParameters):
        self.paths = paths
        self.params = params
    
    def generate_summary_statistics(self, df: pd.DataFrame, name: str):
        """Generate and save summary statistics."""
        summary = df.describe(include='all').T
        summary.to_csv(os.path.join(self.paths.reports, f'{name}_summary_stats.csv'))
        
        print(f"\n{name} Summary Statistics:")
        print(summary)
    
    def plot_distribution(self, df: pd.DataFrame, columns: List[str], name: str):
        """Plot distribution of key features."""
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    df[col].value_counts().plot(kind='bar', ax=axes[idx])
                else:
                    df[col].hist(bins=30, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.output, f'{name}_distributions.png'), 
                   dpi=self.params.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, name: str):
        """Plot correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 0:
            plt.figure(figsize=(14, 12))
            correlation = numeric_df.corr()
            sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title(f'{name} - Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.paths.output, f'{name}_correlation.png'),
                       dpi=self.params.dpi, bbox_inches='tight')
            plt.close()
            
            # Save correlation matrix
            correlation.to_csv(os.path.join(self.paths.reports, f'{name}_correlation.csv'))
    
    def calculate_vif(self, df: pd.DataFrame, name: str):
        """Calculate Variance Inflation Factor for multicollinearity check."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2:
            print(f"Not enough numeric columns for VIF calculation in {name}")
            return
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                          for i in range(len(numeric_df.columns))]
        
        vif_data = vif_data.sort_values('VIF', ascending=False)
        vif_data.to_csv(os.path.join(self.paths.reports, f'{name}_vif_audit.csv'), index=False)
        
        print(f"\n{name} - VIF Analysis (Top 10):")
        print(vif_data.head(10))
    
    def check_data_leakage(self, df: pd.DataFrame, target_col: str, name: str):
        """Check for potential data leakage."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Flag high correlations (> 0.95) as potential leakage
            leakage_suspects = correlations[correlations > 0.95].drop(target_col, errors='ignore')
            
            leakage_report = pd.DataFrame({
                'Feature': leakage_suspects.index,
                'Correlation': leakage_suspects.values,
                'Potential_Leakage': 'High Risk'
            })
            
            leakage_report.to_csv(os.path.join(self.paths.reports, f'{name}_leakage_audit.csv'), index=False)
            
            print(f"\n{name} - Potential Data Leakage:")
            print(leakage_report)
    
    def run_full_eda(self, risk_df: pd.DataFrame, resilience_df: pd.DataFrame):
        """Run complete EDA pipeline."""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Risk dataset
        self.generate_summary_statistics(risk_df, 'risk')
        self.plot_distribution(risk_df, risk_df.select_dtypes(include=[np.number]).columns[:9].tolist(), 'risk')
        self.plot_correlation_matrix(risk_df, 'risk')
        self.calculate_vif(risk_df, 'risk')
        
        # Resilience dataset
        self.generate_summary_statistics(resilience_df, 'resilience')
        self.plot_distribution(resilience_df, resilience_df.select_dtypes(include=[np.number]).columns[:9].tolist(), 'resilience')
        self.plot_correlation_matrix(resilience_df, 'resilience')
        self.calculate_vif(resilience_df, 'resilience')


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

class GraphBuilder:
    """Build heterogeneous graph from supply chain data."""
    
    def __init__(self, params: HyperParameters):
        self.params = params
        self.node_mappings = {}
    
    def build_graph_from_resilience(self, df: pd.DataFrame) -> HeteroData:
        """Build heterogeneous graph from resilience dataset."""
        print("\nBuilding heterogeneous graph...")
        
        data = HeteroData()
        
        # --- Create node mappings ---
        # Ensure only non-null, unique IDs are used
        suppliers = df['Supplier_ID'].dropna().unique()
        buyers = df['Buyer_ID'].dropna().unique()
        products = df['Product_Category'].dropna().unique()
        
        self.node_mappings = {
            'supplier': {sid: idx for idx, sid in enumerate(suppliers)},
            'buyer': {bid: idx for idx, bid in enumerate(buyers)},
            'product': {pid: idx for idx, pid in enumerate(products)}
        }
        
        # --- Supplier node features ---
        supplier_features_df = df.groupby('Supplier_ID').agg({
            'Supplier_Reliability_Score': 'mean',
            'Historical_Disruption_Count': 'sum',
            'Delay_Days': 'mean',
            'Order_Value_USD': 'sum'
        }).reindex(suppliers).fillna(0) # Ensure order matches mapping
        
        data['supplier'].x = torch.tensor(supplier_features_df.values, dtype=torch.float)
        
        # --- Buyer node features ---
        buyer_features_df = df.groupby('Buyer_ID').agg({
            'Order_Value_USD': 'sum',
            'Delay_Days': 'mean',
            'Quantity_Ordered': 'sum'
        }).reindex(buyers).fillna(0) # Ensure order matches mapping
        
        data['buyer'].x = torch.tensor(buyer_features_df.values, dtype=torch.float)
        
        # --- Product node features ---
        product_features_df = df.groupby('Product_Category').agg({
            'Order_Value_USD': 'sum',
            'Quantity_Ordered': 'sum',
            'Delay_Days': 'mean'
        }).reindex(products).fillna(0) # Ensure order matches mapping
        
        data['product'].x = torch.tensor(product_features_df.values, dtype=torch.float)
        
        # --- Build edges: supplier -> buyer, buyer -> product ---
        supplier_buyer_edges = []
        buyer_product_edges = []
        
        for _, row in df.iterrows():
            # Check if all entities for this row exist in our mappings
            s_id = row['Supplier_ID']
            b_id = row['Buyer_ID']
            p_id = row['Product_Category']
            
            if s_id in self.node_mappings['supplier'] and \
               b_id in self.node_mappings['buyer'] and \
               p_id in self.node_mappings['product']:
                
                s_idx = self.node_mappings['supplier'][s_id]
                b_idx = self.node_mappings['buyer'][b_id]
                p_idx = self.node_mappings['product'][p_id]
                
                supplier_buyer_edges.append([s_idx, b_idx])
                buyer_product_edges.append([b_idx, p_idx])
        
        # --- Create edge_index tensors (robustly handles empty lists) ---
        
        # Build edges: supplier -> buyer
        if supplier_buyer_edges:
            sb_edge_index = torch.tensor(supplier_buyer_edges, dtype=torch.long).T
        else:
            sb_edge_index = torch.empty((2, 0), dtype=torch.long)
        data['supplier', 'supplies', 'buyer'].edge_index = sb_edge_index

        # Build edges: buyer -> product
        if buyer_product_edges:
            bp_edge_index = torch.tensor(buyer_product_edges, dtype=torch.long).T
        else:
            bp_edge_index = torch.empty((2, 0), dtype=torch.long)
        data['buyer', 'orders', 'product'].edge_index = bp_edge_index
        
        print(f"Graph built: {data}")
        return data

# ============================================================================
# GNN MODEL
# ============================================================================

from torch_geometric.nn import HeteroConv  # Make sure to import HeteroConv

# ============================================================================
# GNN MODEL
# ============================================================================

from torch_geometric.nn import HeteroConv  # Make sure to import HeteroConv

# ============================================================================
# GNN MODEL
# ============================================================================

class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for supply chain.
    
    This version uses HeteroConv and handles input projections
    and source-only nodes correctly.
    """
    
    def __init__(self, metadata, hidden_channels, num_layers, dropout, graph_data: HeteroData):
        super().__init__()
        
        # Create input projection layers for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in graph_data.node_types:
            in_channels = graph_data[node_type].x.shape[1]
            self.lin_dict[node_type] = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)
            
            h_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(h_conv)
            
        self.dropout = dropout
    
    def forward(self, x_dict, edge_index_dict):
        # Apply initial linear projection and ReLU
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }
        
        # Store this for the reconstruction loss
        self.projected_features = x_dict 

        for conv in self.convs:
            # Store the input features for this layer
            x_in = x_dict 
            
            # Pass the data through the HeteroConv layer
            # This returns a dict with only destination nodes (e.g., 'buyer', 'product')
            x_out = conv(x_in, edge_index_dict) 
            
            # Rebuild the feature dictionary for the next layer
            x_dict = {}
            for node_type in x_in.keys():
                if node_type in x_out:
                    # This node was a destination, use its updated features
                    x_dict[node_type] = x_out[node_type]
                else:
                    # This node was only a source, pass its features through
                    x_dict[node_type] = x_in[node_type]

            # Apply ReLU and dropout to all node types
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}
                      
        return x_dict

class GNNEmbeddingGenerator:
    """Train GNN and generate node embeddings."""
    
    def __init__(self, params: HyperParameters, paths: PathHandler):
        self.params = params
        self.paths = paths
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_graph_data(self, graph_data: HeteroData) -> HeteroData:
        """Prepare graph data by moving it to the device."""
        # The projection logic is now handled by the HeteroGNN model itself
        return graph_data.to(self.device)
    
    def train_gnn(self, graph_data: HeteroData) -> Dict[str, np.ndarray]:
        """Train GNN and extract embeddings."""
        print("\nTraining GNN...")
        
        # 1. Prepare data (just moves to device)
        graph_data = self.prepare_graph_data(graph_data)
        
        # 2. Initialize model (pass graph_data for projections)
        self.model = HeteroGNN(
            graph_data.metadata(),
            self.params.gnn_hidden_channels,
            self.params.gnn_num_layers,
            self.params.gnn_dropout,
            graph_data  # Pass the graph data
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.gnn_lr,
            weight_decay=self.params.gnn_weight_decay
        )
        
        self.model.train()
        for epoch in range(self.params.gnn_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(graph_data.x_dict, graph_data.edge_index_dict)
            
            # Reconstruction loss against the *projected* features (from the 1st layer)
            loss = sum([F.mse_loss(out[node_type], self.model.projected_features[node_type]) 
                        for node_type in graph_data.node_types])
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.params.gnn_epochs}, Loss: {loss.item():.4f}")
        
        # Extract embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(graph_data.x_dict, graph_data.edge_index_dict)
        
        # Convert to numpy
        embeddings_np = {
            node_type: emb.cpu().numpy() 
            for node_type, emb in embeddings.items()
        }
        
        print("GNN training complete!")
        return embeddings_np
    
    def visualize_embeddings(self, embeddings: Dict[str, np.ndarray], method='tsne'):
        """Visualize node embeddings using t-SNE or PCA."""
        print(f"\nVisualizing embeddings using {method.upper()}...")
        
        fig, axes = plt.subplots(1, len(embeddings), figsize=(15, 5))
        if len(embeddings) == 1:
            axes = [axes]
        
        for idx, (node_type, emb) in enumerate(embeddings.items()):
            if method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=min(30, len(emb)-1),
                               max_iter=self.params.tsne_n_iter, random_state=self.params.random_state)
            else:
                reducer = PCA(n_components=2)
            
            emb_2d = reducer.fit_transform(emb)
            
            axes[idx].scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6)
            axes[idx].set_title(f'{node_type.capitalize()} Embeddings ({method.upper()})')
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.output, f'gnn_embeddings_{method}.png'),
                      dpi=self.params.dpi, bbox_inches='tight')
        plt.close()


# ============================================================================
# HYBRID FEATURE FUSION
# ============================================================================

class FeatureFusion:
    """Combine GNN embeddings with tabular features."""
    
    def __init__(self, node_mappings: Dict):
        self.node_mappings = node_mappings
    
    def fuse_features(self, df: pd.DataFrame, embeddings: Dict[str, np.ndarray],
                     node_col: str, node_type: str) -> pd.DataFrame:
        """Fuse GNN embeddings with tabular features."""
        df = df.copy()
        
        if node_type not in embeddings:
            print(f"Warning: {node_type} not found in embeddings")
            return df
        
        # Map node IDs to embedding indices
        node_to_idx = self.node_mappings.get(node_type, {})
        
        # Create embedding columns
        embedding_dim = embeddings[node_type].shape[1]
        embedding_cols = [f'gnn_emb_{i}' for i in range(embedding_dim)]
        
        # Map embeddings to dataframe rows
        embedding_matrix = np.zeros((len(df), embedding_dim))
        
        for idx, node_id in enumerate(df[node_col]):
            if node_id in node_to_idx:
                emb_idx = node_to_idx[node_id]
                embedding_matrix[idx] = embeddings[node_type][emb_idx]
        
        # Add embeddings as new columns
        for i, col in enumerate(embedding_cols):
            df[col] = embedding_matrix[:, i]
        
        print(f"Fused {embedding_dim} GNN features with {len(df.columns) - embedding_dim} tabular features")
        return df


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

class ModelTrainer:
    """Train and evaluate gradient boosting models."""
    
    def __init__(self, params: HyperParameters, paths: PathHandler):
        self.params = params
        self.paths = paths
        self.models = {}
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, task_type: str):
        """Prepare train/val/test splits."""
        
        # Define ALL columns that should NOT be features
        # This includes all target-like or leaky columns
        cols_to_drop = ['risk_score', 'resilience_score', 'resilience_label', 'manual_risk_label']

        # Separate features and target
        # Drop all target-like columns from X
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = df[target_col].copy()
        
        # Encode target if classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.target_encoder = le
        
        # Train/temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.params.test_size + self.params.val_size,
            random_state=self.params.random_state,
            stratify=y if task_type == 'classification' else None
        )
        
        # Val/Test split
        val_ratio = self.params.val_size / (self.params.test_size + self.params.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio,
            random_state=self.params.random_state,
            stratify=y_temp if task_type == 'classification' else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_lightgbm_classifier(self, X_train, X_val, y_train, y_val, name: str):
        """Train LightGBM classifier."""
        print(f"\nTraining LightGBM Classifier: {name}")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            self.params.lgb_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
        )
        
        self.models[f'lgb_clf_{name}'] = model
        return model
    
    def train_lightgbm_regressor(self, X_train, X_val, y_train, y_val, name: str):
        """Train LightGBM regressor."""
        print(f"\nTraining LightGBM Regressor: {name}")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            self.params.lgb_regression_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
        )
        
        self.models[f'lgb_reg_{name}'] = model
        return model
    
    def train_catboost_classifier(self, X_train, X_val, y_train, y_val, name: str):
        """Train CatBoost classifier."""
        print(f"\nTraining CatBoost Classifier: {name}")
        
        model = cb.CatBoostClassifier(**self.params.catboost_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=50
        )
        
        self.models[f'cb_clf_{name}'] = model
        return model
    
    def train_catboost_regressor(self, X_train, X_val, y_train, y_val, name: str):
        """Train CatBoost regressor."""
        print(f"\nTraining CatBoost Regressor: {name}")
        
        model = cb.CatBoostRegressor(**self.params.catboost_regression_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=50
        )
        
        self.models[f'cb_reg_{name}'] = model
        return model
    
    def evaluate_classifier(self, model, X_test, y_test, name: str, model_type: str):
        """Evaluate classification model."""
        print(f"\nEvaluating {model_type} Classifier: {name}")
        
        if model_type == 'lgb':
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:  # catboost
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # AUC (if multiclass)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            auc = None
        
        results = {
            'Model': f'{model_type.upper()}_{name}',
            'Task': 'Classification',
            'Accuracy': accuracy,
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted,
            'AUC': auc
        }
        
        self.results[f'{model_type}_clf_{name}'] = results
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        if auc:
            print(f"AUC: {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_type.upper()} - {name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.reports, f'{model_type}_clf_{name}_confusion_matrix.png'),
                   dpi=self.params.dpi, bbox_inches='tight')
        plt.close()
        
        # Classification Report
        if hasattr(self, 'target_encoder'):
            target_names = self.target_encoder.classes_
        else:
            target_names = None
        
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).T
        report_df.to_csv(os.path.join(self.paths.reports, f'{model_type}_clf_{name}_classification_report.csv'))
        
        return results
    
    def evaluate_regressor(self, model, X_test, y_test, name: str, model_type: str):
        """Evaluate regression model."""
        print(f"\nEvaluating {model_type} Regressor: {name}")
        
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'Model': f'{model_type.upper()}_{name}',
            'Task': 'Regression',
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        self.results[f'{model_type}_reg_{name}'] = results
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Prediction vs Actual Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_type.upper()} - {name} - Predictions vs Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.reports, f'{model_type}_reg_{name}_predictions.png'),
                   dpi=self.params.dpi, bbox_inches='tight')
        plt.close()
        
        # Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_type.upper()} - {name} - Residual Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.reports, f'{model_type}_reg_{name}_residuals.png'),
                   dpi=self.params.dpi, bbox_inches='tight')
        plt.close()
        
        return results
    
    def plot_feature_importance(self, model, feature_names: List[str], name: str, model_type: str):
        """Plot feature importance."""
        if model_type == 'lgb':
            importance = model.feature_importance(importance_type='gain')
        else:  # catboost
            importance = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title(f'{model_type.upper()} - {name} - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths.output, f'{model_type}_{name}_feature_importance.png'),
                   dpi=self.params.dpi, bbox_inches='tight')
        plt.close()
        
        feature_importance_df.to_csv(os.path.join(self.paths.reports, f'{model_type}_{name}_feature_importance.csv'),
                                    index=False)
    
    def save_results_summary(self):
        """Save all results to CSV."""
        results_df = pd.DataFrame(self.results.values())
        results_df.to_csv(os.path.join(self.paths.reports, 'model_comparison_summary.csv'), index=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class SupplyChainRiskPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, paths: PathHandler, params: HyperParameters):
        self.paths = paths
        self.params = params
        
        # Initialize components
        self.data_loader = DataLoader(paths)
        self.preprocessor = DataPreprocessor(paths)
        self.eda_visualizer = EDAVisualizer(paths, params)
        self.graph_builder = GraphBuilder(params)
        self.gnn_generator = GNNEmbeddingGenerator(params, paths)
        self.trainer = ModelTrainer(params, paths)
    
    def run_full_pipeline(self):
        """Execute complete pipeline."""
        print("\n" + "="*80)
        print("SUPPLY CHAIN RISK MODELING PIPELINE")
        print("="*80)
        
        # Step 1: Load Data
        print("\n[STEP 1] Loading datasets...")
        risk_df, resilience_df = self.data_loader.load_datasets()
        
        # Step 2: Preprocessing
        print("\n[STEP 2] Preprocessing...")
        risk_clean, resilience_clean = self.preprocessor.process_and_save(risk_df, resilience_df)
        
        # Step 3: EDA
        print("\n[STEP 3] Exploratory Data Analysis...")
        self.eda_visualizer.run_full_eda(risk_clean, resilience_clean)
        
        # Step 4: Build Graph
        print("\n[STEP 4] Building Graph...")
        graph_data = self.graph_builder.build_graph_from_resilience(resilience_clean)
        
        # Step 5: Train GNN and Generate Embeddings
        print("\n[STEP 5] Training GNN...")
        embeddings = self.gnn_generator.train_gnn(graph_data)
        self.gnn_generator.visualize_embeddings(embeddings, method='tsne')
        self.gnn_generator.visualize_embeddings(embeddings, method='pca')
        
        # Step 6: Feature Fusion
        print("\n[STEP 6] Fusing GNN embeddings with tabular features...")
        fusion = FeatureFusion(self.graph_builder.node_mappings)
        
        # Fuse for resilience dataset (using Supplier_ID)
        resilience_fused = fusion.fuse_features(
            resilience_clean, embeddings, 'Supplier_ID', 'supplier'
        )
        
        # Step 7: Model Training & Evaluation
        print("\n[STEP 7] Training and Evaluating Models...")
        
        # Task 1: Risk Label Classification (Resilience dataset)
        if 'resilience_label' in resilience_fused.columns:
            print("\n>>> TASK 1: Resilience Label Classification <<<")
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
                resilience_fused, 'resilience_label', 'classification'
            )
            
            # LightGBM
            lgb_clf = self.trainer.train_lightgbm_classifier(X_train, X_val, y_train, y_val, 'resilience_label')
            self.trainer.evaluate_classifier(lgb_clf, X_test, y_test, 'resilience_label', 'lgb')
            self.trainer.plot_feature_importance(lgb_clf, X_train.columns.tolist(), 'resilience_label_clf', 'lgb')
            
            # CatBoost
            cb_clf = self.trainer.train_catboost_classifier(X_train, X_val, y_train, y_val, 'resilience_label')
            self.trainer.evaluate_classifier(cb_clf, X_test, y_test, 'resilience_label', 'cb')
            self.trainer.plot_feature_importance(cb_clf, X_train.columns.tolist(), 'resilience_label_clf', 'cb')
        
        # Task 2: Risk Score Regression
        if 'risk_score' in resilience_fused.columns:
            print("\n>>> TASK 2: Risk Score Regression <<<")
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
                resilience_fused, 'risk_score', 'regression'
            )
            
            # LightGBM
            lgb_reg = self.trainer.train_lightgbm_regressor(X_train, X_val, y_train, y_val, 'risk_score')
            self.trainer.evaluate_regressor(lgb_reg, X_test, y_test, 'risk_score', 'lgb')
            self.trainer.plot_feature_importance(lgb_reg, X_train.columns.tolist(), 'risk_score_reg', 'lgb')
            
            # CatBoost
            cb_reg = self.trainer.train_catboost_regressor(X_train, X_val, y_train, y_val, 'risk_score')
            self.trainer.evaluate_regressor(cb_reg, X_test, y_test, 'risk_score', 'cb')
            self.trainer.plot_feature_importance(cb_reg, X_train.columns.tolist(), 'risk_score_reg', 'cb')
        
        # Task 3: Resilience Score Regression
        if 'resilience_score' in resilience_fused.columns:
            print("\n>>> TASK 3: Resilience Score Regression <<<")
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
                resilience_fused, 'resilience_score', 'regression'
            )
            
            # LightGBM
            lgb_reg = self.trainer.train_lightgbm_regressor(X_train, X_val, y_train, y_val, 'resilience_score')
            self.trainer.evaluate_regressor(lgb_reg, X_test, y_test, 'resilience_score', 'lgb')
            self.trainer.plot_feature_importance(lgb_reg, X_train.columns.tolist(), 'resilience_score_reg', 'lgb')
            
            # CatBoost
            cb_reg = self.trainer.train_catboost_regressor(X_train, X_val, y_train, y_val, 'resilience_score')
            self.trainer.evaluate_regressor(cb_reg, X_test, y_test, 'resilience_score', 'cb')
            self.trainer.plot_feature_importance(cb_reg, X_train.columns.tolist(), 'resilience_score_reg', 'cb')
        
        # Step 8: Save Results
        print("\n[STEP 8] Saving results...")
        self.trainer.save_results_summary()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"All results saved to: {self.paths.reports}")
        print(f"All visualizations saved to: {self.paths.output}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize configuration
    paths = PathHandler(
        data="data",
        cleaned="data/cleaned",
        output="output",
        reports="reports"
    )
    
    params = HyperParameters()
    
    # Run pipeline
    pipeline = SupplyChainRiskPipeline(paths, params)
    pipeline.run_full_pipeline()
        