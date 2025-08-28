import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.df = None
        self.data_summary = {}
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(file_path)
            self._clean_data()
            self._generate_summary()
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def _clean_data(self):
        if self.df is None:
            return
            
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Remove outliers (optional)
        Q1 = self.df['revenue'].quantile(0.25)
        Q3 = self.df['revenue'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Flag outliers but don't remove (for F&B business, high revenue might be valid)
        self.df['is_outlier'] = (self.df['revenue'] < lower_bound) | (self.df['revenue'] > upper_bound)
        
        # Add derived columns
        self.df['avg_price'] = self.df['revenue'] / self.df['qty']
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['week'] = self.df['date'].dt.isocalendar().week
    
    def _generate_summary(self):
        if self.df is None:
            return
            
        self.data_summary = {
            'total_records': len(self.df),
            'date_range': {
                'start': self.df['date'].min().strftime('%Y-%m-%d'),
                'end': self.df['date'].max().strftime('%Y-%m-%d')
            },
            'total_revenue': self.df['revenue'].sum(),
            'total_qty_sold': self.df['qty'].sum(),
            'avg_transaction_value': self.df['revenue'].mean(),
            'stores': self.df['store'].unique().tolist(),
            'customer_segments': self.df['customer_segment'].unique().tolist(),
            'unique_menu_items': self.df['menu_item'].nunique(),
            'top_selling_items': self.df.groupby('menu_item')['qty'].sum().nlargest(5).to_dict(),
            'revenue_by_store': self.df.groupby('store')['revenue'].sum().to_dict(),
            'revenue_by_segment': self.df.groupby('customer_segment')['revenue'].sum().to_dict()
        }
    
    def get_sales_trends(self) -> Dict:
        if self.df is None:
            return {}
            
        # Daily trends
        daily_sales = self.df.groupby('date').agg({
            'revenue': 'sum',
            'qty': 'sum'
        }).reset_index()
        
        # Monthly trends
        monthly_sales = self.df.groupby('month').agg({
            'revenue': 'sum',
            'qty': 'sum'
        }).reset_index()
        
        # Weekly patterns
        weekly_pattern = self.df.groupby('day_of_week').agg({
            'revenue': 'mean',
            'qty': 'mean'
        }).reset_index()
        
        return {
            'daily_trends': daily_sales.to_dict('records'),
            'monthly_trends': monthly_sales.to_dict('records'),
            'weekly_patterns': weekly_pattern.to_dict('records')
        }
    
    def analyze_menu_performance(self) -> Dict:
        if self.df is None:
            return {}
            
        menu_analysis = self.df.groupby('menu_item').agg({
            'revenue': ['sum', 'mean', 'count'],
            'qty': 'sum',
            'avg_price': 'mean'
        }).round(2)
        
        menu_analysis.columns = ['total_revenue', 'avg_revenue', 'frequency', 'total_qty', 'avg_price']
        menu_analysis = menu_analysis.reset_index()
        
        # Add performance metrics
        menu_analysis['revenue_per_order'] = menu_analysis['total_revenue'] / menu_analysis['frequency']
        menu_analysis['popularity_score'] = (menu_analysis['frequency'] / menu_analysis['frequency'].max()) * 100
        
        # Categorize performance
        def categorize_performance(row):
            if row['total_revenue'] > menu_analysis['total_revenue'].quantile(0.8):
                return 'High Performer'
            elif row['total_revenue'] < menu_analysis['total_revenue'].quantile(0.2):
                return 'Low Performer'
            else:
                return 'Average'
        
        menu_analysis['performance_category'] = menu_analysis.apply(categorize_performance, axis=1)
        
        return {
            'menu_performance': menu_analysis.to_dict('records'),
            'top_performers': menu_analysis.nlargest(10, 'total_revenue')[['menu_item', 'total_revenue', 'total_qty']].to_dict('records'),
            'low_performers': menu_analysis.nsmallest(5, 'total_revenue')[['menu_item', 'total_revenue', 'total_qty']].to_dict('records')
        }
    
    def analyze_customer_segments(self) -> Dict:
        if self.df is None:
            return {}
            
        segment_analysis = self.df.groupby('customer_segment').agg({
            'revenue': ['sum', 'mean', 'count'],
            'qty': ['sum', 'mean'],
            'avg_price': 'mean'
        }).round(2)
        
        segment_analysis.columns = ['total_revenue', 'avg_revenue_per_transaction', 'transaction_count', 
                                   'total_qty', 'avg_qty_per_transaction', 'avg_price_per_item']
        segment_analysis = segment_analysis.reset_index()
        
        # Calculate segment metrics
        total_revenue = self.df['revenue'].sum()
        segment_analysis['revenue_share'] = (segment_analysis['total_revenue'] / total_revenue * 100).round(2)
        segment_analysis['customer_value'] = segment_analysis['total_revenue'] / segment_analysis['transaction_count']
        
        return {
            'segment_performance': segment_analysis.to_dict('records'),
            'segment_comparison': self.df.groupby('customer_segment')['avg_price'].describe().to_dict()
        }
    
    def analyze_store_performance(self) -> Dict:
        if self.df is None:
            return {}
            
        store_analysis = self.df.groupby('store').agg({
            'revenue': ['sum', 'mean', 'count'],
            'qty': ['sum', 'mean'],
            'menu_item': 'nunique'
        }).round(2)
        
        store_analysis.columns = ['total_revenue', 'avg_revenue_per_transaction', 'transaction_count',
                                 'total_qty_sold', 'avg_qty_per_transaction', 'menu_variety']
        store_analysis = store_analysis.reset_index()
        
        # Performance metrics
        store_analysis['revenue_per_day'] = store_analysis['total_revenue'] / self.df['date'].nunique()
        store_analysis['efficiency_score'] = (store_analysis['total_revenue'] / store_analysis['transaction_count']).round(2)
        
        # Popular items per store
        store_popular_items = {}
        for store in self.df['store'].unique():
            store_data = self.df[self.df['store'] == store]
            popular = store_data.groupby('menu_item')['qty'].sum().nlargest(5).to_dict()
            store_popular_items[store] = popular
        
        return {
            'store_performance': store_analysis.to_dict('records'),
            'store_popular_items': store_popular_items
        }
    
    def get_correlation_analysis(self) -> Dict:
        if self.df is None:
            return {}
            
        # Numerical columns for correlation
        numerical_cols = ['revenue', 'qty', 'avg_price']
        correlation_matrix = self.df[numerical_cols].corr().round(3)
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'insights': {
                'price_qty_correlation': correlation_matrix.loc['avg_price', 'qty'],
                'price_revenue_correlation': correlation_matrix.loc['avg_price', 'revenue'],
                'qty_revenue_correlation': correlation_matrix.loc['qty', 'revenue']
            }
        }
    
    def get_data_context_for_llm(self) -> str:
        if self.df is None:
            return "No data loaded"
            
        context = f"""
        DATASET SUMMARY:
        - Total Records: {self.data_summary['total_records']}
        - Date Range: {self.data_summary['date_range']['start']} to {self.data_summary['date_range']['end']}
        - Total Revenue: Rp {self.data_summary['total_revenue']:,}
        - Total Items Sold: {self.data_summary['total_qty_sold']}
        - Average Transaction: Rp {self.data_summary['avg_transaction_value']:,.0f}
        
        STORES: {', '.join(self.data_summary['stores'])}
        CUSTOMER SEGMENTS: {', '.join(self.data_summary['customer_segments'])}
        UNIQUE MENU ITEMS: {self.data_summary['unique_menu_items']}
        
        TOP SELLING ITEMS:
        {json.dumps(self.data_summary['top_selling_items'], indent=2)}
        
        REVENUE BY STORE:
        {json.dumps(self.data_summary['revenue_by_store'], indent=2)}
        
        REVENUE BY SEGMENT:
        {json.dumps(self.data_summary['revenue_by_segment'], indent=2)}
        """
        
        return context
    
    def search_data(self, query: str) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
            
        query = query.lower()
        
        # Search in menu items, stores, or customer segments
        mask = (
            self.df['menu_item'].str.lower().str.contains(query, na=False) |
            self.df['store'].str.lower().str.contains(query, na=False) |
            self.df['customer_segment'].str.lower().str.contains(query, na=False)
        )
        
        return self.df[mask]