import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from io import BytesIO

# Set style
plt.style.use('default')
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def plot_sales_trend(self, df: pd.DataFrame, time_column: str = 'date') -> go.Figure:
        """Create sales trend visualization"""
        daily_sales = df.groupby(time_column).agg({
            'revenue': 'sum',
            'qty': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Revenue Trend', 'Daily Quantity Sold'),
            vertical_spacing=0.1
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(
                x=daily_sales[time_column],
                y=daily_sales['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Quantity trend
        fig.add_trace(
            go.Scatter(
                x=daily_sales[time_column],
                y=daily_sales['qty'],
                mode='lines+markers',
                name='Quantity',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Sales Trends Over Time',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Revenue (Rp)", row=1, col=1)
        fig.update_yaxes(title_text="Quantity", row=2, col=1)
        
        return fig
    
    def plot_menu_performance(self, df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        menu_performance = df.groupby('menu_item').agg({
            'revenue': 'sum',
            'qty': 'sum'
        }).reset_index()
        
        menu_performance = menu_performance.nlargest(top_n, 'revenue')
        
        fig = go.Figure(data=[
            go.Bar(
                x=menu_performance['revenue'],
                y=menu_performance['menu_item'],
                orientation='h',
                marker_color='#FF6B6B',
                text=menu_performance['revenue'],
                textposition='outside',
                texttemplate='Rp %{text:,.0f}'
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Menu Items by Revenue',
            xaxis_title='Revenue (Rp)',
            yaxis_title='Menu Items',
            height=600,
            margin=dict(l=200)
        )
        
        return fig
    
    def plot_store_comparison(self, df: pd.DataFrame) -> go.Figure:
        store_metrics = df.groupby('store').agg({
            'revenue': 'sum',
            'qty': 'sum',
            'menu_item': 'nunique'
        }).reset_index()
        
        store_metrics.columns = ['Store', 'Total Revenue', 'Total Quantity', 'Menu Variety']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Revenue by Store', 'Quantity Sold', 'Menu Variety'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Revenue
        fig.add_trace(
            go.Bar(
                x=store_metrics['Store'],
                y=store_metrics['Total Revenue'],
                name='Revenue',
                marker_color='#FF6B6B',
                text=store_metrics['Total Revenue'],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Quantity
        fig.add_trace(
            go.Bar(
                x=store_metrics['Store'],
                y=store_metrics['Total Quantity'],
                name='Quantity',
                marker_color='#4ECDC4',
                text=store_metrics['Total Quantity'],
                texttemplate='%{text}',
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Menu Variety
        fig.add_trace(
            go.Bar(
                x=store_metrics['Store'],
                y=store_metrics['Menu Variety'],
                name='Menu Items',
                marker_color='#45B7D1',
                text=store_metrics['Menu Variety'],
                texttemplate='%{text}',
                textposition='outside'
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Store Performance Comparison',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_customer_segments(self, df: pd.DataFrame) -> go.Figure:
        segment_data = df.groupby('customer_segment').agg({
            'revenue': ['sum', 'mean'],
            'qty': 'sum'
        }).round(2)
        
        segment_data.columns = ['total_revenue', 'avg_revenue', 'total_qty']
        segment_data = segment_data.reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'pie'}, {'type': 'bar'}]],
            subplot_titles=('Revenue Share by Segment', 'Average Transaction Value')
        )
        
        # Pie chart for revenue share
        fig.add_trace(
            go.Pie(
                labels=segment_data['customer_segment'],
                values=segment_data['total_revenue'],
                hole=0.4,
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ),
            row=1, col=1
        )
        
        # Bar chart for average transaction
        fig.add_trace(
            go.Bar(
                x=segment_data['customer_segment'],
                y=segment_data['avg_revenue'],
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                text=segment_data['avg_revenue'],
                texttemplate='Rp %{text:,.0f}',
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Customer Segment Analysis',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_weekly_patterns(self, df: pd.DataFrame) -> go.Figure:
        df['day_of_week'] = df['date'].dt.day_name()
        
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekly_data = df.groupby('day_of_week').agg({
            'revenue': 'mean',
            'qty': 'mean'
        }).reindex(day_order).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Daily Revenue', 'Average Daily Quantity'),
            vertical_spacing=0.1
        )
        
        # Revenue pattern
        fig.add_trace(
            go.Bar(
                x=weekly_data['day_of_week'],
                y=weekly_data['revenue'],
                name='Avg Revenue',
                marker_color='#FF6B6B',
                text=weekly_data['revenue'],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Quantity pattern
        fig.add_trace(
            go.Bar(
                x=weekly_data['day_of_week'],
                y=weekly_data['qty'],
                name='Avg Quantity',
                marker_color='#4ECDC4',
                text=weekly_data['qty'],
                texttemplate='%{text:.1f}',
                textposition='outside'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Weekly Sales Patterns',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_price_quantity_correlation(self, df: pd.DataFrame) -> go.Figure:
        df['avg_price'] = df['revenue'] / df['qty']
        
        fig = go.Figure()
        
        # Scatter plot by customer segment
        for segment in df['customer_segment'].unique():
            segment_data = df[df['customer_segment'] == segment]
            
            fig.add_trace(go.Scatter(
                x=segment_data['avg_price'],
                y=segment_data['qty'],
                mode='markers',
                name=segment,
                marker=dict(
                    size=8,
                    opacity=0.6
                )
            ))
        
        fig.update_layout(
            title='Price vs Quantity by Customer Segment',
            xaxis_title='Average Price per Item (Rp)',
            yaxis_title='Quantity Sold',
            height=500
        )
        
        return fig
    
    def create_dashboard_summary(self, df: pd.DataFrame) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 10 Menu Items', 'Revenue by Store', 
                'Customer Segments', 'Weekly Pattern'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'pie'}, {'type': 'bar'}]
            ]
        )
        
        # Top menu items
        top_menu = df.groupby('menu_item')['revenue'].sum().nlargest(10)
        fig.add_trace(
            go.Bar(x=top_menu.values, y=top_menu.index, orientation='h'),
            row=1, col=1
        )
        
        # Revenue by store
        store_revenue = df.groupby('store')['revenue'].sum()
        fig.add_trace(
            go.Pie(labels=store_revenue.index, values=store_revenue.values),
            row=1, col=2
        )
        
        # Customer segments
        segment_revenue = df.groupby('customer_segment')['revenue'].sum()
        fig.add_trace(
            go.Pie(labels=segment_revenue.index, values=segment_revenue.values),
            row=2, col=1
        )
        
        # Weekly pattern
        df['day_of_week'] = df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_revenue = df.groupby('day_of_week')['revenue'].mean().reindex(day_order)
        
        fig.add_trace(
            go.Bar(x=weekly_revenue.index, y=weekly_revenue.values),
            row=2, col=2
        )
        
        fig.update_layout(
            title='F&B Sales Dashboard Overview',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def save_plot_as_html(self, fig: go.Figure, filename: str) -> str:
        fig.write_html(filename)
        return filename
    
    def fig_to_base64(self, fig: go.Figure) -> str:
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"