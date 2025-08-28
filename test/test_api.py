import pytest
import pandas as pd
import os
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen import QwenClient
from data_utils import DataProcessor
from viz_utils import DataVisualizer

class TestQwenClient:
    def setup_method(self):
        self.client = QwenClient(api_key="test_key")
    
    def test_init(self):
        """Test client initialization"""
        assert self.client.api_key == "test_key"
        assert "litellm.bangka.productionready.xyz" in self.client.base_url
    
    @patch('requests.post')
    def test_chat_completion_success(self, mock_post):
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        result = self.client.chat_completion(messages)
        
        assert result == "Test response"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_chat_completion_failure(self, mock_post):
        mock_post.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "Test message"}]
        
        with pytest.raises(Exception):
            self.client.chat_completion(messages)
    
    def test_generate_insight(self):
        assert hasattr(self.client, 'generate_insight')
        assert callable(self.client.generate_insight)
    
    def test_answer_question(self):
        assert hasattr(self.client, 'answer_question')
        assert callable(self.client.answer_question)

class TestDataProcessor:
    def setup_method(self):
        self.processor = DataProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'store': ['Store A', 'Store B', 'Store A'],
            'menu_item': ['Item 1', 'Item 2', 'Item 1'],
            'revenue': [100000, 150000, 120000],
            'qty': [5, 3, 6],
            'customer_segment': ['Regular', 'Premium', 'Budget']
        })
    
    def test_init(self):
        assert self.processor.df is None
        assert isinstance(self.processor.data_summary, dict)
    
    def test_load_data(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        self.processor._generate_summary()
        
        assert len(self.processor.df) == 3
        assert 'avg_price' in self.processor.df.columns
        assert self.processor.data_summary['total_records'] == 3
    
    def test_clean_data(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        
        # Check if derived columns are created
        assert 'avg_price' in self.processor.df.columns
        assert 'month' in self.processor.df.columns
        assert 'day_of_week' in self.processor.df.columns
    
    def test_generate_summary(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        self.processor._generate_summary()
        
        summary = self.processor.data_summary
        assert summary['total_records'] == 3
        assert summary['total_revenue'] == 370000
        assert summary['total_qty_sold'] == 14
        assert len(summary['stores']) == 2
    
    def test_get_sales_trends(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        
        trends = self.processor.get_sales_trends()
        assert 'daily_trends' in trends
        assert 'monthly_trends' in trends
        assert 'weekly_patterns' in trends
    
    def test_analyze_menu_performance(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        
        menu_analysis = self.processor.analyze_menu_performance()
        assert 'menu_performance' in menu_analysis
        assert 'top_performers' in menu_analysis
    
    def test_analyze_customer_segments(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        
        segment_analysis = self.processor.analyze_customer_segments()
        assert 'segment_performance' in segment_analysis
    
    def test_get_data_context(self):
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        self.processor._generate_summary()
        
        context = self.processor.get_data_context_for_llm()
        assert isinstance(context, str)
        assert "DATASET SUMMARY" in context
        assert "Total Records: 3" in context

class TestDataVisualizer:
    def setup_method(self):
        self.viz = DataVisualizer()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'store': ['Store A', 'Store B', 'Store A'],
            'menu_item': ['Item 1', 'Item 2', 'Item 1'],
            'revenue': [100000, 150000, 120000],
            'qty': [5, 3, 6],
            'customer_segment': ['Regular', 'Premium', 'Budget']
        })
    
    def test_init(self):
        assert self.viz.figsize == (12, 8)
        assert len(self.viz.color_palette) > 0
    
    def test_plot_methods_exist(self):
        methods = [
            'plot_sales_trend',
            'plot_menu_performance', 
            'plot_store_comparison',
            'plot_customer_segments',
            'plot_weekly_patterns',
            'plot_price_quantity_correlation'
        ]
        
        for method in methods:
            assert hasattr(self.viz, method)
            assert callable(getattr(self.viz, method))
    
    def test_plot_sales_trend(self):
        fig = self.viz.plot_sales_trend(self.sample_data)
        assert fig is not None

        # Basic check that it's a plotly figure
        assert hasattr(fig, 'data')
    
    def test_plot_menu_performance(self):
        fig = self.viz.plot_menu_performance(self.sample_data)
        assert fig is not None
        assert hasattr(fig, 'data')

class TestIntegration:
    def setup_method(self):
        self.processor = DataProcessor()
        self.viz = DataVisualizer()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'] * 10),
            'store': ['Store A', 'Store B', 'Store C'] * 10,
            'menu_item': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'] * 6,
            'revenue': [100000, 150000, 120000] * 10,
            'qty': [5, 3, 6] * 10,
            'customer_segment': ['Regular', 'Premium', 'Budget'] * 10
        })
    
    def test_full_pipeline(self):
        # Load and process data
        self.processor.df = self.sample_data.copy()
        self.processor._clean_data()
        self.processor._generate_summary()
        
        # Generate analysis
        trends = self.processor.get_sales_trends()
        menu_analysis = self.processor.analyze_menu_performance()
        segment_analysis = self.processor.analyze_customer_segments()
        
        # Create visualizations
        fig_trend = self.viz.plot_sales_trend(self.processor.df)
        fig_menu = self.viz.plot_menu_performance(self.processor.df)
        
        # Assertions
        assert len(self.processor.df) == 30
        assert trends is not None
        assert menu_analysis is not None
        assert segment_analysis is not None
        assert fig_trend is not None
        assert fig_menu is not None
        
        # Check data context generation
        context = self.processor.get_data_context_for_llm()
        assert isinstance(context, str)
        assert len(context) > 100  

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])