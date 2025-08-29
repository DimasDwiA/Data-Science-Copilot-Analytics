# Data Science Copilot untuk F&B Analytics

AI-powered assistant to analyze Food & Beverage sales data with natural language processing capabilities using the Qwen3-4B-Instruct model.

## Features

### AI-Powered Analysis
- **Automatic Insight Generation**: Automatically generate business insights from sales data
- **Natural Language Q&A**: Query data using natural Indonesian language
- **Strategic Recommendations**: Business recommendations based on data analysis
- **Interactive Chat**: Chat interface for intuitive data exploration

### Comprehensive Analytics
- **Sales Trends**: Analysis of daily, weekly, and monthly sales trends
- **Menu Performance**: Evaluation of the performance of each menu item
- **Customer Segmentation**: Analysis of behavior and value per customer segment
- **Store Comparison**: Comparison of performance between outlets
- **Correlation Analysis**: Analysis of the relationship between variables

### Advanced Visualizations
- Interactive dashboard with Plotly
- Various types of graphs: line graphs, bar graphs, pie charts, and scatter plots
- Responsive and exportable visualizations
- Real-time data updates

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **AI/LLM**: Qwen3-4B-Instruct via LiteLLM API
- **Language**: Python 3.8+

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/DimasDwiA/Data-Science-Copilot-Analytics
cd Data-Science-Copilot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
# Option 1: Create .env file
echo "QWEN_API_KEY=your_api_key_here" > .env

# Option 2: Export directly
export QWEN_API_KEY="your_api_key_here"
```

### 4. Prepare Sample Data
Ensure that the file `data/sample_sales.csv` is available, or prepare your dataset in the following format:

| date | store | menu_item | revenue | qty | customer_segment |
|------|-------|-----------|---------|-----|------------------|
| 2024-01-15 | Legit Cafe Central | Nasi Goreng Special | 85000 | 5 | Regular |
| 2024-01-15 | Legit Bistro Plaza | Beef Rendang | 120000 | 4 | Premium |

### 5. Run Application
```bash
streamlit run src/app_streamlit.py
```

The website will open in your browser at `http://localhost:8501`.

## Data Format Requirements

The CSV dataset must have the following columns:

- **date**: Transaction date (format: YYYY-MM-DD)
- **store**: Outlet/branch name
- **menu_item**: Menu/product name
- **revenue**: Revenue in Rupiah (integer)
- **qty**: Number of items sold (integer)
- **customer_segment**: Customer segment (Budget/Regular/Premium)

## Use Cases

### For Restaurant Owners
- Monitor sales in real time across all outlets
- Identify the best and worst performing menu items
- Understand customer preferences per segment
- Get actionable recommendations for growth

### For F&B Analysts
- Perform comprehensive data analysis
- Generate insights and automated reports
- Explore data with natural language queries
- Create compelling visualizations for stakeholders

### For Business Consultants
- Quick assessment of F&B business performance
- Data-driven recommendations for clients
- Comparative analysis across various metrics
- Strategic planning support

## AI Capabilities

### Insight Generation
- Automatic business insight from sales patterns
- Trend identification and explanation
- Anomaly detection and interpretation
- Performance benchmarking

### Natural Language Q&A
**Examples of questions that can be answered:**
- "Menu apa yang paling laris di weekend?"
- "Bagaimana performa outlet Plaza dibanding Central?"
- "Segment mana yang memberikan profit tertinggi?"
- "Kapan waktu penjualan paling tinggi?"

### Strategic Recommendations
- Revenue optimization strategies
- Menu pricing recommendations
- Customer retention tactics
- Operational efficiency improvements

## API Usage

### Basic LLM Client Usage
```python
from src.qwen import QwenClient

client = QwenClient(api_key="your_api_key")

# Generate insights
insight = client.generate_insight(data_summary, "general")

# Answer questions
answer = client.answer_question("What's the best selling item?", context)

# Get recommendations
recommendations = client.generate_recommendations(data_summary, business_goals)
```

### Data Processing
```python
from src.data_utils import DataProcessor

processor = DataProcessor()
df = processor.load_dataset("data/sales.csv")

# Get various analyses
trends = processor.get_sales_trends()
menu_performance = processor.analyze_menu_performance()
segments = processor.analyze_customer_segments()
```

### Visualization
```python
from src.viz_utils import DataVisualizer

viz = DataVisualizer()

# Create various plots
fig_trend = viz.plot_sales_trend(df)
fig_menu = viz.plot_menu_performance(df)
fig_segments = viz.plot_customer_segments(df)
```

## Testing

Run test using pytest:

```bash
pytest test/ -v
```

## Configuration

### Environment Variables
- `QWEN_API_KEY`: API key for the Qwen3 model (required)
- `QWEN_BASE_URL`: Base URL for the API (optional, default: provided endpoint)

### Customization
- Change the prompt in `src/prompts_llm.py` for different use cases
- Adjust the visualization style in `src/viz_utils.py`
- Extend the data processing logic in `src/data_utils.py`

## Documentation

### Key Components

1. **QwenClient**: Handles all LLM API interactions
2. **DataProcessor**: Manages data loading, cleaning, dan analysis
3. **DataVisualizer**: Creates interactive visualizations
4. **PromptTemplates**: Manages prompt engineering untuk different scenarios

### Extending the Application

To add new analysis types:

1. Add analysis logic in `DataProcessor`
2. Create corresponding visualization in `DataVisualizer`
3. Add prompt template in `PromptTemplates`
4. Update UI in `app_streamlit.py`

## Limitations

- API credit limit: $50 for assessment
- Model context limit: 8192 tokens
- Dataset size: Recommended < 10MB for optimal performance
