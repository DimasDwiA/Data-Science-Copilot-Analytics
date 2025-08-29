import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from dotenv import load_dotenv

from qwen import QwenClient
from data_utils import DataProcessor
from viz_utils import DataVisualizer
from prompts_llm import PromptTemplates

load_dotenv()

# Page config
st.set_page_config(
    page_title="Data Science Copilot - F&B Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        min-height: 120px;   
        display: flex;       
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }
    .insight-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .recommendation-box {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DataVisualizer()
if 'llm_client' not in st.session_state:
    # Get API key from environment or user input
    api_key = os.getenv('QWEN_API_KEY', 'sk-2pMILc-7********Q')
    st.session_state.llm_client = QwenClient(api_key=api_key)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Data Science Copilot untuk F&B Analytics</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Analytics untuk Bisnis Food & Beverage**")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload dataset CSV",
            type=['csv'],
            help="Upload file CSV dengan kolom: date, store, menu_item, revenue, qty, customer_segment"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.session_state.data_processor.df = df
                st.session_state.data_processor._clean_data()
                st.session_state.data_processor._generate_summary()
                st.session_state.data_loaded = True
                st.success("✅ Data berhasil dimuat!")
                
                # Show basic info
                st.write(f"**Records:** {len(df)}")
                st.write(f"**Periode:** {df['date'].min()} - {df['date'].max()}")
                st.write(f"**Stores:** {df['store'].nunique()}")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        
        # Sample data
        st.markdown("---")
        if st.button("🔄 Load Sample Data"):
            try:
                sample_path = os.path.join("data", "sample_sales.csv")
                if os.path.exists(sample_path):
                    st.session_state.data_processor.load_dataset(sample_path)
                    st.session_state.data_loaded = True
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                else:
                    st.warning("Sample data tidak ditemukan. Silakan upload file CSV.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("Silakan upload dataset CSV atau gunakan sample data dari sidebar.")
        
        # Show expected format
        st.subheader("📋 Format Data yang Diharapkan")
        sample_df = pd.DataFrame({
            'date': ['2024-01-15', '2024-01-15', '2024-01-16'],
            'store': ['Store A', 'Store B', 'Store A'],
            'menu_item': ['Nasi Goreng', 'Ayam Bakar', 'Es Teh'],
            'revenue': [85000, 78000, 15000],
            'qty': [5, 4, 10],
            'customer_segment': ['Regular', 'Premium', 'Budget']
        })
        st.dataframe(sample_df)
        return
    
    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Overview", 
        "🔍 Data Analysis", 
        "💡 AI Insights", 
        "❓ Ask Questions", 
        "📈 Recommendations"
    ])
    
    dp = st.session_state.data_processor
    viz = st.session_state.visualizer
    llm = st.session_state.llm_client
    
    with tab1:
        show_dashboard_overview(dp, viz)
    
    with tab2:
        show_data_analysis(dp, viz)
    
    with tab3:
        show_ai_insights(dp, llm)
    
    with tab4:
        show_qa_interface(dp, llm)
    
    with tab5:
        show_recommendations(dp, llm)

def show_dashboard_overview(dp: DataProcessor, viz: DataVisualizer):
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>💰 Total Revenue</h3>
            <h2>Rp {dp.data_summary['total_revenue']:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📦 Items Sold</h3>
            <h2>{dp.data_summary['total_qty_sold']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🏪 Active Stores</h3>
            <h2>{len(dp.data_summary['stores'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🍽️ Menu Items</h3>
            <h2>{dp.data_summary['unique_menu_items']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales trend
        fig_trend = viz.plot_sales_trend(dp.df)
        st.plotly_chart(fig_trend, use_container_width=True, key="sales_chart")
    
    with col2:
        # Store comparison
        fig_store = viz.plot_store_comparison(dp.df)
        st.plotly_chart(fig_store, use_container_width=True, key="store_comparison")
    
    # Customer segments and menu performance
    col1, col2 = st.columns(2)
    
    with col1:
        fig_segments = viz.plot_customer_segments(dp.df)
        st.plotly_chart(fig_segments, use_container_width=True, key="customer_segment_chart")
    
    with col2:
        fig_menu = viz.plot_menu_performance(dp.df)
        st.plotly_chart(fig_menu, use_container_width=True, key="menu_chart")

def show_data_analysis(dp: DataProcessor, viz: DataVisualizer):
    st.header("🔍 Data Analysis")
    
    analysis_type = st.selectbox(
        "Pilih jenis analisis:",
        ["Sales Trends", "Menu Performance", "Customer Segments", "Store Performance", "Weekly Patterns", "Price-Quantity Correlation"]
    )
    
    if analysis_type == "Sales Trends":
        trends = dp.get_sales_trends()
        st.subheader("📈 Sales Trends Analysis")
        
        fig = viz.plot_sales_trend(dp.df)
        st.plotly_chart(fig, use_container_width=True, key="sales_trend")
        
        # Show trend data
        if trends['daily_trends']:
            st.subheader("Daily Trends Data")
            daily_df = pd.DataFrame(trends['daily_trends'])
            st.dataframe(daily_df.tail(10))
    
    elif analysis_type == "Menu Performance":
        menu_analysis = dp.analyze_menu_performance()
        st.subheader("🍽️ Menu Performance Analysis")
        
        fig = viz.plot_menu_performance(dp.df)
        st.plotly_chart(fig, use_container_width=True, key="menu_performance")
        
        # Top and low performers
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Top Performers")
            top_df = pd.DataFrame(menu_analysis['top_performers'])
            st.dataframe(top_df)
        
        with col2:
            st.subheader("⚠️ Low Performers")
            low_df = pd.DataFrame(menu_analysis['low_performers'])
            st.dataframe(low_df)
    
    elif analysis_type == "Customer Segments":
        segment_analysis = dp.analyze_customer_segments()
        st.subheader("👥 Customer Segment Analysis")
        
        fig = viz.plot_customer_segments(dp.df)
        st.plotly_chart(fig, use_container_width=True, key="customer_segment")
        
        # Segment performance table
        segment_df = pd.DataFrame(segment_analysis['segment_performance'])
        st.dataframe(segment_df)
    
    elif analysis_type == "Store Performance":
        store_analysis = dp.analyze_store_performance()
        st.subheader("🏪 Store Performance Analysis")
        
        fig = viz.plot_store_comparison(dp.df)
        st.plotly_chart(fig, use_container_width=True, key="store_performance")
        
        # Store performance table
        store_df = pd.DataFrame(store_analysis['store_performance'])
        st.dataframe(store_df)
    
    elif analysis_type == "Weekly Patterns":
        st.subheader("📅 Weekly Sales Patterns")
        
        fig = viz.plot_weekly_patterns(dp.df)
        st.plotly_chart(fig, use_container_width=True, key="weekly_patterns")
    
    elif analysis_type == "Price-Quantity Correlation":
        st.subheader("💰 Price vs Quantity Analysis")
        
        fig = viz.plot_price_quantity_correlation(dp.df)
        st.plotly_chart(fig, use_container_width=True, key="price_qty_corr")
        
        # Correlation insights
        correlation_analysis = dp.get_correlation_analysis()
        st.write("**Correlation Insights:**")
        for key, value in correlation_analysis['insights'].items():
            st.write(f"- {key.replace('_', ' ').title()}: {value:.3f}")

def show_ai_insights(dp: DataProcessor, llm: QwenClient):
    st.header("💡 AI-Generated Insights")
    
    insight_type = st.selectbox(
        "Pilih jenis insight:",
        ["General Business Insights", "Sales Trend Analysis", "Menu Optimization", "Customer Behavior", "Store Performance"]
    )
    
    if st.button("🤖 Generate Insights", type="primary"):
        with st.spinner("Generating AI insights..."):
            try:
                data_context = dp.get_data_context_for_llm()
                
                if insight_type == "General Business Insights":
                    insight = llm.generate_insight(data_context, "general")
                elif insight_type == "Sales Trend Analysis":
                    trends = dp.get_sales_trends()
                    prompt = PromptTemplates.format_analysis_prompt("sales_trend", data_summary=data_context)
                    insight = llm.chat_completion([
                        {"role": "system", "content": PromptTemplates.get_system_prompt("data_analyst")},
                        {"role": "user", "content": prompt}
                    ])
                elif insight_type == "Menu Optimization":
                    menu_data = dp.analyze_menu_performance()
                    prompt = PromptTemplates.format_analysis_prompt("menu_optimization", menu_data=str(menu_data))
                    insight = llm.chat_completion([
                        {"role": "system", "content": PromptTemplates.get_system_prompt("business_consultant")},
                        {"role": "user", "content": prompt}
                    ])
                elif insight_type == "Customer Behavior":
                    segment_data = dp.analyze_customer_segments()
                    prompt = PromptTemplates.format_analysis_prompt("customer_segmentation", segment_data=str(segment_data))
                    insight = llm.chat_completion([
                        {"role": "system", "content": PromptTemplates.get_system_prompt("data_analyst")},
                        {"role": "user", "content": prompt}
                    ])
                elif insight_type == "Store Performance":
                    store_data = dp.analyze_store_performance()
                    prompt = PromptTemplates.format_analysis_prompt("store_performance", store_data=str(store_data))
                    insight = llm.chat_completion([
                        {"role": "system", "content": PromptTemplates.get_system_prompt("business_consultant")},
                        {"role": "user", "content": prompt}
                    ])
                
                st.markdown(f"""
                <div class="insight-box">
                    <h3>🔍 AI Insights: {insight_type}</h3>
                    {insight}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating insights: {str(e)}")

def show_qa_interface(dp: DataProcessor, llm: QwenClient):
    st.header("❓ Ask Questions About Your Data")
    st.write("Tanyakan apa saja tentang data penjualan F&B Anda dalam bahasa Indonesia!")
    
    # Example questions
    with st.expander("💡 Contoh Pertanyaan"):
        st.write("""
        - "Menu apa yang paling laris di weekend?"
        - "Bagaimana performa Legit Bistro Plaza dibanding outlet lain?"
        - "Segment pelanggan mana yang paling menguntungkan?"
        - "Produk apa yang harganya paling mahal tapi jarang dibeli?"
        - "Kapan waktu penjualan paling tinggi dalam seminggu?"
        - "Berapa rata-rata pembelian customer premium?"
        """)
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {question}")
        st.markdown(f"**A{i+1}:** {answer}")
        st.markdown("---")
    
    # Input area
    question = st.text_input("Tanyakan sesuatu tentang data Anda:", placeholder="Contoh: Menu apa yang paling laris bulan ini?")
    
    if st.button("🔍 Tanya AI") and question:
        with st.spinner("Mencari jawaban..."):
            try:
                data_context = dp.get_data_context_for_llm()
                
                # Try to provide more specific context based on question
                if any(word in question.lower() for word in ['menu', 'makanan', 'minuman']):
                    menu_data = dp.analyze_menu_performance()
                    data_context += f"\n\nMENU PERFORMANCE DATA:\n{str(menu_data)}"
                
                if any(word in question.lower() for word in ['store', 'outlet', 'cabang']):
                    store_data = dp.analyze_store_performance()
                    data_context += f"\n\nSTORE PERFORMANCE DATA:\n{str(store_data)}"
                
                if any(word in question.lower() for word in ['segment', 'customer', 'pelanggan']):
                    segment_data = dp.analyze_customer_segments()
                    data_context += f"\n\nCUSTOMER SEGMENT DATA:\n{str(segment_data)}"
                
                prompt = PromptTemplates.format_qa_prompt(
                    "general_question", 
                    user_question=question,
                    data_context=data_context
                )
                
                answer = llm.chat_completion([
                    {"role": "system", "content": PromptTemplates.get_system_prompt("qa_assistant")},
                    {"role": "user", "content": prompt}
                ])
                
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                
                # Display the new answer
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {answer}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_recommendations(dp: DataProcessor, llm: QwenClient):
    st.header("📈 Strategic Recommendations")
    
    rec_type = st.selectbox(
        "Pilih jenis rekomendasi:",
        ["Revenue Optimization", "Operational Efficiency", "Marketing Strategy", "Menu Strategy"]
    )
    
    business_goal = st.text_area(
        "Business goals atau fokus khusus (optional):",
        placeholder="Contoh: Ingin meningkatkan revenue 20% dalam 6 bulan, fokus pada customer retention, dll."
    )
    
    if st.button("🎯 Generate Recommendations", type="primary"):
        with st.spinner("Generating strategic recommendations..."):
            try:
                data_context = dp.get_data_context_for_llm()
                
                if rec_type == "Revenue Optimization":
                    analysis_summary = f"{data_context}\n\nMENU ANALYSIS:\n{dp.analyze_menu_performance()}\nSTORE ANALYSIS:\n{dp.analyze_store_performance()}"
                    prompt = PromptTemplates.format_recommendation_prompt(
                        "revenue_optimization",
                        analysis_summary=analysis_summary
                    )
                elif rec_type == "Operational Efficiency":
                    operational_data = f"{data_context}\nSTORE PERFORMANCE:\n{dp.analyze_store_performance()}\nTRENDS:\n{dp.get_sales_trends()}"
                    prompt = PromptTemplates.format_recommendation_prompt(
                        "operational_efficiency",
                        operational_data=operational_data
                    )
                elif rec_type == "Marketing Strategy":
                    customer_sales_data = f"{data_context}\nCUSTOMER SEGMENTS:\n{dp.analyze_customer_segments()}\nMENU PERFORMANCE:\n{dp.analyze_menu_performance()}"
                    prompt = PromptTemplates.format_recommendation_prompt(
                        "marketing_strategy",
                        customer_sales_data=customer_sales_data
                    )
                else:  # Menu Strategy
                    prompt = f"""
                    Berdasarkan data menu performance:
                    {dp.analyze_menu_performance()}
                    
                    Dan customer behavior:
                    {dp.analyze_customer_segments()}
                    
                    Business goals: {business_goal}
                    
                    Berikan rekomendasi strategis untuk optimasi menu, termasuk:
                    1. Menu items yang perlu dipromosikan
                    2. Pricing strategy
                    3. Menu innovation opportunities
                    4. Seasonal menu recommendations
                    """
                
                recommendations = llm.chat_completion([
                    {"role": "system", "content": PromptTemplates.get_system_prompt("business_consultant")},
                    {"role": "user", "content": prompt}
                ])
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h3>🎯 Strategic Recommendations: {rec_type}</h3>
                    {recommendations}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
    
    # Export recommendations
    st.markdown("---")
    st.subheader("📄 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Data Summary"):
            summary = dp.data_summary
            st.download_button(
                label="Download JSON",
                data=json.dumps(summary, indent=2, default=str),
                file_name=f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📈 Export Analysis Report"):
            # Create comprehensive report
            report = {
                "generated_at": datetime.now().isoformat(),
                "data_summary": dp.data_summary,
                "sales_trends": dp.get_sales_trends(),
                "menu_performance": dp.analyze_menu_performance(),
                "customer_segments": dp.analyze_customer_segments(),
                "store_performance": dp.analyze_store_performance(),
                "correlation_analysis": dp.get_correlation_analysis()
            }
            
            st.download_button(
                label="Download Report",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"fnb_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()