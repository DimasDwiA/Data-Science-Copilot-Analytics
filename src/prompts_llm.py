# Template prompts untuk analisis dan insight generation
class PromptTemplates:
    SYSTEM_PROMPTS = {
        'data_analyst': """Anda adalah seorang senior data analyst yang spesialis di industri Food & Beverage.
        Anda memiliki expertise dalam:
        - Analisis penjualan dan tren bisnis F&B
        - Customer segmentation dan behavior analysis  
        - Menu optimization dan pricing strategy
        - Operational efficiency untuk restaurant chains
        
        Berikan insight yang:
        - Actionable dan specific
        - Berdasarkan data yang tersedia
        - Relevan untuk decision making
        - Menggunakan bahasa Indonesia yang profesional namun mudah dipahami
        """,
        
        'business_consultant': """Anda adalah business consultant berpengalaman untuk industri F&B di Indonesia.
        Anda memahami:
        - Market dynamics industri F&B Indonesia
        - Consumer behavior dan preferences
        - Operational challenges restaurant chains
        - Best practices untuk growth dan profitability
        
        Fokus pada rekomendasi yang:
        - Praktis dan dapat diimplementasikan
        - Cost-effective
        - Sesuai dengan kondisi pasar Indonesia
        - Mempertimbangkan resource constraints
        """,
        
        'qa_assistant': """Anda adalah AI assistant yang membantu user memahami data penjualan F&B.
        Anda dapat:
        - Menjawab pertanyaan spesifik tentang data
        - Menjelaskan trends dan patterns
        - Memberikan context untuk angka-angka
        - Membantu interpretasi hasil analisis
        
        Selalu:
        - Jawab berdasarkan data yang tersedia
        - Jelaskan keterbatasan jika data tidak cukup
        - Berikan contoh konkret
        - Gunakan bahasa yang mudah dipahami
        """
    }
    
    ANALYSIS_PROMPTS = {
        'sales_trend': """
        Analisis data penjualan F&B berikut dan berikan insight tentang tren penjualan:
        
        Data Summary:
        {data_summary}
        
        Fokus pada:
        1. Tren penjualan harian/mingguan/bulanan
        2. Seasonal patterns yang terlihat
        3. Growth rate dan momentum
        4. Anomali atau outliers yang menarik
        
        Berikan insight dalam format:
        - Key Findings (3-4 poin utama)
        - Business Implications
        - Recommended Actions
        """,
        
        'menu_optimization': """
        Berdasarkan data performa menu berikut:
        
        {menu_data}
        
        Analisis:
        1. Menu items dengan ROI tertinggi dan terendah
        2. Items yang underperforming vs overperforming
        3. Pricing opportunities
        4. Menu mix optimization
        
        Berikan rekomendasi:
        - Menu items yang perlu dipromosikan
        - Items yang perlu di-review pricing
        - Menu yang sebaiknya dihentikan
        - Peluang untuk menu baru berdasarkan gap analysis
        """,
        
        'customer_segmentation': """
        Analisis segmen customer berdasarkan data:
        
        {segment_data}
        
        Berikan insight tentang:
        1. Karakteristik masing-masing segment
        2. Value contribution per segment
        3. Behavior patterns yang terlihat
        4. Opportunities untuk growth per segment
        
        Rekomendasi strategi:
        - Marketing approach per segment
        - Pricing strategy
        - Menu targeting
        - Customer retention tactics
        """,
        
        'store_performance': """
        Evaluasi performa antar outlet berdasarkan:
        
        {store_data}
        
        Analisis:
        1. Store ranking berdasarkan berbagai metrics
        2. Best performing vs underperforming stores
        3. Factors yang mempengaruhi performa
        4. Consistency across locations
        
        Rekomendasi:
        - Stores yang butuh attention khusus
        - Best practices dari top performers
        - Operational improvements needed
        - Resource allocation optimization
        """
    }
    
    QA_PROMPTS = {
        'general_question': """
        User bertanya tentang data F&B:
        "{user_question}"
        
        Data context:
        {data_context}
        
        Jawab pertanyaan dengan:
        - Informasi spesifik dari data
        - Penjelasan yang mudah dipahami  
        - Contoh konkret jika relevan
        - Mention jika ada keterbatasan data
        """,
        
        'comparison_question': """
        User ingin membandingkan:
        "{user_question}"
        
        Data untuk perbandingan:
        {comparison_data}
        
        Berikan:
        - Perbandingan yang jelas dan objektif
        - Metrics yang relevan
        - Context untuk interpretasi
        - Insights dari perbedaan yang terlihat
        """,
        
        'trend_question': """
        User bertanya tentang trend:
        "{user_question}"
        
        Trend data:
        {trend_data}
        
        Jelaskan:
        - Pattern yang terlihat
        - Possible explanations
        - What it means for business
        - Future implications
        """
    }
    
    RECOMMENDATION_PROMPTS = {
        'revenue_optimization': """
        Berdasarkan analisis data penjualan:
        {analysis_summary}
        
        Berikan rekomendasi strategis untuk meningkatkan revenue:
        
        1. SHORT TERM (1-3 bulan):
           - Quick wins yang bisa diimplementasi segera
           - Low-hanging fruits
           
        2. MEDIUM TERM (3-6 bulan):
           - Strategic initiatives
           - Process improvements
           
        3. LONG TERM (6+ bulan):
           - Structural changes
           - Investment opportunities
           
        Prioritaskan berdasarkan:
        - Expected impact
        - Implementation difficulty
        - Resource requirements
        """,
        
        'operational_efficiency': """
        Data operasional menunjukkan:
        {operational_data}
        
        Rekomendasi untuk efisiensi operasional:
        
        INVENTORY MANAGEMENT:
        - Stock optimization per outlet
        - Demand forecasting improvements
        
        STAFFING & OPERATIONS:
        - Peak hours management  
        - Service efficiency
        
        COST OPTIMIZATION:
        - Food cost management
        - Operational cost reduction
        
        QUALITY CONTROL:
        - Consistency across outlets
        - Customer satisfaction metrics
        """,
        
        'marketing_strategy': """
        Customer dan sales data:
        {customer_sales_data}
        
        Strategi marketing yang disarankan:
        
        CUSTOMER ACQUISITION:
        - Target customer profiles
        - Channel optimization
        
        CUSTOMER RETENTION:
        - Loyalty program opportunities
        - Personalization strategies
        
        PROMOTION STRATEGY:
        - Menu items untuk promosi
        - Timing dan targeting
        
        BRAND POSITIONING:
        - Value proposition per segment
        - Competitive differentiation
        """
    }
    
    @staticmethod
    def get_system_prompt(role: str) -> str:
        return PromptTemplates.SYSTEM_PROMPTS.get(role, PromptTemplates.SYSTEM_PROMPTS['data_analyst'])
    
    @staticmethod
    def format_analysis_prompt(analysis_type: str, **kwargs) -> str:
        template = PromptTemplates.ANALYSIS_PROMPTS.get(analysis_type, "")
        return template.format(**kwargs)
    
    @staticmethod
    def format_qa_prompt(question_type: str, **kwargs) -> str:
        template = PromptTemplates.QA_PROMPTS.get(question_type, PromptTemplates.QA_PROMPTS['general_question'])
        return template.format(**kwargs)
    
    @staticmethod
    def format_recommendation_prompt(rec_type: str, **kwargs) -> str:
        template = PromptTemplates.RECOMMENDATION_PROMPTS.get(rec_type, "")
        return template.format(**kwargs)