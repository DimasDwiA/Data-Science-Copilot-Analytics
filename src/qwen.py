import os
import requests
import json
import time
from typing import Dict, List, Optional

class QwenClient:
    def __init__(self, api_key: str = None, base_url: str = "https://litellm.bangka.productionready.xyz"):
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def chat_completion(self, messages: List[Dict], **kwargs) -> str:
        payload = {
            "model": "vllm-qwen3",
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("No response from API")
                
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            raise
        except Exception as e:
            print(f"Parsing Error: {e}")
            raise
    
    def generate_insight(self, data_summary: str, analysis_type: str = "general") -> str:
        messages = [
            {
                "role": "system",
                "content": """Anda adalah ahli data science untuk bisnis F&B. 
                Berikan insight yang actionable dan specific berdasarkan data yang diberikan.
                Gunakan bahasa Indonesia yang profesional namun mudah dipahami."""
            },
            {
                "role": "user", 
                "content": f"""
                Berdasarkan data penjualan F&B berikut:
                
                {data_summary}
                
                Berikan insight mendalam tentang:
                1. Tren penjualan yang terlihat
                2. Produk/menu dengan performa terbaik dan terburuk
                3. Segmen pelanggan yang paling menguntungkan
                4. Rekomendasi strategis untuk meningkatkan revenue
                
                Format jawaban dalam poin-poin yang jelas dan actionable.
                """
            }
        ]
        
        return self.chat_completion(messages, temperature=0.3)
    
    def answer_question(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """Anda adalah assistant data science yang membantu menganalisis data penjualan F&B.
                Jawab pertanyaan user berdasarkan data yang tersedia dengan akurat dan informatif.
                Jika data tidak cukup untuk menjawab, jelaskan keterbatasannya."""
            },
            {
                "role": "user",
                "content": f"""
                Data context:
                {context}
                
                Pertanyaan user: {question}
                
                Berikan jawaban yang spesifik berdasarkan data yang tersedia.
                """
            }
        ]
        
        return self.chat_completion(messages, temperature=0.2)
    
    def generate_recommendations(self, data_summary: str, business_goals: str = "") -> str:
        messages = [
            {
                "role": "system",
                "content": """Anda adalah business consultant untuk industri F&B.
                Berikan rekomendasi yang praktis dan dapat diimplementasi berdasarkan data."""
            },
            {
                "role": "user",
                "content": f"""
                Data penjualan:
                {data_summary}
                
                Business goals: {business_goals}
                
                Berikan rekomendasi strategis untuk:
                1. Optimasi menu dan pricing
                2. Strategi marketing per segment
                3. Operasional per outlet
                4. Inventory management
                5. Customer retention
                
                Prioritaskan rekomendasi berdasarkan potential impact.
                """
            }
        ]
        
        return self.chat_completion(messages, temperature=0.4)
    
    def explain_analysis(self, analysis_result: str, chart_description: str = "") -> str:
        messages = [
            {
                "role": "system", 
                "content": "Anda adalah data interpreter yang menjelaskan hasil analisis data dengan bahasa yang mudah dipahami oleh business stakeholder."
            },
            {
                "role": "user",
                "content": f"""
                Hasil analisis:
                {analysis_result}
                
                Deskripsi chart/visualisasi:
                {chart_description}
                
                Jelaskan secara natural apa arti dari hasil analisis ini dan apa implikasinya untuk bisnis.
                """
            }
        ]
        
        return self.chat_completion(messages, temperature=0.3)