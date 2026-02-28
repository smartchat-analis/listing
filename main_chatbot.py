import os
import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ===============================
# LOAD API
# ===============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY tidak ditemukan di file .env")

# ===============================
# MEMORY CLASS
# ===============================
@dataclass
class ConversationMemory:
    """Menyimpan konteks percakapan"""
    conv_id: str
    history: List[Dict] = field(default_factory=list)
    
    def add_exchange(self, user_query: str, assistant_response: str):
        """Menambahkan percakapan baru ke history"""
        exchange = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(exchange)
    
    def get_recent_context(self, n_last: int = 3) -> str:
        """Mendapatkan konteks dari n percakapan terakhir"""
        recent = self.history[-n_last:] if self.history else []
        context_str = ""
        for exchange in recent:
            context_str += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        return context_str
    
    def get_last_topic(self) -> str:
        """Mendapatkan topik terakhir yang dibahas"""
        if not self.history:
            return ""
        last_exchange = self.history[-1]
        user_msg = last_exchange['user'].lower()
        
        if 'website' in user_msg or 'web' in user_msg:
            return 'website'
        elif 'seo' in user_msg:
            return 'seo'
        elif 'iklan' in user_msg or 'ads' in user_msg:
            return 'iklan'
        elif 'harga' in user_msg or 'biaya' in user_msg or 'berapa' in user_msg:
            return 'harga'
        else:
            return ''

# ===============================
# EMBEDDING CLASS
# ===============================
class EmbeddingService:
    """Mengelola embeddings menggunakan text-embedding-3-small"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embeddings_data = []
        self.similarity_threshold = 0.4   # Threshold
        
    def load_embeddings(self, file_path: str):
        """Load embeddings dari file JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.embeddings_data = json.load(f)
        
    def get_embedding(self, text: str) -> List[float]:
        """Melakukan embedding pada input user"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Menghitung cosine similarity antara input user dengan vektor bubble"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[Dict, float]], float]:
        """Mengambil top k jawaban paling mirip"""
        query_embedding = self.get_embedding(query)
        
        similarities = []
        for item in self.embeddings_data:
            similarity = self.cosine_similarity(query_embedding, item['vector'])
            similarities.append((item, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        max_similarity = similarities[0][1] if similarities else 0
        
        return similarities[:top_k], max_similarity

# ===============================
# CHATBOT CLASS
# ===============================
class ChatBot:
    """Main program ChatBot"""
    
    def __init__(self, embeddings_file: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()
        self.embedding_service.load_embeddings(embeddings_file)
        
        self.conversations: Dict[str, ConversationMemory] = {}
        
        self.system_prompt = """Anda adalah asisten layanan pelanggan PT. Asa Inovasi Software (Asain).

        IDENTITAS:
        - Nama Anda: Asisten Asain
        - Panggilan customer: "kak"
        - Gaya bicara: Ramah, profesional, jelas

        ATURAN WAJIB:
        1. Gunakan hanya informasi dari DATA REFERENSI.
        2. Jangan mengarang informasi.
        3. Jangan menambahkan nomor telepon/email jika tidak ada di data.
        4. Default: jawab singkat dan langsung ke inti.
        5. Hanya berikan penjelasan panjang jika user meminta detail, fitur, atau perbandingan.
        6. Hindari pengulangan.
        7. Jika data tidak tersedia, katakan:
        "Maaf kak, informasi tersebut belum tersedia di sistem saya."
        """
    
    def get_or_create_conversation(self, conv_id: str) -> ConversationMemory:
        """Mengambil/membuat history percakapan"""
        if conv_id not in self.conversations:
            self.conversations[conv_id] = ConversationMemory(conv_id=conv_id)
        return self.conversations[conv_id]
    
    def is_detail_question(self, query: str) -> bool:
        query_lower = query.lower()
        detail_keywords = [
            "detail", "fitur", "rincian", "lengkap",
            "jelaskan", "perbedaan", "apa saja",
            "bagaimana", "proses", "spesifikasi"
        ]
        return any(keyword in query_lower for keyword in detail_keywords)
    
    def build_prompt(self, query: str, conv_memory: ConversationMemory) -> str:
        similar_chunks, max_similarity = self.embedding_service.search_similar_chunks(query, top_k=3)
        
        if max_similarity < 0.45:
            return """DATA REFERENSI:
            (Tidak ada data relevan)

            Jawab:
            "Maaf kak, info tersebut belum ada. Mau saya bantu tanyakan dulu ke admin?"
            """
        
        reference_text = ""
        for i, (chunk, sim) in enumerate(similar_chunks, 1):
            reference_text += f"\n[DATA {i} | similarity: {sim:.2f}]\n{chunk['text']}\n"
        
        history_context = conv_memory.get_recent_context(n_last=2)
        is_detail = self.is_detail_question(query)
        
        length_rule = (
            "Jelaskan secara detail dan terstruktur (boleh bullet point)."
            if is_detail
            else "Jawab singkat, maksimal 3–4 kalimat langsung ke inti."
        )
        
        prompt = f"""
        KONTEKS PERCAKAPAN:
        {history_context if history_context else "(Tidak ada konteks sebelumnya)"}

        DATA REFERENSI:
        {reference_text}

        PERTANYAAN USER:
        {query}

        ATURAN JAWABAN:
        - Gunakan hanya DATA REFERENSI.
        - Gabungkan informasi dari beberapa DATA jika relevan.
        - Jangan menambahkan informasi baru.
        - {length_rule}
        - Hindari pengulangan kalimat.

        JAWABAN:
        """
        return prompt
    
    def generate_response(self, query: str, conv_id: str) -> str:
        conv_memory = self.get_or_create_conversation(conv_id)
        prompt = self.build_prompt(query, conv_memory)
        
        try:
            # Model LLM yang digunakan
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.25,
                max_tokens=500
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
        except Exception:
            assistant_response = "Maaf kak, sedang ada kendala teknis. Bisa dicoba lagi nanti ya."
        
        conv_memory.add_exchange(query, assistant_response)
        return assistant_response
    
    def chat(self, user_input: str, conv_id: str = None) -> str:
        if conv_id is None:
            conv_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.generate_response(user_input, conv_id)