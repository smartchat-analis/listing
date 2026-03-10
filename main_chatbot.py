import os
import numpy as np
import json
import sqlite3
from dataclasses import dataclass, field
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
# CONFIG
# ===============================
DB_NAME = "knowledge_base.db"
TABLE_NAME = "conversation_embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.5
TOP_K = 3

# ===============================
# MEMORY CLASS
# ===============================
@dataclass
class ConversationMemory:
    """Menyimpan konteks percakapan"""

    conv_id: str
    history: list[dict] = field(default_factory=list)

    # Produk, package, dan intent
    current_product: str = ""
    current_package: str = ""
    last_intent: str = ""

    # Nama orang, perusahaan, dan usaha jika ada
    user_name: str = ""
    company_name: str = ""
    business_type: str = ""
    
    def add_exchange(self, user_query: str, assistant_response: str):
        """Menambahkan percakapan baru ke history"""
        self.history.append({
            "user": user_query,
            "assistant": assistant_response
        })

# ===============================
# EMBEDDING CLASS
# ===============================
class EmbeddingService:
    """Mengelola embeddings menggunakan text-embedding-3-small"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embeddings_data = []
        self.similarity_threshold = SIMILARITY_THRESHOLD
        
    def load_embeddings(self, db_path: str, table_name: str = "conversation_embeddings"):
        """Load embeddings dari SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT text, vector FROM {table_name}")
        rows = cursor.fetchall()

        self.embeddings_data = []

        for text, vector in rows:
            self.embeddings_data.append({
                "text": text,
                "vector": json.loads(vector)
            })

        conn.close()

    def get_embedding(self, text: str) -> list[float]:
        """Melakukan embedding pada input user"""
        response = self.client.embeddings.create(
            model = EMBEDDING_MODEL,
            input = text
        )
        return response.data[0].embedding
    
    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Menghitung cosine similarity antara input user dengan vektor bubble"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar_chunks(self, query: str, top_k: int = TOP_K) -> tuple[list[tuple[dict, float]], float]:
        """Mengambil top k jawaban paling mirip"""
        query_embedding = self.get_embedding(query)
        
        similarities = []
        for item in self.embeddings_data:
            similarity = self.cosine_similarity(query_embedding, item['vector'])
            similarities.append((item, similarity))

        # Hanya ambil chunk dengan similarity > threshold
        similarities = [
            (item, sim) for item, sim in similarities
            if sim >= self.similarity_threshold
        ]

        if not similarities:
            return [], 0
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        max_similarity = similarities[0][1] if similarities else 0
        
        return similarities[:top_k], max_similarity

# ===============================
# CHATBOT CLASS
# ===============================
class ChatBot:
    """Main program ChatBot"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()
        self.embedding_service.load_embeddings(DB_NAME, TABLE_NAME)
        
        self.conversations: dict[str, ConversationMemory] = {}
        
        self.system_prompt = """Anda adalah admin di PT. Asa Inovasi Software (Asain) yang tugasnya menjawab pertanyaan pelanggan.

        IDENTITAS:
        - Nama Anda: Admin Asain
        - Panggilan customer: "kak"
        - Gaya bicara: Ramah, profesional, jelas

        ATURAN WAJIB:
        - Gunakan hanya informasi dari DATA REFERENSI.
        - Jangan mengarang informasi.
        - Jangan menambahkan nomor telepon/email jika tidak ada di data.
        - Jangan menambahkan informasi yang tidak ada di pertanyaan
        - Hanya berikan penjelasan panjang jika user meminta detail, fitur, atau perbandingan.
        - Jangan selalu menutup jawaban dengan kalimat penutup yang sama dan jangan terlalu kaku.
        - Gunakan variasi natural sesuai konteks.
        - Jika jawaban sudah cukup jelas, boleh diakhiri tanpa kalimat tambahan.
        - Ikuti gaya bahasa yang ada di DATA REFERENSI.
        - Jika data tidak tersedia, katakan:
        "Maaf kak, informasi tersebut belum tersedia di database saya."
        """

    def get_or_create_conversation(self, conv_id: str) -> ConversationMemory:
        """Mengambil/membuat history percakapan"""            
        if conv_id not in self.conversations:
            self.conversations[conv_id] = ConversationMemory(conv_id=conv_id)
        return self.conversations[conv_id]
    
    def analyze_query(self, query: str, conv_memory: ConversationMemory):
        """Menentukan product dan package dari user query"""

        analysis_prompt = f"""
        Anda bertugas menganalisis pertanyaan user dan mengembalikan JSON dengan format:

        {{
            "product": string atau null,
            "package": string atau null,
            "intent": string atau null,
            "user_name": string atau null,
            "company_name": string atau null,
            "business_type": string atau null
        }}

        ATURAN PRODUCT:
        - Product adalah jenis layanan seperti website, seo, iklan, dan company profile.
        - Jika user menyebutkan variasi kata seperti:
            - web, website → website
            - seo → seo
            - ads, iklan → iklan
            - compro → company profile
        - Jika tidak disebut tapi sudah ada sebelumnya:
        Product saat ini: {conv_memory.current_product}
        maka gunakan product tersebut.

        ATURAN PACKAGE:
        - Package adalah nama paket spesifik yang disebut di DATA REFERENSI.
        - Jika user memilih atau menyebut salah satu paket di referensi, isi field package.
        - Jika user tidak menyebutkan paket tertentu, isi null.
        - Jika sebelumnya sudah ada package aktif dan masih relevan, gunakan package tersebut.

        Contoh intent:
        - tanya_harga
        - tanya_fitur
        - tanya_proses
        - perbandingan
        - lainnya

        ATURAN BUSINESS TYPE:
        - Business type adalah jenis usaha user jika disebutkan.
        - Contoh: restoran, toko online, bimbel, klinik, jasa travel, otomotif, akrilik, dll.
        - Jika user menyebut usaha seperti:
        "saya jual rendang kemasan"
        "usaha laundry"
        "bisnis travel"
        maka isi field business_type.
        - Jika tidak disebutkan, isi null.

        Jika tidak disebutkan, isi null.
        Gunakan konteks sebelumnya:
        Product saat ini: {conv_memory.current_product}
        Package saat ini: {conv_memory.current_package}
        Business type saat ini: {conv_memory.business_type}

        Pertanyaan:
        {query}
        """

        response = self.client.chat.completions.create(
            model = LLM_MODEL,
            messages = [{"role": "user", "content": analysis_prompt}],
            temperature = 0,
            response_format = {"type": "json_object"}
        )

        try:
            result = json.loads(response.choices[0].message.content)
            return  (
                result.get("product"),
                result.get("package"),
                result.get("intent"),
                result.get("user_name"),
                result.get("company_name"),
                result.get("business_type")
            )
        except:
            return None, None, None, None, None, None
    
    def generate_response(self, query: str, conv_id: str) -> str:
        """Create jawaban untuk response"""
        conv_memory = self.get_or_create_conversation(conv_id)
        product, package, intent, user, company, business = self.analyze_query(query, conv_memory)

        # Product fallback
        if not product and conv_memory.current_product:
            product = conv_memory.current_product

        # Package fallback
        if not package and conv_memory.current_package:
            package = conv_memory.current_package

        # =============================
        # UPDATE MEMORY
        # =============================
        if product and product != conv_memory.current_product:
            conv_memory.current_product = product
            conv_memory.current_package = ""

        if package:
            conv_memory.current_package = package

        if intent:
            conv_memory.last_intent = intent

        if user:
            conv_memory.user_name = user

        if company:
            conv_memory.company_name = company

        if business:
            conv_memory.business_type = business

        search_query = query

        if conv_memory.current_product:
            search_query += f"produk {conv_memory.current_product}. {search_query}"

        if conv_memory.current_package:
            search_query += f" paket {conv_memory.current_package}. {search_query}"

        similar_chunks, max_similarity = self.embedding_service.search_similar_chunks(
            search_query, top_k = TOP_K
        )

        if max_similarity < self.embedding_service.similarity_threshold:
            assistant_response = "Maaf kak, informasi tersebut belum tersedia di database saya."
            conv_memory.add_exchange(query, assistant_response)
            return {
                "answer": assistant_response,
                "product": conv_memory.current_product,
                "package": conv_memory.current_package,
                "intent": conv_memory.last_intent,
                "user_name": conv_memory.user_name,
                "company_name": conv_memory.company_name,
                "business_type": conv_memory.business_type
            }

        reference_text = ""
        for i, (chunk, _) in enumerate(similar_chunks, 1):
            reference_text += f"\n[DATA {i}]\n{chunk['text']}\n"

        # Identitas user
        user_identity = ""

        if conv_memory.user_name:
            user_identity += f"\nNama klien: {conv_memory.user_name}"

        if conv_memory.company_name:
            user_identity += f"\nNama perusahaan: {conv_memory.company_name}"

        if conv_memory.business_type:
            user_identity += f"\nJenis usaha: {conv_memory.business_type}"
            
        user_message = f"""
        IDENTITAS USER:
        {user_identity}

        PERTANYAAN USER:
        {query}

        DATA REFERENSI:
        {reference_text}

        ATURAN TAMBAHAN:
        - Jika nama user tersedia, panggil dengan sopan.
        - Jangan menyebut nama perusahaan kecuali relevan dengan jawaban.
        - Perhatikan gaya bahasa pada DATA REFERENSI.
        - Jika di referensi tidak ada kalimat penutup, jangan menambahkan kalimat penutup baru.
        - Jangan menambahkan ajakan bertanya kembali kecuali memang relevan.

        ATURAN USAHA:
        - Jika jenis usaha user belum diketahui, jangan memberikan contoh usaha.
        - Jangan mengasumsikan jenis usaha dan bidang usaha user.
        - Jangan menambahkan contoh seperti:
        toko online, bimbel, restoran, klinik, akrilik, dll.
        - Berikan penjelasan layanan secara umum saja.
        - Contoh usaha hanya boleh disebutkan jika user sudah menyebutkan jenis usahanya.

        ATURAN PENTING:
        - Jika di DATA REFERENSI terdapat contoh usaha, atau bidang usaha atau produk yang dijual tetapi user tidak menyebutkan usaha, atau bidang usaha, atau produk yang dijual, maka abaikan bagian tersebut dan jangan menyebutkannya di jawaban.
        - Jangan menambahkan contoh website jika belum diketahui usaha atau produk yang dijual oleh klien.

        - Gunakan referensi hanya sebagai sumber informasi.
        - Tulis ulang jawaban dengan kalimat yang lebih natural.
        """
        
        messages = [{"role": "system", "content": self.system_prompt}]

        for exchange in conv_memory.history[-3:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        messages.append({"role": "user", "content": user_message})

        try:
            # Memanggil LLM OpenAI
            response = self.client.chat.completions.create(
                model = LLM_MODEL,
                messages = messages,
                temperature = 0.3,
                max_tokens = 700
            )
            assistant_response = response.choices[0].message.content.strip()
            
        except Exception:
            assistant_response = "Maaf kak, sedang ada kendala teknis. Bisa dicoba lagi nanti ya."
        
        conv_memory.add_exchange(query, assistant_response)
        return {
            "answer": assistant_response,
            "product": conv_memory.current_product,
            "package": conv_memory.current_package,
            "intent": conv_memory.last_intent,
            "user_name": conv_memory.user_name,
            "company_name": conv_memory.company_name,
            "business_type": conv_memory.business_type
        }
    
    def chat(self, user_input: str, conv_id: str = None):
        return self.generate_response(user_input, conv_id)
    
    def get_conversation_history(self, conv_id: str) -> list:
        """Mengambil history percakapan"""
        if conv_id in self.conversations:
            return self.conversations[conv_id].history
        return []