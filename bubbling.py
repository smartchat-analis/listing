import json
import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# LOAD ENV
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# CONFIG
# =========================
INPUT_FILE = "cleaned_conversations.json"
DB_NAME = "knowledge_base.db"
TABLE_NAME = "conversation_embeddings"

BUBBLE_PER_CHUNK = 5
EMBED_MODEL = "text-embedding-3-small"

# Jumlah chunk per request API
BATCH_SIZE = 100  

# =========================
# CONNECT TO SQLITE
# =========================
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# =========================
# CREATE TABLE
# =========================
cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id TEXT,
    chunk_index INTEGER,
    bubble_count INTEGER,
    text TEXT,
    vector TEXT,
    priority INTEGER DEFAULT 0,
    UNIQUE(conv_id, chunk_index)
)
""")

# =========================
# SEARCH INDEX
# =========================
cursor.execute(f"""
CREATE INDEX IF NOT EXISTS idx_conv_id
ON {TABLE_NAME}(conv_id, chunk_index)
""")

conn.commit()

# =========================
# LOAD DATA
# =========================
if not os.path.exists(INPUT_FILE):
    print(f"File {INPUT_FILE} tidak ditemukan.")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# BUBBLING + CHUNKING
# =========================
all_chunks = []
new_conv_ids = set()

for conv_id, sessions in data.items():
    chunk_id = 0
    new_conv_ids.add(conv_id)

    for session_index, session in enumerate(sessions):
        bubbles = []

        for msg in session:
            role = msg.get("role", "").lower()
            text = msg.get("text", "").strip()

            if not text:
                continue

            formatted_text = f"{role.title()}:\n{text}"
            bubbles.append(formatted_text)

        # Skip session yang hanya 1 bubble
        if len(bubbles) <= 1:
            continue

        for i in range(0, len(bubbles), BUBBLE_PER_CHUNK):
            chunk = bubbles[i:i+BUBBLE_PER_CHUNK]

            # Jika sisa 1 percakapan maka digabung ke chunk sebelumnya
            if len(chunk) == 1 and all_chunks:
                all_chunks[-1]["text"] += "\n" + chunk[0]
                all_chunks[-1]["bubble_count"] += 1
                continue

            chunk_text = "\n".join(chunk)

            all_chunks.append({
                "conv_id": conv_id,
                "chunk_index": chunk_id,
                "bubble_count": len(chunk),
                "text": chunk_text
            })

            chunk_id += 1

# =========================
# BATCH EMBEDDING
# =========================
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i+BATCH_SIZE]
    batch_texts = [item["text"] for item in batch]

    print(f"Embedding batch {i//BATCH_SIZE + 1}...")

    response = client.embeddings.create(
        model = EMBED_MODEL,
        input = batch_texts
    )

    embeddings = [item.embedding for item in response.data]

    # =========================
    # INSERT TO SQLITE
    # =========================
    for chunk_data, vector in zip(batch, embeddings):
        vector_json = json.dumps(vector)

        cursor.execute(f"""
        INSERT OR REPLACE INTO {TABLE_NAME}
        (conv_id, chunk_index, bubble_count, text, vector)
        VALUES (?, ?, ?, ?, ?)
        """, (
            chunk_data["conv_id"],
            chunk_data["chunk_index"],
            chunk_data["bubble_count"],
            chunk_data["text"],
            vector_json
        ))

    conn.commit()

# =========================
# CLOSE DATABASE
# =========================
conn.close()

print("Selesai.")
print(f"Disimpan di database {DB_NAME} tabel {TABLE_NAME}")