import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# LOAD ENV
# =========================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# =========================
# CONFIG
# =========================
INPUT_FILE = "cleaned_conversations.json"
OUTPUT_FILE = "conversation_embeddings.json"

BUBBLE_PER_CHUNK = 5
EMBED_MODEL = "text-embedding-3-small"

# Jumlah chunk per request API
BATCH_SIZE = 100  

# =========================
# LOAD DATA
# =========================
if not os.path.exists(INPUT_FILE):
    print(f"File {INPUT_FILE} tidak ditemukan.")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# 1. BUBBLING + CHUNKING
# =========================

all_chunks = []

for conv_id, sessions in data.items():
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
            chunk_text = "\n".join(chunk)

            all_chunks.append({
                "conv_id": conv_id,
                "session_index": session_index,
                "chunk_index": i // BUBBLE_PER_CHUNK,
                "bubble_count": len(chunk),
                "text": chunk_text
            })

# =========================
# 2. BATCH EMBEDDING
# =========================

vector_store = []

for i in range(0, len(all_chunks), BATCH_SIZE):

    batch = all_chunks[i:i+BATCH_SIZE]
    batch_texts = [item["text"] for item in batch]

    print(f"Embedding batch {i//BATCH_SIZE + 1}...")

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch_texts
    )

    embeddings = [item.embedding for item in response.data]

    # Pasangkan kembali embedding ke metadata
    for chunk_data, vector in zip(batch, embeddings):
        chunk_data["vector"] = vector
        vector_store.append(chunk_data)

# =========================
# 3. SAVE JSON
# =========================

def dump_with_inline_vector(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")

        for idx, item in enumerate(data):
            f.write("  {\n")
            keys = list(item.keys())

            for key in keys:
                value = item[key]

                if key == "vector":
                    vector_str = ",".join(f"{v:.8f}" for v in value)
                    f.write(f'    "vector": [{vector_str}]')
                else:
                    json_value = json.dumps(value, ensure_ascii=False)
                    f.write(f'    "{key}": {json_value}')

                if key != keys[-1]:
                    f.write(",")

                f.write("\n")

            f.write("  }")

            if idx != len(data) - 1:
                f.write(",")

            f.write("\n")

        f.write("]")

dump_with_inline_vector(vector_store, OUTPUT_FILE)

print("Selesai.")
print(f"Disimpan di: {OUTPUT_FILE}")