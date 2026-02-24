import json
import os
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

INPUT_FILE = "cleaned_conversations.json"
OUTPUT_FILE = "conversation_embeddings.json"

BUBBLE_PER_CHUNK = 5
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load model embedding local
model = SentenceTransformer(EMBED_MODEL)

# =========================
# LOAD DATA
# =========================

if not os.path.exists(INPUT_FILE):
    print(f"File {INPUT_FILE} tidak ditemukan.")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# BUBBLING + CHUNKING + EMBEDDING
# =========================

vector_store = []

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

        # Skip jika session hanya 1 bubble
        if len(bubbles) <= 1:
            continue

        # =========================
        # CHUNK SETIAP 5 BUBBLE
        # =========================
        for i in range(0, len(bubbles), BUBBLE_PER_CHUNK):

            chunk = bubbles[i:i+BUBBLE_PER_CHUNK]
            chunk_text = "\n".join(chunk)

            # Embedding
            vector = model.encode(chunk_text).tolist()

            vector_store.append({
                "conv_id": conv_id,
                "session_index": session_index,
                "chunk_index": i // BUBBLE_PER_CHUNK,
                "bubble_count": len(chunk),
                "text": chunk_text,
                "vector": vector
            })

# =========================
# SAVE JSON
# =========================

def dump_with_inline_vector(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")

        for idx, item in enumerate(data):
            f.write("  {\n")
            for key, value in item.items():
                if key == "vector":
                    # Vector ditulis 1 baris
                    vector_str = ",".join(f"{v:.8f}" for v in value)
                    f.write(f'    "vector": [{vector_str}]')
                else:
                    json_value = json.dumps(value, ensure_ascii=False)
                    f.write(f'    "{key}": {json_value}')
                # Tambah koma kalau bukan field terakhir
                if key != list(item.keys())[-1]:
                    f.write(",")
                f.write("\n")

            f.write("  }")

            if idx != len(data) - 1:
                f.write(",")

            f.write("\n")

        f.write("]")


dump_with_inline_vector(vector_store, OUTPUT_FILE)

print("Selesai.")
print(f"Total vector dibuat: {len(vector_store)}")
print(f"Disimpan di: {OUTPUT_FILE}")