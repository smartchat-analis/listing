import requests
import json
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================
CONV_IDS = list(range(11500, 11551))
OUTPUT_FILE = "cleaned_conversations.json"
LIMIT_CONVERSATIONS = 100
FOLLOWUP_HOURS = 20

# =========================
# LOAD ENV
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# GET DATA
# =========================
URL = os.getenv("URL_API")

def get_data(conversation_ids):
    """Mengambil data untuk dicleaning"""
    params = {
        "conversation_ids": ",".join(map(str, conversation_ids))
    }

    response = requests.get(URL, params=params)
    response.raise_for_status()

    data = response.json()
    return data

# =========================
# TEXT FILTERING FUNCTIONS
# =========================
def clean_text(text):
    """Menghapus spasi berlebih"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_link_only(text):
    """Menghapus teks yang hanya berisi link"""
    url_pattern = r'^(https?://[^\s]+|www\.[^\s]+)$'
    return re.match(url_pattern, text.lower()) is not None

def is_emoji_only(text):
    """Menghapus teks yang hanya berisi emoji"""
    if not text:
        return True

    emoji_pattern = re.compile(
        r'^[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251\s]+$'
    )

    return bool(emoji_pattern.match(text))

def is_low_information(text):
    """Menghapus teks yang kurang relevan"""
    text = text.strip().lower()

    # Teks terlalu pendek <= 1 huruf
    if len(text) <= 1:
        return True

    # Tidak ada huruf atau angka (spasi)
    if not any(c.isalnum() for c in text):
        return True

    return False

def remove_punctuation(text):
    """Hapus tanda baca berlebihan"""
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    text = re.sub(r'\s+([!?.,])', r'\1', text)
    return text.strip()

def remove_number_censored(text):
    """Hapus kata 'number censored' saja"""
    text = re.sub(r'\bnumber censored\b', '', text, flags=re.IGNORECASE)
    return text.strip()

# =========================
# LLM BATCH FILTERING FUNCTIONS
# =========================
def trigger_llm(text):
    triggers = ["hehe", "wkwk", "hmmm"]
    return any(w in text.lower() for w in triggers)

def llm_filter_batch(text):
    """Prompt filter teks menggunakan LLM"""
    if not text:
        return []

    prompt_messages = []
    batch_text = """Bersihkan setiap message berikut:
    - Hapus kata/kalimat tidak relevan (misal: filler seperti hehe, wkwk)
    - Jangan menghapus emoji jika masih ada teks lain
    - Jangan ubah kata lain atau struktur kalimat
    - Output hanya teks bersih tanpa penjelasan tambahan
    """

    for i, msg in enumerate(text, 1):
        batch_text += f"Message {i}: \"{msg}\"\n"

    batch_text += "\nKembalikan hasil dalam format:\nMessage 1: <hasil>\nMessage 2: <hasil>\n..."

    prompt_messages = [
        {"role": "system", "content": "Kamu adalah filter teks percakapan, hanya menghapus noise."},
        {"role": "user", "content": batch_text}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=prompt_messages
        )
        result_text = response.choices[0].message.content.strip()

    except Exception as e:
        print("LLM batch error:", e)
        return text

    # Memisahkan setiap message
    cleaned = []
    for i in range(1, len(text)+1):
        pattern = rf"Message {i}:\s*(.*)"
        match = re.search(pattern, result_text)
        if match:
            cleaned.append(match.group(1).strip())
        else:
            cleaned.append(text[i-1])
    return cleaned

# =========================
# MAIN PROCESS
# =========================
def run_cleaning_process():
    """Menjalankan proses mengambil dan cleaning data"""
    data = get_data(CONV_IDS)
    keys = list(data.keys())

    if LIMIT_CONVERSATIONS:
        selected_keys = keys[:LIMIT_CONVERSATIONS]
    else:
        selected_keys = keys

    cleaned_result = {}

    for conv_id in selected_keys:
        print(f"Processing conversation: {conv_id}")

        raw_chats = data[conv_id]
        filtered = []

        # =========================
        # 1. BASIC FILTERING
        # =========================
        for chat in raw_chats:
            role = chat.get("role", "").lower()
            text = chat.get("chat", "")
            created_at = chat.get("created_at")

            if role == "media" or not text:
                continue

            text = clean_text(text)
            text = remove_number_censored(text)

            if is_link_only(text) or is_emoji_only(text) or is_low_information(text):
                continue

            filtered.append({
                "role": role,
                "text": text,
                "created_at": created_at
            })

        if not filtered:
            continue

        # =========================
        # 2. MERGE CONSECUTIVE ROLE
        # =========================
        merged_messages = []
        buffer_role = None
        buffer_text = []
        buffer_time = None

        for msg in filtered:
            role = msg["role"]
            text = msg["text"]
            time = datetime.fromisoformat(
                msg["created_at"].replace("Z", "")
            )

            if buffer_role is None:
                buffer_role = role
                buffer_text = [text]
                buffer_time = time
                continue

            time_diff = time - buffer_time

            if role == buffer_role and time_diff <= timedelta(hours=FOLLOWUP_HOURS):
                buffer_text.append(text)
            else:
                combined_text = " ".join(buffer_text)
                combined_text = remove_punctuation(combined_text).strip()

                if combined_text:
                    if trigger_llm(combined_text):
                        cleaned_text = llm_filter_batch([combined_text])[0].strip()
                    else:
                        cleaned_text = combined_text

                    if cleaned_text and not is_emoji_only(cleaned_text):
                        merged_messages.append({
                            "role": buffer_role,
                            "text": cleaned_text,
                            "created_at": buffer_time.isoformat()
                        })

                buffer_role = role
                buffer_text = [text]
                buffer_time = time

        if buffer_role is not None:
            combined_text = " ".join(buffer_text)
            combined_text = remove_punctuation(combined_text).strip()

            if combined_text:
                if trigger_llm(combined_text):
                    cleaned_text = llm_filter_batch([combined_text])[0].strip()
                else:
                    cleaned_text = combined_text

                if cleaned_text and not is_emoji_only(cleaned_text):
                    merged_messages.append({
                        "role": buffer_role,
                        "text": cleaned_text,
                        "created_at": buffer_time.isoformat()
                    })
        
        if not merged_messages:
            continue
                    
        # =========================
        # 3. SESSION SPLIT (>20 JAM)
        # =========================
        sessions = []
        current_session = []
        last_user_time = None

        for msg in merged_messages:
            current_time = datetime.fromisoformat(msg["created_at"])
            role = msg["role"]

            if role == "user":
                last_user_time = current_time

            # Assistant follow-up lebih dari 20 jam
            if (
                role in ["assistant", "agent"]
                and last_user_time is not None
                and (current_time - last_user_time) > timedelta(hours=FOLLOWUP_HOURS)
            ):
                if current_session:
                    sessions.append(current_session)

                current_session = [msg]
                continue

            current_session.append(msg)

        if current_session:
            sessions.append(current_session)

        cleaned_result[conv_id] = sessions

    # =========================
    # SAVE OUTPUT
    # =========================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_result, f, indent=2, ensure_ascii=False)

    print("\nSelesai.")
    print(f"Hasil ditambahkan ke: {OUTPUT_FILE}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_cleaning_process()