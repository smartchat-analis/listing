import json
import os
import re
from groq import Groq
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load env
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY tidak ditemukan di file .env")

# --- CONFIGURATION ---
INPUT_FILE = 'response.json'
OUTPUT_FILE = 'cleaned_conversations.json'
LIMIT_CONVERSATIONS = 10
MODEL_NAME = "llama-3.1-8b-instant"

FOLLOWUP_HOURS = 20

client = Groq(api_key=API_KEY)

# =========================
# FUNCTION
# =========================

def is_link_only(text):
    url_pattern = r'^(https?://[^\s]+|www\.[^\s]+)$'
    return re.match(url_pattern, text.strip().lower()) is not None

def is_emoji_only(text):
    """
    Return True jika text hanya berisi emoji / simbol / whitespace
    """
    if not text:
        return True

    cleaned = text.strip()

    # Hanya emoji & simbol unicode umum
    emoji_pattern = re.compile(
        r'^[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251\s]+$'
    )

    return bool(emoji_pattern.match(cleaned))

def clean_batch_messages(messages_list):
    if not messages_list:
        return []

    content_to_check = "\n".join([f"- {m}" for m in messages_list])

    prompt = f"""
    Tugas: Filter pesan berikut.
    Pisahkan mana percakapan manusia asli dan mana noise sistem.
    Kembalikan JSON array boolean.
    Pesan:
    {content_to_check}
    """

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            response_format={"type": "json_object"}
        )
        res = json.loads(completion.choices[0].message.content)
        for key in res:
            if isinstance(res[key], list):
                return res[key]
        return [True] * len(messages_list)
    except:
        return [True] * len(messages_list)

# =========================
# MAIN PROCESS
# =========================

def run_cleaning_process():

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} tidak ditemukan.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    selected_keys = (
        list(data.keys())
        if not LIMIT_CONVERSATIONS
        else list(data.keys())[:LIMIT_CONVERSATIONS]
    )

    cleaned_result = {}

    for conv_id in selected_keys:
        print(f"Cleaning Conversation: {conv_id}...")
        raw_chats = data[conv_id]

        # =========================
        # 1️⃣ FILTER AWAL
        # =========================
        filtered_locally = []
        texts_to_ai = []

        for chat in raw_chats:
            role = chat.get('role', '').lower()
            text = chat.get('chat', '')

            # Membuang role = media
            if role == 'media':
                continue
            
            # Membuang pesan yang hanya berisi link saja
            if not text or is_link_only(text):
                continue

            # Membuang pesan kosong
            if not text:
                continue

            # Membuang pesan yang hanya berisi emoji
            if is_emoji_only(text):
                continue

            filtered_locally.append(chat)
            texts_to_ai.append(text)

        # =========================
        # 2️⃣ FILTER AI
        # =========================
        current_conv_flow = []

        if texts_to_ai:
            valid_map = clean_batch_messages(texts_to_ai)

            for idx, chat_item in enumerate(filtered_locally):
                is_valid = valid_map[idx] if idx < len(valid_map) else True

                if is_valid:
                    current_conv_flow.append({
                        "created_at": chat_item.get('created_at'),
                        "role": chat_item.get('role').lower(),
                        "text": chat_item.get('chat').strip()
                    })

        if not current_conv_flow:
            cleaned_result[conv_id] = []
            continue

        # =========================
        # 3️⃣ MERGE CONSECUTIVE ROLE (WITH TIME CHECK)
        # =========================
        merged_messages = []

        buffer_role = None
        buffer_text = []
        buffer_time = None

        for msg in current_conv_flow:

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

            # Merge hanya jika role sama dan beda waktu <=20 jam
            if role == buffer_role and time_diff <= timedelta(hours=FOLLOWUP_HOURS):
                buffer_text.append(text)
            else:
                merged_messages.append({
                    "role": buffer_role,
                    "text": " ".join(buffer_text),
                    "created_at": buffer_time.isoformat()
                })

                buffer_role = role
                buffer_text = [text]
                buffer_time = time

        if buffer_role is not None:
            merged_messages.append({
                "role": buffer_role,
                "text": " ".join(buffer_text),
                "created_at": buffer_time.isoformat()
            })

        # =========================
        # 4️⃣ SESSION SPLIT (FOLLOW-UP FIXED)
        # =========================
        sessions = []
        current_session = []
        last_user_time = None

        for msg in merged_messages:

            current_time = datetime.fromisoformat(msg["created_at"])
            role = msg["role"]

            if role == "user":
                last_user_time = current_time

            # Jika assistant >20 jam dari user terakhir → session baru
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
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_result, f, indent=2, ensure_ascii=False)

    print("\nSelesai.")
    print(f"Hasil disimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_cleaning_process()