import json
import os
import re
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
INPUT_FILE = "response.json"
OUTPUT_FILE = "cleaned_conversations.json"
LIMIT_CONVERSATIONS = 20
FOLLOWUP_HOURS = 20

# =========================
# TEXT FILTERING FUNCTIONS
# =========================

# Menghapus spasi berlebih
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Menghapus teks yang hanya berisi link
def is_link_only(text):
    url_pattern = r'^(https?://[^\s]+|www\.[^\s]+)$'
    return re.match(url_pattern, text.lower()) is not None

# Menghapus teks yang hanya berisi emoji
def is_emoji_only(text):
    if not text:
        return True

    emoji_pattern = re.compile(
        r'^[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251\s]+$'
    )

    return bool(emoji_pattern.match(text))

# Menghapus teks yang kurang relevan
def is_low_information(text):
    text = text.strip().lower()

    # Teks terlalu pendek <= 1 huruf
    if len(text) <= 1:
        return True

    # Hanya karakter sama berulang (hehehe, wkwkwk, aaa)
    if len(set(text)) <= 2 and len(text) <= 6:
        return True

    # Tidak ada huruf atau angka (spasi)
    if not any(c.isalnum() for c in text):
        return True

    return False

# =========================
# MAIN PROCESS
# =========================

def run_cleaning_process():

    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} tidak ditemukan.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected_keys = (
        list(data.keys())
        if not LIMIT_CONVERSATIONS
        else list(data.keys())[:LIMIT_CONVERSATIONS]
    )

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

            if role == "media":
                continue

            if not text:
                continue

            text = clean_text(text)

            if is_link_only(text):
                continue

            if is_emoji_only(text):
                continue

            if is_low_information(text):
                continue

            filtered.append({
                "role": role,
                "text": text,
                "created_at": created_at
            })

        if not filtered:
            cleaned_result[conv_id] = []
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
    print(f"Hasil disimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_cleaning_process()