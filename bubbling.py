import json
import os
import re
from datetime import datetime
from groq import Groq

# --- CONFIGURATION ---
GROQ_API_KEY = ""
INPUT_FILE = 'response.json'
OUTPUT_FILE = 'final_data.json' 
LIMIT_CONVERSATIONS = 3
MODEL_NAME = "llama-3.1-8b-instant"
MAX_HOURS_GAP = 20 

client = Groq(api_key=GROQ_API_KEY)

# Global Store untuk Node unik
UNIQUE_NODES = {}
NODE_COUNTER = 1

def get_detailed_intent(text, role):
    prompt = f"""
    Tentukan intent dari chat {role} berikut. 
    Intent harus PADAT dan konsisten agar bisa digabungkan jika konteksnya mirip.
    Contoh: "menanyakan biaya SEO", "menyapa dan memperkenalkan diri", "memberikan link website".
    
    Teks: "{text}"
    WAJIB JSON: {{"intent": "..."}}
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            response_format={"type": "json_object"}
        )
        res = json.loads(completion.choices[0].message.content)
        return str(res.get('intent', f"pesan {role}")).lower().strip()
    except:
        return f"percakapan {role}"

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def calculate_hours_gap(time_str1, time_str2):
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    try:
        t1, t2 = datetime.strptime(time_str1, fmt), datetime.strptime(time_str2, fmt)
        return abs((t2 - t1).total_seconds()) / 3600
    except: return 0

def process_merged_flow():
    global NODE_COUNTER
    if not os.path.exists(INPUT_FILE): return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    selected_keys = list(data.keys())[:LIMIT_CONVERSATIONS]
    
    # Grouping & Intent Analysis per Percakapan
    all_conv_steps = [] # List of list of node_info

    for conv_id in selected_keys:
        raw_chats = data[conv_id]
        grouped = []
        current = None

        for chat_item in raw_chats:
            role, text, time = chat_item['role'], chat_item['chat'], chat_item['created_at']
            if current and current['role'] == role and calculate_hours_gap(current['last_time'], time) < MAX_HOURS_GAP:
                current['texts'].append(text)
                current['last_time'] = time
            else:
                if current: grouped.append(current)
                current = {"role": role, "texts": [text], "last_time": time}
        if current: grouped.append(current)

        # Analisis Intent untuk tiap grup
        steps = []
        for g in grouped:
            txt = "\n\n".join(g['texts'])
            intent = get_detailed_intent(txt, g['role'])
            steps.append({"role": g['role'], "intent": intent, "chat": txt})
        all_conv_steps.append(steps)

    # Merging ke Global Structure
    intent_to_node_id = {} # Map "intent_role" -> "N1"

    for steps in all_conv_steps:
        for i, step in enumerate(steps):
            key = f"{slugify(step['intent'])}_{step['role']}"
            
            # Jika intent + role belum pernah ada, buat Node baru
            if key not in intent_to_node_id:
                node_id = f"N{NODE_COUNTER}"
                intent_to_node_id[key] = node_id
                UNIQUE_NODES[node_id] = {
                    "intent": step['intent'],
                    "role": step['role'],
                    "texts": [{"chat": step['chat']}],
                    "answers": {}
                }
                NODE_COUNTER += 1
            else:
                node_id = intent_to_node_id[key]
                if {"chat": step['chat']} not in UNIQUE_NODES[node_id]["texts"]:
                    UNIQUE_NODES[node_id]["texts"].append({"chat": step['chat']})

            # Hubungkan ke langkah berikutnya (Branching)
            if i < len(steps) - 1:
                next_step = steps[i+1]
                next_key = f"{slugify(next_step['intent'])}_{next_step['role']}"
                
                # ID node tujuan
                if next_key not in intent_to_node_id:
                    next_node_id = f"N{NODE_COUNTER}"
                    intent_to_node_id[next_key] = next_node_id
                    UNIQUE_NODES[next_node_id] = {
                        "intent": next_step['intent'],
                        "role": next_step['role'],
                        "texts": [{"chat": next_step['chat']}],
                        "answers": {}
                    }
                    NODE_COUNTER += 1
                else:
                    next_node_id = intent_to_node_id[next_key]

                # Masukkan ke answers
                ans_key = next_step['intent']
                if ans_key not in UNIQUE_NODES[node_id]["answers"]:
                    UNIQUE_NODES[node_id]["answers"][ans_key] = []
                
                # Tambahkan link ke node tujuan jika belum ada
                if {"to": next_node_id} not in UNIQUE_NODES[node_id]["answers"][ans_key]:
                    UNIQUE_NODES[node_id]["answers"][ans_key].append({"to": next_node_id})

    # Simpan Hasil
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(UNIQUE_NODES, f, indent=2, ensure_ascii=False)
    
    print(f"Berhasil Merging! Total Node Unik: {len(UNIQUE_NODES)}")

if __name__ == "__main__":
    process_merged_flow()