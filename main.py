import json, random, os, datetime
from pathlib import Path

# ====== 你要先設定 API KEY ======
# 方法 1：直接在終端機先 export
# export OPENAI_API_KEY="你的key"
# 方法 2：或你在這裡填
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise ValueError("找不到 OPENAI_API_KEY。請先在環境變數設定或寫在 main.py。")

# ====== OpenAI SDK ======
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"  # 你也可以改成課堂指定的模型

# ====== 檔案路徑 ======
BASE_DIR = Path(__file__).parent
PROMPT_DIR = BASE_DIR / "prompts"
RUNS_DIR = BASE_DIR / "runs"
CARDS_PATH = BASE_DIR / "cards.json"

RUNS_DIR.mkdir(exist_ok=True)

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def call_llm(prompt: str, variables: dict, json_mode: bool):
    """用 variables 填 prompt 後 call LLM。json_mode=True 會要求模型只回 JSON。"""
    filled = prompt.format(**variables)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "你是嚴謹又友善的塔羅遊戲 LLM 模組。"},
            {"role": "user", "content": filled}
        ],
        response_format={"type": "json_object"} if json_mode else None
    )

    content = response.choices[0].message.content
    if json_mode:
        return json.loads(content)
    return content

def draw_card(cards: list):
    """隨機抽一張牌 + 正逆位"""
    card = random.choice(cards)
    orientation = random.choice(["upright", "reversed"])
    keywords = card["upright_keywords"] if orientation == "upright" else card["reversed_keywords"]
    return card["name"], orientation, keywords

def main():
    # ====== 讀卡牌 ======
    cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))

    # ====== 讀 prompts ======
    intent_prompt = load_text(PROMPT_DIR / "intent_parser.txt")
    first_prompt = load_text(PROMPT_DIR / "interpret_first.txt")
    next_prompt = load_text(PROMPT_DIR / "interpret_next.txt")
    review_prompt = load_text(PROMPT_DIR / "final_review.txt")

    print("=== 塔羅時間序占卜遊戲 ===")
    print("主題選擇：love / study / career / other")

    topic_input = input("請輸入主題： ").strip()
    player_text = input("請描述你的困擾 / 問題： ").strip()

    # ====== Task0：意圖解析 ======
    intent = call_llm(
        intent_prompt,
        {"player_text": f"[topic_hint={topic_input}] {player_text}"},
        json_mode=True
    )

    state = {
        "topic": intent.get("topic", topic_input),
        "question": intent["question"],
        "emotion": intent.get("emotion", "other"),
        "constraints": intent.get("constraints", []),
        "draw_history": []
    }

    positions = ["past", "present", "future"]

    # ====== 逐張抽牌＋解讀 ======
    for step, pos in enumerate(positions, start=1):
        card_name, orientation, keywords = draw_card(cards)

        if step == 1:
            # Task1：解第一張牌
            interp = call_llm(
                first_prompt,
                {
                    "topic": state["topic"],
                    "question": state["question"],
                    "emotion": state["emotion"],
                    "constraints": state["constraints"],
                    "position": pos,
                    "card_name": card_name,
                    "orientation": orientation,
                    "keywords": keywords
                },
                json_mode=True
            )
        else:
            # Task2：解後續牌（吃前面輸出）
            history_json = json.dumps(state["draw_history"], ensure_ascii=False, indent=2)
            interp = call_llm(
                next_prompt,
                {
                    "step": step,
                    "topic": state["topic"],
                    "question": state["question"],
                    "emotion": state["emotion"],
                    "constraints": state["constraints"],
                    "history_json": history_json,
                    "position": pos,
                    "card_name": card_name,
                    "orientation": orientation,
                    "keywords": keywords
                },
                json_mode=True
            )

        record = {
            "step": step,
            "position": pos,
            "card": card_name,
            "orientation": orientation,
            "keywords": keywords,
            "interpretation": interp
        }
        state["draw_history"].append(record)

        # ====== 顯示當下解讀 ======
        print(f"\n--- 第 {step} 張牌 ({pos}) ---")
        print("抽到：", card_name, f"({orientation})")
        print("關鍵字：", "、".join(keywords))
        print("解讀：")
        print(json.dumps(interp, ensure_ascii=False, indent=2))

    # ====== Task3：Final Review ======
    history_json = json.dumps(state["draw_history"], ensure_ascii=False, indent=2)
    review_md = call_llm(
        review_prompt,
        {
            "topic": state["topic"],
            "question": state["question"],
            "history_json": history_json
        },
        json_mode=False
    )

    # ====== 存檔 runs/ ======
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json_path = RUNS_DIR / f"run_{ts}.json"
    run_md_path = RUNS_DIR / f"run_{ts}.md"

    run_json_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    run_md_path.write_text(review_md, encoding="utf-8")

    print("\n=== 本次占卜回顧 Review ===\n")
    print(review_md)
    print(f"\n(已存檔：{run_json_path.name}, {run_md_path.name})")

if __name__ == "__main__":
    main()
