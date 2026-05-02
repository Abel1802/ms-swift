import pandas as pd
import json
import os

# ================= 配置区域 =================
BASE_PATH = '/projects/0/prjs0864/phd_projects/IS26/ms-swift/raw_data/MCSD1'

AUDIO_ROOT_DIR = os.path.join(BASE_PATH, 'audios')
VIDEO_ROOT_DIR = os.path.join(BASE_PATH, 'videos')

TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
VALID_CSV = os.path.join(BASE_PATH, 'valid.csv')
TEST_CSV  = os.path.join(BASE_PATH, 'test.csv')

OUTPUT_DIR = os.path.join(BASE_PATH, 'ms_swift') 
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===========================================

def get_audio_path(key):
    return os.path.join(AUDIO_ROOT_DIR, f"{key}.wav")

def get_video_path(key):
    return os.path.join(VIDEO_ROOT_DIR, f"{key}.mp4")

def format_prompt_modality(sentence, modalities):
    instruction = "请判断说话者是否在表达讽刺（sarcasm）。\n\n"

    if modalities == "T":
        modality_desc = "请仅根据文本内容进行分析。"
        reasoning = "从文本角度说明你的推理过程。"

    elif modalities == "A":
        modality_desc = "请仅根据语音语调（prosody）进行分析。"
        reasoning = "从语音角度说明你的推理过程。"

    elif modalities == "V":
        modality_desc = "请仅根据面部表情（facial expressions）进行分析。"
        reasoning = "从视觉角度说明你的推理过程。"

    elif modalities == "T+A":
        modality_desc = "请结合文本和语音语调进行分析。"
        reasoning = "从文本和语音两个方面说明推理过程。"

    elif modalities == "T+V":
        modality_desc = "请结合文本和面部表情进行分析。"
        reasoning = "从文本和视觉两个方面说明推理过程。"

    elif modalities == "A+V":
        modality_desc = "请结合语音语调和面部表情进行分析。"
        reasoning = "从语音和视觉两个方面说明推理过程。"

    else:  # T+A+V
        modality_desc = "请结合文本、语音语调和面部表情进行分析。"
        reasoning = "从文本、语音和视觉三个方面说明推理过程。"

    return (
        f"以下语句为：“{sentence}”。\n"
        f"{modality_desc}\n\n"
        f"步骤1：在<think>标签中，{reasoning}\n"
        "步骤2：在<answer>标签中，仅输出“Yes”或“No”。"
    )

def format_prompt(sentence):
    return (
        f"以下语句为：“{sentence}”。\n"
        "请结合视频中的语音语调（prosody）和面部表情（facial expressions）进行分析。\n"
        "判断说话者是否在表达讽刺（sarcasm）。\n\n"
        "步骤1：在<think>标签中，从文本、语音和视觉三个方面详细说明你的推理过程。\n"
        "步骤2：在<answer>标签中，仅输出“Yes”或“No”。"
    )


def format_prompt_reasoning(sentence):
    return (
        f"以下语句为：“{sentence}”。\n"
        "请结合视频中的语音和视觉信息进行分析。\n"
        "判断说话者是否在表达讽刺（sarcasm）。\n\n"
        "步骤1：在<reasoning>标签中，从文本(只分析原句内容)、语音(只描述实际听到的语音,包括音高、重音、语速等)和视觉(只描述实际看到的表情或动作)三个方面总结你的推理过程。\n"
        "步骤2：在<answer>标签中，仅输出“Yes”或“No”。"
    )


def format_prompt_detailed(sentence):
    return (
        f"<video><audio>以下语句为：“{sentence}”。\n"
        "判断说话者是否在表达讽刺（sarcasm）。讽刺要求字面表达与真实意图之间存在明显反差。"
        "幽默、夸张、自嘲、吐槽不一定是讽刺。\n\n"
        "只允许输出以下格式，不能输出其他内容：\n"
        "<think>\n"
        "[Textual evidence]: ...\n"
        "[Acoustic evidence]: ...\n"
        "[Visual evidence]: ...\n"
        "[Cross-modal reasoning]: ...\n"
        "</think>\n"
        "<answer>Yes or No</answer>\n"
        "要求：\n"
        "- [Textual evidence] 只分析原句内容。\n"
        "- [Acoustic evidence] 只描述实际听到的语音线索；不明显则写 No strong acoustic evidence.\n"
        "- [Visual evidence] 只描述实际看到的表情或动作；不明显则写 No strong visual evidence.\n"
        "- [Cross-modal reasoning] 说明三种模态是否支持讽刺，或是否更像幽默/夸张/自嘲/吐槽。\n"
        "- 最后一行必须是 <answer>Yes</answer> 或 <answer>No</answer>。\n"
        "- 禁止脑补音频或视觉证据。"
    )

def string2float(s):
    """将字符串转换为浮点数"""
    if s == 's':
        return 1.0
    elif s == 'ns':
        return 0.0
    else:
        print(f"⚠️ 输入字符串 {s} 无法转换为浮点数")

def process_csv(csv_path, output_jsonl_path, is_test=False):
    """处理CSV并转换为 ms-swift 兼容的 JSONL 格式"""
    if not os.path.exists(csv_path):
        print(f"⏭️ 跳过: 未找到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"⏳ 正在处理 {csv_path}，共计 {len(df)} 条数据...")
    
    jsonl_data = []

    for _, row in df.iterrows():
        key = str(row['File Name'])
        text = str(row['Transcriptions'])
        
        # 标签处理：大于0.5视为反语
        is_sarcastic = string2float(row['Labels']) > 0.5

        audio_path = get_audio_path(key)
        video_path = get_video_path(key)

        # 校验音视频文件是否双双存在
        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            print(f"⚠️ 缺失文件，跳过 KEY: {key}")
            continue

        target_answer = "Yes" if is_sarcastic else "No"
        # user_content = f"<video><audio>{format_prompt(text)}"

        # user_content = f"{format_prompt_modality(text, 'T')}"
        # user_content = f"<video>{format_prompt_modality(text, 'T+V')}"
        # user_content = f"<video><audio>{format_prompt_detailed(text)}"
        user_content = f"<video><audio>{format_prompt_reasoning(text)}"

        # 构建统一的字典结构 (训练、验证、测试均使用此格式)
        if is_test:
            jsonl_data.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"<answer>{target_answer}</answer>"}
            ],
            "audios": [audio_path],
            "videos": [video_path],
            "solution": target_answer
        })
        else:
            jsonl_data.append({
            "messages": [
                {"role": "user", "content": user_content},
                # {"role": "assistant", "content": f"<answer>{target_answer}</answer>"}
                
            ],
            "audios": [audio_path],
            "videos": [video_path],
            "solution": target_answer,
        })

    # 保存 JSONL 文件
    if jsonl_data:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ 成功保存 {len(jsonl_data)} 条数据至: {output_jsonl_path}\n")

if __name__ == "__main__":
    # 处理训练集
    # process_csv(TRAIN_CSV, os.path.join(OUTPUT_DIR, 'grpo_train_assistant_zh.jsonl'))
    process_csv(TRAIN_CSV, os.path.join(OUTPUT_DIR, 'grpo_train_zh.jsonl'))
    
    # 处理验证集
    # process_csv(VALID_CSV, os.path.join(OUTPUT_DIR, 'grpo_val_assistant_zh.jsonl'))
    process_csv(VALID_CSV, os.path.join(OUTPUT_DIR, 'grpo_val_zh.jsonl'))
    
    # 处理测试集 (移除 is_test 参数，统一处理)
    process_csv(TEST_CSV, os.path.join(OUTPUT_DIR, 'test_zh.jsonl'), is_test=True)