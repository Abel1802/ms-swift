import pandas as pd
import json
import os

# ================= 配置区域 =================
BASE_PATH = 'raw_data/MCSD1'

AUDIO_ROOT_DIR = os.path.join(BASE_PATH, 'audios')
VIDEO_ROOT_DIR = os.path.join(BASE_PATH, 'videos')

TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
VALID_CSV = os.path.join(BASE_PATH, 'valid.csv')
TEST_CSV  = os.path.join(BASE_PATH, 'test.csv')

OUTPUT_DIR = os.path.join(BASE_PATH, 'ms_swift_video_audio') 
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===========================================

def get_audio_path(key):
    return os.path.join(AUDIO_ROOT_DIR, f"{key}.wav")

def get_video_path(key):
    return os.path.join(VIDEO_ROOT_DIR, f"{key}.mp4")


# def format_prompt(sentence):
#     """构建包含思维链(CoT)的多模态Prompt"""
#     return (
#         f"The following utterance is: \"{sentence}\".\n"
#         "Please analyze the speaker's tone, prosody, and facial expressions from the video and audio.\n" 
#         "Determine if the speaker is being sarcastic.\n\n"
#         "Step 1: In the <think> tags, explain your reasoning based on audio, visual, and text clues.\n"
#         "Step 2: In the <answer> tags, output 'Yes' if it is sarcastic, or 'No' otherwise."
#     )


def format_prompt(sentence):
    return (
        f"以下语句为：“{sentence}”。\n"
        "请结合视频中的语音语调（prosody）和面部表情（facial expressions）进行分析。\n"
        "判断说话者是否在表达讽刺（sarcasm）。\n\n"
        "步骤1：在<think>标签中，从文本、语音和视觉三个方面详细说明你的推理过程。\n"
        "步骤2：在<answer>标签中，仅输出“Yes”或“No”。"
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
        user_content = f"<video><audio>{format_prompt(text)}"

        # 构建统一的字典结构 (训练、验证、测试均使用此格式)
        if is_test:
            jsonl_data.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"<answer>{target_answer}</answer>"}
            ],
            "audios": [audio_path],
            "videos": [video_path],
        })
        else:
            jsonl_data.append({
            "messages": [
                {"role": "user", "content": user_content},
                # {"role": "assistant", "content": f"<answer>{target_answer}</answer>"}
                
            ],
            "audios": [audio_path],
            "videos": [video_path],
            "label": target_answer,
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