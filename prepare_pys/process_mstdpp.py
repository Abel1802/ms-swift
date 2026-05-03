import os
import json
import pandas as pd


# ================= Configuration =================
BASE_PATH = "raw_data/mstdpp"

AUDIO_ROOT_DIR = os.path.join(BASE_PATH, "final_utterance_audios")
VIDEO_ROOT_DIR = os.path.join(BASE_PATH, "final_utterance_videos")

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
VALID_CSV = os.path.join(BASE_PATH, "valid.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")

# OUTPUT_DIR = os.path.join(BASE_PATH, "ms_swift")
OUTPUT_DIR = "jsonl_data_train/msdtpp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Candidate column names. The script will automatically choose the first existing one.
ID_COLUMNS = ["KEY", "Key", "key", "id", "ID", "File Name", "file_name", "filename"]
TEXT_COLUMNS = ["SENTENCE", "Sentence", "sentence", "Transcriptions", "transcription", "text", "Text", "utterance"]
LABEL_COLUMNS = ["Sarcasm", "sarcasm", "Labels", "label", "Label", "is_sarcastic"]
# =================================================


def find_column(df, candidates, required=True):
    """Find the first matching column from a list of possible names."""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Cannot find any column from {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def get_audio_path(key):
    return os.path.join(AUDIO_ROOT_DIR, f"{key}.wav")


def get_video_path(key):
    return os.path.join(VIDEO_ROOT_DIR, f"{key}.mp4")


def normalize_key(x):
    """
    Convert:
    123 -> "123"
    123.wav -> "123"
    /path/to/123.wav -> "123"
    """
    x = str(x).strip()
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return x


def label_to_bool(label):
    """
    Convert MUStARD++ sarcasm labels to boolean.

    Supports:
    1, 1.0, True, true, yes, sarcastic, s -> True
    0, 0.0, False, false, no, non-sarcastic, ns -> False
    """
    if pd.isna(label):
        raise ValueError("Label is NaN.")

    label_str = str(label).strip().lower()

    if label_str in ["1", "1.0", "true", "yes", "y", "sarcastic", "s"]:
        return True

    if label_str in ["0", "0.0", "false", "no", "n", "non-sarcastic", "non_sarcastic", "ns"]:
        return False

    try:
        return float(label) > 0.5
    except Exception:
        raise ValueError(f"Cannot convert label to bool: {label}")


def format_prompt_modality(sentence, modalities):
    instruction = "Please determine whether the speaker is expressing sarcasm.\n\n"

    if modalities == "T":
        modality_desc = "Please analyze only the textual content."
        reasoning = "explain your reasoning from the textual perspective."

    elif modalities == "A":
        modality_desc = "Please analyze only the speech prosody."
        reasoning = "explain your reasoning from the acoustic/prosodic perspective."

    elif modalities == "V":
        modality_desc = "Please analyze only the facial expressions."
        reasoning = "explain your reasoning from the visual perspective."

    elif modalities == "T+A":
        modality_desc = "Please analyze both the textual content and speech prosody."
        reasoning = "explain your reasoning from the textual and acoustic perspectives."

    elif modalities == "T+V":
        modality_desc = "Please analyze both the textual content and facial expressions."
        reasoning = "explain your reasoning from the textual and visual perspectives."

    elif modalities == "A+V":
        modality_desc = "Please analyze both speech prosody and facial expressions."
        reasoning = "explain your reasoning from the acoustic and visual perspectives."

    else:  # T+A+V
        modality_desc = "Please analyze the textual content, speech prosody, and facial expressions."
        reasoning = "explain your reasoning from the textual, acoustic, and visual perspectives."

    return (
        f'The utterance is: "{sentence}".\n'
        f"{modality_desc}\n\n"
        f"Step 1: In the <think> tag, {reasoning}\n"
        'Step 2: In the <answer> tag, output only "Yes" or "No".'
    )


def format_prompt(sentence):
    return (
        f'The utterance is: "{sentence}".\n'
        "Please analyze the speech prosody and facial expressions in the video.\n"
        "Determine whether the speaker is expressing sarcasm.\n\n"
        "Step 1: In the <think> tag, explain your reasoning from the textual, acoustic, and visual perspectives.\n"
        'Step 2: In the <answer> tag, output only "Yes" or "No".'
    )


def format_prompt_reasoning(sentence):
    return (
        f'The utterance is: "{sentence}".\n'
        "Please analyze the speech and visual information in the video.\n"
        "Determine whether the speaker is expressing sarcasm.\n\n"
        "Step 1: In the <reasoning> tag, explain your reasoning from the textual, acoustic, and visual perspectives.\n"
        "Step 2: In the <answer> tag, output only 'Yes' or 'No'."
    )


def format_prompt_detailed(sentence):
    return (
        f'<video><audio>The utterance is: "{sentence}".\n'
        "Determine whether the speaker is expressing sarcasm. Sarcasm requires a clear contrast "
        "between the literal expression and the intended meaning. Humor, exaggeration, self-deprecation, "
        "or complaint is not necessarily sarcasm.\n\n"
        "Only output the following format and nothing else:\n"
        "<think>\n"
        "[Textual evidence]: ...\n"
        "[Acoustic evidence]: ...\n"
        "[Visual evidence]: ...\n"
        "[Cross-modal reasoning]: ...\n"
        "</think>\n"
        "<answer>Yes or No</answer>\n\n"
        "Requirements:\n"
        "- [Textual evidence] must analyze only the original utterance.\n"
        "- [Acoustic evidence] must describe only actually audible speech cues; if there is no clear acoustic evidence, write: No strong acoustic evidence.\n"
        "- [Visual evidence] must describe only actually visible expressions or gestures; if there is no clear visual evidence, write: No strong visual evidence.\n"
        "- [Cross-modal reasoning] must explain whether the modalities jointly support sarcasm, or whether the utterance is better explained as humor, exaggeration, self-deprecation, or complaint.\n"
        '- The final line must be either <answer>Yes</answer> or <answer>No</answer>.\n'
        "- Do not hallucinate acoustic or visual evidence."
    )


def build_user_content(text, prompt_type="reasoning", modalities="T+A+V"):
    """
    prompt_type:
    - "basic"
    - "reasoning"
    - "detailed"
    - "modality"

    modalities is used only when prompt_type == "modality".
    """
    if prompt_type == "basic":
        prompt = format_prompt(text)
        return f"<video><audio>{prompt}"

    if prompt_type == "reasoning":
        prompt = format_prompt_reasoning(text)
        return f"<video><audio>{prompt}"

    if prompt_type == "detailed":
        # format_prompt_detailed already includes <video><audio>
        return format_prompt_detailed(text)

    if prompt_type == "modality":
        prompt = format_prompt_modality(text, modalities)

        if modalities == "T":
            return prompt
        elif modalities == "A":
            return f"<audio>{prompt}"
        elif modalities == "V":
            return f"<video>{prompt}"
        elif modalities == "T+A":
            return f"<audio>{prompt}"
        elif modalities == "T+V":
            return f"<video>{prompt}"
        elif modalities == "A+V":
            return f"<video><audio>{prompt}"
        else:
            return f"<video><audio>{prompt}"

    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def process_csv(
    csv_path,
    output_jsonl_path,
    is_test=False,
    prompt_type="reasoning",
    modalities="T+A+V",
    require_audio=True,
    require_video=True,
):
    """Convert a MUStARD++ CSV file into ms-swift compatible JSONL."""
    if not os.path.exists(csv_path):
        print(f"⏭️ Skipped: file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"⏳ Processing {csv_path}, total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    id_col = find_column(df, ID_COLUMNS)
    text_col = find_column(df, TEXT_COLUMNS)
    label_col = find_column(df, LABEL_COLUMNS)

    print(f"Using columns: id={id_col}, text={text_col}, label={label_col}")

    jsonl_data = []
    skipped_missing_file = 0
    skipped_bad_label = 0

    for _, row in df.iterrows():
        key = normalize_key(row[id_col])
        text = str(row[text_col]).strip()

        try:
            is_sarcastic = label_to_bool(row[label_col])
        except Exception as e:
            print(f"⚠️ Bad label, skip KEY={key}, label={row[label_col]}, error={e}")
            skipped_bad_label += 1
            continue

        audio_path = get_audio_path(key)
        video_path = get_video_path(key)

        if require_audio and not os.path.exists(audio_path):
            print(f"⚠️ Missing audio, skip KEY={key}: {audio_path}")
            skipped_missing_file += 1
            continue

        if require_video and not os.path.exists(video_path):
            print(f"⚠️ Missing video, skip KEY={key}: {video_path}")
            skipped_missing_file += 1
            continue

        target_answer = "Yes" if is_sarcastic else "No"
        user_content = build_user_content(
            text=text,
            prompt_type=prompt_type,
            modalities=modalities,
        )

        entry = {
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "label": target_answer,
        }

        # Add media fields only if needed.
        if require_audio or modalities in ["A", "T+A", "A+V", "T+A+V"]:
            entry["audios"] = [audio_path]

        if require_video or modalities in ["V", "T+V", "A+V", "T+A+V"]:
            entry["videos"] = [video_path]

        if is_test:
            entry["messages"].append(
                {"role": "assistant", "content": f"<answer>{target_answer}</answer>"}
            )

        jsonl_data.append(entry)

    if jsonl_data:
        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ Saved {len(jsonl_data)} examples to: {output_jsonl_path}")
        print(f"Skipped missing files: {skipped_missing_file}")
        print(f"Skipped bad labels: {skipped_bad_label}\n")
    else:
        print(f"⚠️ No valid examples saved for {csv_path}")


if __name__ == "__main__":
    # Main trimodal GRPO data.

    process_csv(
        TRAIN_CSV,
        os.path.join(OUTPUT_DIR, "grpo_train_en.jsonl"),
        is_test=True,
        prompt_type="reasoning",
        modalities="T+A+V",
        require_audio=True,
        require_video=True,
    )

    process_csv(
        VALID_CSV,
        os.path.join(OUTPUT_DIR, "grpo_val_en.jsonl"),
        is_test=True,
        prompt_type="reasoning",
        modalities="T+A+V",
        require_audio=True,
        require_video=True,
    )

    process_csv(
        TEST_CSV,
        os.path.join(OUTPUT_DIR, "test_en.jsonl"),
        is_test=True,
        prompt_type="reasoning",
        modalities="T+A+V",
        require_audio=True,
        require_video=True,
    )

    # Example for text-only ablation:
    # process_csv(
    #     TRAIN_CSV,
    #     os.path.join(OUTPUT_DIR, "grpo_train_en_text_only.jsonl"),
    #     is_test=False,
    #     prompt_type="modality",
    #     modalities="T",
    #     require_audio=False,
    #     require_video=False,
    # )