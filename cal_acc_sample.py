import json
import re
import os
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ================= 配置区域 =================
# 你的 CSV 真实标签文件路径
CSV_PATH = "raw_data/mstdpp/utterance_data.csv"
# 你刚生成的 JSONL 预测文件路径

PRED_JSONL_PATH = "jsonl_data_train/msdtpp/sample_out/val_sample.jsonl"

def load_ground_truth(csv_path):
    """读取 CSV 文件，构建 KEY -> Label 的映射字典"""
    df = pd.read_csv(csv_path)
    label_dict = {}
    for _, row in df.iterrows():
        key = str(row['KEY'])
        label_dict[key] = 1 if float(row['Sarcasm']) > 0.5 else 0
    return label_dict


def load_ground_truth_mcsd():
    '''读取 CSV 文件，构建 KEY -> Label 的映射字典，适用于 MCSD 数据集'''
    CSV_PATH = "raw_data/MCSD1/sarcasm_labels.csv"
    df = pd.read_csv(CSV_PATH)
    label_dict = {}
    for _, row in df.iterrows():
        key = str(row['File Name'])
        sarcasm_label = str(row['Labels'])
        if sarcasm_label == 's':
            label_dict[key] = 1
        elif sarcasm_label == 'ns':
            label_dict[key] = 0
        else:
            raise ValueError(f"Invalid sarcasm label: {sarcasm_label}")
    print("samples:", len(label_dict))
    return label_dict

def extract_answer(text):
    """从大模型回复中正则提取 <answer> 标签的内容"""
    match = re.search(r"<answer>\s*(Yes|No)\s*</answer>", text, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0
    return None

def evaluate(pred_file, label_dict):
    """评估 JSONL 结果 (严格逐行评估，带数据对账功能)"""
    valid_preds, valid_labels = [], []   # 只有效格式的样本
    total_preds, total_labels = [], []   # 全量测试样本 (格式失败记为错)
    
    # 对账计数器
    total_lines = 0
    missing_video_path = 0
    unmatched_keys = 0
    missing_answer_count = 0

    print(f"⏳ 正在解析预测文件: {pred_file}")
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_lines += 1  # 记录总行数
            data = json.loads(line)
            
            # 1. 提取 KEY
            videos = data.get("videos", [])
            if not videos or not videos[0]:
                missing_video_path += 1
                continue
            
            video_path = videos[0]
            uid = os.path.splitext(os.path.basename(video_path))[0]
            
            # 2. 匹配真实标签
            if uid not in label_dict:
                unmatched_keys += 1
                continue
                
            true_label = label_dict[uid]
            
            # 只要在 CSV 中找到了标签，这就是一条需要被全量评测的样本
            total_labels.append(true_label)
            
            # 3. 提取助手的回答内容
            assistant_content = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    break 
            
            # 4. 正则提取 1 或 0
            pred = extract_answer(assistant_content)
            
            if pred is not None:
                # 格式正确：加入局部评测，同时也加入全局评测
                valid_preds.append(pred)
                valid_labels.append(true_label)
                total_preds.append(pred)
            else:
                # 格式失败：不加入局部评测，但在全局评测中强行记为错误
                missing_answer_count += 1
                wrong_pred = 1 - true_label 
                total_preds.append(wrong_pred)

    # ================= 打印对账单 =================
    print("\n" + "="*60)
    print("📋 数据对账单 (Data Reconciliation)")
    print("="*60)
    print(f"📄 读取 JSONL 总行数        : {total_lines}")
    print(f"⚠️ 缺失 video 路径被跳过    : {missing_video_path}")
    print(f"⚠️ CSV 找不到标签被跳过     : {unmatched_keys}")
    print(f"❌ 格式崩溃 (计入全局惩罚)  : {missing_answer_count}")
    print(f"✅ 格式正确 (计入局部评测)  : {len(valid_preds)}")
    print("-" * 60)
    print(f"🔍 理论全量评测总数 ({total_lines} - {missing_video_path} - {unmatched_keys}) = {len(total_preds)}")
    print("="*60)

    if len(total_preds) == 0:
        print("❌ 错误：没有有效的数据可供计算！")
        return

    # ================= 打印评测报告 =================
    print("\n" + "="*60)
    print("📊 评测报告 (逐行独立评测)")
    print("="*60)

    # --- 报告 1：仅计算格式正确的样本 ---
    print(f"🟢 【局部视角】有效格式样本指标 (Valid Format Only)")
    print(f"   参与计算的样本数: {len(valid_preds)}")
    if valid_preds:
        v_acc = accuracy_score(valid_labels, valid_preds)
        v_prec = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
        v_rec = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
        v_f1 = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
        print(f"   🏆 Accuracy  : {v_acc:.4f}  ({v_acc*100:.2f}%)")
        print(f"   🎯 Precision : {v_prec:.4f}")
        print(f"   🔍 Recall    : {v_rec:.4f}")
        print(f"   🔥 F1 Score  : {v_f1:.4f}")
    
    print("-" * 60)

    # --- 报告 2：计算全量测试集样本 ---
    print(f"🔵 【全局视角】全量测试集指标 (Total Test Set - 包含格式惩罚)")
    print(f"   参与计算的样本数: {len(total_preds)}")
    t_acc = accuracy_score(total_labels, total_preds)
    t_prec = precision_score(total_labels, total_preds, average='macro', zero_division=0)
    t_rec = recall_score(total_labels, total_preds, average='macro', zero_division=0)
    t_f1 = f1_score(total_labels, total_preds, average='macro', zero_division=0)
    print(f"   🏆 Accuracy  : {t_acc:.4f}  ({t_acc*100:.2f}%)")
    print(f"   🎯 Precision : {t_prec:.4f}")
    print(f"   🔍 Recall    : {t_rec:.4f}")
    print(f"   🔥 F1 Score  : {t_f1:.4f}")
    print("="*60 + "\n")

    # ================= 新增：打印混淆矩阵 =================
    cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
    print("============================================================")
    print("🧮 混淆矩阵 (Confusion Matrix - 全局视角)")
    print("============================================================")
    print("👇 请直接将以下代码块复制到画图脚本中：")
    print(f"np.array([\n    [{cm[0][0]}, {cm[0][1]}],  # Non-Sarcasm 真实标签 (TN, FP)\n    [{cm[1][0]}, {cm[1][1]}]   # Sarcasm 真实标签 (FN, TP)\n])")
    print("============================================================")


if __name__ == "__main__":
    gt_dict = load_ground_truth(CSV_PATH)
    # gt_dict = load_ground_truth_mcsd()

    evaluate(PRED_JSONL_PATH, gt_dict)
    
    # for epoch in [50, 100, 150, 200, 250, 300, 350, 400, 420]:
    #     # pred_jsonl_path = f"shared_output/omni_grpo_a_v_omni3_reason_correct/v6-20260221-101058/checkpoint-50-merged"
    #     # pred_jsonl_path = f"shared_output/omni_grpo_a_v_omni3_reason_correct_rm/v3-20260223-140429/checkpoint-{epoch}-merged/test_pred_v_a_sample_1448_epoch_{epoch}.jsonl"
    #     # pred_jsonl_path = f"shared_output/omni_grpo_a_v_omni3_reason_correct_rm/v2-20260222-115829/checkpoint-{epoch}-merged/test_pred_v_a_sample_1448_epoch_{epoch}.jsonl"
    #     evaluate(pred_jsonl_path, gt_dict)
    