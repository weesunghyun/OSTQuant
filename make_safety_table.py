import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_dir_name(dir_name):
    parts = dir_name.split('_')
    model = parts[0]
    quant = parts[1]
    if "contrastive" in parts[2]:
        loss_type = '_'.join(parts[2:-1])
        weight = parts[-1]
    else:
        loss_type = '_'.join(parts[2:])
        weight = None
    return model, quant, loss_type, weight

def extract_metrics(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None

    metrics = {
        'PPL': None,
        'toxigen_acc': None,
        'truthfulqa_gen_bleu_acc': None,
        'truthfulqa_gen_rougeL_acc': None,
        'truthfulqa_mc1_acc': None,
        'truthfulqa_mc2_acc': None
    }

    # PPL line
    for line in lines:
        if 'After Dynamic Act&Weight Quant PPL:' in line:
            match = re.search(r'PPL:([0-9.]+)', line)
            if match:
                metrics['PPL'] = float(match.group(1))
                break

    # Table parsing
    current_task = None
    for line in lines:
        if not line.strip().startswith('|') or 'Tasks' in line or '------' in line:
            continue

        cols = [col.strip() for col in line.strip().split('|') if col.strip()]
        if len(cols) < 6:
            continue

        # If a task name is present, update current task
        if not line.startswith('|              '):  # i.e., task name present
            current_task = cols[0]
        # else:
        #     breakpoint()

        metric_name = cols[-5]
        try:
            value = float(cols[-3])
        except ValueError:
            continue

        if current_task == 'toxigen' and metric_name == 'acc':
            metrics['toxigen_acc'] = value
        elif current_task == 'truthfulqa_gen' and metric_name == 'bleu_acc':
            metrics['truthfulqa_gen_bleu_acc'] = value
        elif current_task == 'truthfulqa_gen' and metric_name == 'rougeL_acc':
            metrics['truthfulqa_gen_rougeL_acc'] = value
        elif current_task == 'truthfulqa_mc1' and metric_name == 'acc':
            metrics['truthfulqa_mc1_acc'] = value
        elif current_task == 'truthfulqa_mc2' and metric_name == 'acc':
            metrics['truthfulqa_mc2_acc'] = value

    return metrics

def main(output_dir):
    data = []
    output_path = Path(output_dir)

    for subdir in tqdm(sorted(output_path.iterdir()), desc="Processing directories"):
        if subdir.is_dir():
            log_file = subdir / 'log.txt'
            if log_file.exists():
                model, quant, loss_type, weight = parse_dir_name(subdir.name)
                metrics = extract_metrics(log_file)
                if metrics:
                    data.append({
                        'Model': model,
                        'Quant': quant,
                        'Loss Type': loss_type,
                        'Weight': weight,
                        **metrics
                    })

    df = pd.DataFrame(data)
    df.sort_values(by=['Model', 'Quant', 'Loss Type', 'Weight'], inplace=True)
    return df

if __name__ == "__main__":
    output_directory = './output'
    df_metrics = main(output_directory)
    print(df_metrics)
    df_metrics.to_csv('model_metrics_summary_0520_safety.csv', index=False)
