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
    """
    Extracts the latest PPL and average accuracy from the log file.
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None, None

    # Extract the last occurrence of PPL
    ppl = None
    for line in reversed(lines):
        match = re.search(r'After Dynamic Act&Weight Quant PPL:([0-9.]+)', line)
        if match:
            ppl = float(match.group(1))
            break

    # Extract the last occurrence of average accuracy
    acc = None
    for i in range(len(lines) - 1, -1, -1):
        if 'AVERAGE' in lines[i] and 'acc' in lines[i]:
            match = re.search(r'\|\s*AVERAGE\s*\|.*\|acc\s*\|\s*\|\s*([0-9.]+)\s*\|', lines[i])
            if match:
                acc = float(match.group(1))
                break

    return ppl, acc

def main(output_dir):
    """
    Main function to process all subdirectories and compile the metrics into a DataFrame.
    """
    data = []
    output_path = Path(output_dir)

    for subdir in tqdm(sorted(output_path.iterdir()), desc="Processing directories"):
        if subdir.is_dir():
            log_file = subdir / 'log.txt'
            if log_file.exists():
                model, quant, loss_type, weight = parse_dir_name(subdir.name)
                ppl, acc = extract_metrics(log_file)
                data.append({
                    'Model': model,
                    'Quant': quant,
                    'Loss Type': loss_type,
                    'Weight': weight,
                    'PPL': ppl,
                    'Average Accuracy': acc
                })

    df = pd.DataFrame(data)
    df.sort_values(by=['Model', 'Quant', 'Loss Type', 'Weight'], inplace=True)
    return df

if __name__ == "__main__":
    output_directory = '/app/data2/wii/safetyquant/OSTQuant/output'  # Replace with your actual path
    df_metrics = main(output_directory)
    print(df_metrics)
    # Optionally, save to CSV
    df_metrics.to_csv('model_metrics_summary_0520.csv', index=False)
