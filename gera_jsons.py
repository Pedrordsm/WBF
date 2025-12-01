import os
import glob
import json
import numpy as np
from tqdm import tqdm

# --- Funções de Geometria (Mesmas de antes) ---
def yolo_to_x1y1x2y2(yolo_box):
    xc, yc, w, h = yolo_box
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def x1y1x2y2_to_yolo(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# --- Lógica Principal ---

def process_annotations(input_folder, output_txt_folder, output_json_path):
    # Dicionário que vai virar o JSON final
    # Estrutura: { "nome_imagem": { "boxes": [], "scores": [], "labels": [] } }
    wbf_data = {}

    if not os.path.exists(output_txt_folder):
        os.makedirs(output_txt_folder)

    files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    print(f"Processando {len(files)} arquivos...")

    for file_path in tqdm(files):
        filename = os.path.basename(file_path)
        image_id = filename.replace('.txt', '') # Remove extensão para usar como ID
        
        # 1. Ler anotações originais
        original_lines = []
        parsed_boxes = [] # [x1, y1, x2, y2]
        parsed_labels = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls = int(parts[0])
                coords = list(map(float, parts[1:5])) # xc, yc, w, h
                
                original_lines.append(line.strip()) # Guarda texto original para salvar depois
                parsed_boxes.append(yolo_to_x1y1x2y2(coords))
                parsed_labels.append(cls)

        # 2. Encontrar clusters e calcular médias (Lógica de Consistência)
        # Vamos gerar APENAS as novas caixas médias aqui
        new_mean_boxes_yolo = [] # Para salvar no TXT
        
        # Listas para o JSON (formato WBF: normalizado x1,y1,x2,y2)
        wbf_boxes = []
        wbf_scores = []
        wbf_labels = []

        if parsed_boxes:
            used = [False] * len(parsed_boxes)
            
            for i in range(len(parsed_boxes)):
                if used[i]: continue
                
                # Cria cluster
                cluster = [parsed_boxes[i]]
                used[i] = True
                curr_label = parsed_labels[i]
                
                # Busca vizinhos
                for j in range(i + 1, len(parsed_boxes)):
                    if not used[j] and parsed_labels[j] == curr_label:
                        if calculate_iou(parsed_boxes[i], parsed_boxes[j]) > 0.5:
                            cluster.append(parsed_boxes[j])
                            used[j] = True
                
                # --- A MÁGICA: Gera a Média e o Score ---
                mean_box = np.mean(np.array(cluster), axis=0).tolist()
                
                # Score = Média dos IoUs entre a caixa média e as originais do cluster
                ious = [calculate_iou(mean_box, b) for b in cluster]
                score = float(np.mean(ious))
                
                # Prepara dados para o JSON (WBF)
                wbf_boxes.append(mean_box) # [x1, y1, x2, y2]
                wbf_scores.append(score)
                wbf_labels.append(int(curr_label))
                
                # Prepara linha para o TXT (YOLO: class xc yc w h)
                # Opcional: Adicionei o score na linha para você ver, mas formato padrão é 5 colunas
                yolo_mean = x1y1x2y2_to_yolo(mean_box)
                line_str = f"{int(curr_label)} {yolo_mean[0]:.6f} {yolo_mean[1]:.6f} {yolo_mean[2]:.6f} {yolo_mean[3]:.6f}"
                new_mean_boxes_yolo.append(line_str)

        # 3. Salvar novo arquivo TXT (Originais + Novas Médias)
        out_txt_path = os.path.join(output_txt_folder, filename)
        with open(out_txt_path, 'w') as f_out:
            # Escreve as originais primeiro
            for line in original_lines:
                f_out.write(line + "\n")
            
            # Escreve as novas médias (adicionadas ao final)
            for line in new_mean_boxes_yolo:
                # Dica: Adicionei um comentário ou identificador visual se quiser? 
                # Por padrão YOLO não aceita comentários, então vai apenas a linha.
                f_out.write(line + "\n")

        # 4. Salvar dados no dicionário para JSON
        if wbf_boxes:
            wbf_data[image_id] = {
                "boxes": wbf_boxes,   # Lista de listas [x1, y1, x2, y2]
                "scores": wbf_scores, # Lista de floats
                "labels": wbf_labels  # Lista de ints
            }

    # 5. Escrever o arquivo JSON final
    print(f"Salvando arquivo JSON para WBF em: {output_json_path}")
    with open(output_json_path, 'w') as f_json:
        json.dump(wbf_data, f_json, indent=4)

# --- CONFIGURAÇÃO ---
PASTA_ENTRADA = "labels/labels/train"
PASTA_SAIDA_TXT = "txts_com_media_adicionada"
ARQUIVO_JSON_WBF = "dados_para_wbf.json"

if __name__ == "__main__":
    process_annotations(PASTA_ENTRADA, PASTA_SAIDA_TXT, ARQUIVO_JSON_WBF)