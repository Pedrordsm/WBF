import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import os
from pathlib import Path

def read_yolo_annotations(txt_path, img_width=1.0, img_height=1.0):
    """
    Lê anotações no formato YOLO e retorna em formato normalizado
    YOLO format: class x_center y_center width height (todos normalizados 0-1)
    """
    boxes = []
    scores = []
    labels = []
    
    if not os.path.exists(txt_path):
        return boxes, scores, labels
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Converter de YOLO (x_center, y_center, w, h) para (x1, y1, x2, y2)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                scores.append(1.0)  # Score inicial uniforme
                labels.append(class_id)
    
    return boxes, scores, labels

def calculate_iou(box1, box2):
    """
    Calcula IoU entre duas boxes no formato [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def assign_confidence_scores(boxes, labels, iou_threshold=0.5):
    """
    Atribui scores de confiança baseado na redundância (IoU entre boxes da mesma classe)
    Quanto mais boxes similares, maior a confiança
    """
    n = len(boxes)
    scores = np.ones(n)
    
    for i in range(n):
        overlap_count = 0
        total_iou = 0.0
        
        for j in range(n):
            if i != j and labels[i] == labels[j]:
                iou = calculate_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    overlap_count += 1
                    total_iou += iou
        
        # Score baseado na quantidade de overlaps e IoU médio
        if overlap_count > 0:
            avg_iou = total_iou / overlap_count
            # Normalizar: mais overlaps = maior confiança
            scores[i] = min(1.0, 0.5 + (overlap_count * 0.1) + (avg_iou * 0.3))
        else:
            # Box única, menor confiança
            scores[i] = 0.3
    
    return scores.tolist()

def process_annotations_with_wbf(annotation_files, iou_thr=0.5, skip_box_thr=0.0001, conf_type='avg'):
    """
    Processa múltiplos arquivos de anotação usando WBF
    
    Args:
        annotation_files: lista de caminhos para arquivos .txt
        iou_thr: threshold de IoU para o WBF
        skip_box_thr: threshold mínimo de confiança
        conf_type: 'avg' ou 'max' ou 'box_and_model_avg'
    """
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Ler todas as anotações
    for ann_file in annotation_files:
        boxes, scores, labels = read_yolo_annotations(ann_file)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    if not all_boxes:
        return [], [], []
    
    # Calcular scores de confiança baseado em redundância
    # Concatenar todas as boxes para análise
    flat_boxes = []
    flat_labels = []
    for boxes, labels in zip(all_boxes, all_labels):
        flat_boxes.extend(boxes)
        flat_labels.extend(labels)
    
    # Atribuir scores baseado em IoU
    confidence_scores = assign_confidence_scores(flat_boxes, flat_labels, iou_threshold=0.5)
    
    # Redistribuir scores para estrutura original
    idx = 0
    for i in range(len(all_boxes)):
        n_boxes = len(all_boxes[i])
        all_scores[i] = confidence_scores[idx:idx + n_boxes]
        idx += n_boxes
    
    # Aplicar WBF
    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        weights=None,  # Pesos iguais para todos os anotadores
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type
    )
    
    return boxes_fused, scores_fused, labels_fused

def convert_to_yolo_format(boxes, labels):
    """
    Converte boxes de [x1, y1, x2, y2] para formato YOLO [class x_center y_center width height]
    """
    yolo_annotations = []
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        yolo_annotations.append(f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def save_yolo_annotations(output_path, boxes, labels):
    """
    Salva anotações no formato YOLO
    """
    yolo_lines = convert_to_yolo_format(boxes, labels)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo: processar múltiplas anotações da mesma imagem
    annotation_files = [
        'labels/labels/test/002a34c58c5b758217ed1f584ccbcfe9.txt',
        # Adicione outros arquivos de anotação da mesma imagem aqui
    ]
    
    # Processar com WBF
    boxes_fused, scores_fused, labels_fused = process_annotations_with_wbf(
        annotation_files,
        iou_thr=0.5,  # IoU threshold para fusão
        skip_box_thr=0.3,  # Ignorar boxes com confiança < 0.3
        conf_type='avg'  # Tipo de agregação de confiança
    )
    
    print(f"Boxes originais: {sum(len(read_yolo_annotations(f)[0]) for f in annotation_files)}")
    print(f"Boxes após WBF: {len(boxes_fused)}")
    print(f"Scores: {scores_fused}")
    
    # Salvar resultado
    # save_yolo_annotations('output.txt', boxes_fused, labels_fused)
