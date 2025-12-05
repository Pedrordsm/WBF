"""
ABORDAGEM 1: WBF com Confiança Baseada em Redundância
- Calcula IoU entre todas as boxes da mesma classe
- Score = função da quantidade de overlaps e IoU médio
- Usa WBF para fusão final
- VANTAGEM: Aproveita toda informação de redundância
"""

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import os
from pathlib import Path
from collections import defaultdict

def read_yolo_annotations(txt_path):
    """Lê anotações YOLO"""
    boxes, labels = [], []
    
    if not os.path.exists(txt_path):
        return boxes, labels
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center = float(parts[1]), float(parts[2])
                width, height = float(parts[3]), float(parts[4])
                
                # Converter para [x1, y1, x2, y2]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
    
    return boxes, labels

def calculate_iou(box1, box2):
    """Calcula IoU entre duas boxes"""
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

def calculate_confidence_from_redundancy(boxes, labels, iou_threshold=0.5):
    """
    Calcula confiança baseado em redundância
    Mais overlaps = maior confiança (mais anotadores concordam)
    """
    n = len(boxes)
    scores = []
    
    for i in range(n):
        overlaps = []
        
        for j in range(n):
            if i != j and labels[i] == labels[j]:
                iou = calculate_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    overlaps.append(iou)
        
        # Score baseado em quantidade e qualidade dos overlaps
        if overlaps:
            n_overlaps = len(overlaps)
            avg_iou = np.mean(overlaps)
            max_iou = np.max(overlaps)
            
            # Fórmula: combina quantidade e qualidade
            # Normalizado entre 0.5 e 1.0
            score = 0.5 + (min(n_overlaps, 5) / 10) + (avg_iou * 0.3) + (max_iou * 0.1)
            scores.append(min(1.0, score))
        else:
            # Box única, menor confiança (pode ser falso positivo)
            scores.append(0.3)
    
    return scores

def process_with_wbf(annotation_files, iou_thr=0.55, skip_box_thr=0.35):
    """Processa múltiplas anotações com WBF"""
    all_boxes, all_scores, all_labels = [], [], []
    
    # Ler todas as anotações
    for ann_file in annotation_files:
        boxes, labels = read_yolo_annotations(ann_file)
        if boxes:
            all_boxes.append(boxes)
            all_labels.append(labels)
    
    if not all_boxes:
        return [], [], []
    
    # Calcular scores baseado em redundância global
    flat_boxes = [box for boxes in all_boxes for box in boxes]
    flat_labels = [label for labels in all_labels for label in labels]
    confidence_scores = calculate_confidence_from_redundancy(flat_boxes, flat_labels)
    
    # Redistribuir scores
    idx = 0
    for boxes in all_boxes:
        n = len(boxes)
        all_scores.append(confidence_scores[idx:idx + n])
        idx += n
    
    # Aplicar WBF
    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type='avg'
    )
    
    return boxes_fused, scores_fused, labels_fused

def save_yolo_format(output_path, boxes, labels, scores=None):
    """Salva no formato YOLO"""
    lines = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        line = f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        if scores is not None:
            line += f" {scores[i]:.4f}"
        lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    # Exemplo de uso
    files = ['file1.txt', 'file2.txt', 'file3.txt']
    boxes, scores, labels = process_with_wbf(files)
    print(f"Resultado: {len(boxes)} boxes com scores médio {np.mean(scores):.3f}")
