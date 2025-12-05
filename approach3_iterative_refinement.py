"""
ABORDAGEM 3: Refinamento Iterativo com Filtro Adaptativo
- Itera múltiplas vezes refinando as boxes
- A cada iteração, remove outliers e recalcula médias
- Score baseado em estabilidade através das iterações
- VANTAGEM: Mais robusto a anotações ruins/outliers
"""

import numpy as np
import os
from copy import deepcopy

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
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
    
    return boxes, labels

def calculate_iou(box1, box2):
    """Calcula IoU"""
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

def find_similar_boxes(target_box, target_label, all_boxes, all_labels, iou_threshold):
    """Encontra todas as boxes similares"""
    similar_indices = []
    
    for i, (box, label) in enumerate(zip(all_boxes, all_labels)):
        if label == target_label:
            iou = calculate_iou(target_box, box)
            if iou > iou_threshold:
                similar_indices.append(i)
    
    return similar_indices

def remove_outliers_mad(boxes, threshold=2.0):
    """
    Remove outliers usando MAD (Median Absolute Deviation)
    Mais robusto que desvio padrão
    """
    if len(boxes) <= 2:
        return list(range(len(boxes)))
    
    boxes_array = np.array(boxes)
    median = np.median(boxes_array, axis=0)
    
    # Calcular MAD para cada coordenada
    mad = np.median(np.abs(boxes_array - median), axis=0)
    
    # Evitar divisão por zero
    mad = np.where(mad == 0, 1e-6, mad)
    
    # Calcular score de desvio para cada box
    deviations = np.abs(boxes_array - median) / mad
    max_deviation = np.max(deviations, axis=1)
    
    # Manter boxes com desvio menor que threshold
    inliers = np.where(max_deviation < threshold)[0].tolist()
    
    return inliers if inliers else [0]  # Manter pelo menos uma

def iterative_refinement(boxes, labels, iou_threshold=0.5, max_iterations=3):
    """
    Refina boxes iterativamente removendo outliers
    
    Retorna:
        refined_boxes: boxes refinadas
        stability_scores: score baseado em estabilidade
        refined_labels: labels correspondentes
    """
    n = len(boxes)
    processed = [False] * n
    
    refined_boxes = []
    stability_scores = []
    refined_labels = []
    
    for i in range(n):
        if processed[i]:
            continue
        
        # Encontrar grupo de boxes similares
        similar_indices = find_similar_boxes(
            boxes[i], labels[i], boxes, labels, iou_threshold
        )
        
        if not similar_indices:
            continue
        
        # Marcar como processadas
        for idx in similar_indices:
            processed[idx] = True
        
        # Refinamento iterativo
        current_boxes = [boxes[idx] for idx in similar_indices]
        iteration_history = [current_boxes]
        
        for iteration in range(max_iterations):
            # Remover outliers
            inlier_indices = remove_outliers_mad(current_boxes, threshold=2.5)
            current_boxes = [current_boxes[idx] for idx in inlier_indices]
            
            if len(current_boxes) <= 1:
                break
            
            iteration_history.append(current_boxes)
        
        # Calcular box final (média das inliers)
        final_box = np.mean(current_boxes, axis=0).tolist()
        
        # Calcular score de estabilidade
        # Baseado em: quantidade de boxes, variância, e convergência
        n_boxes = len(similar_indices)
        n_final = len(current_boxes)
        retention_rate = n_final / n_boxes  # Proporção mantida após filtro
        
        # Variância das boxes finais
        variance = np.var(current_boxes, axis=0).mean() if len(current_boxes) > 1 else 0
        
        # Score de convergência (mudança entre iterações)
        if len(iteration_history) > 1:
            initial_var = np.var(iteration_history[0], axis=0).mean()
            final_var = np.var(iteration_history[-1], axis=0).mean()
            convergence = 1.0 - (final_var / (initial_var + 1e-6))
            convergence = max(0, min(1, convergence))
        else:
            convergence = 0.5
        
        # Score final combinado
        stability = (
            0.4 * min(1.0, n_boxes / 5) +  # Quantidade de concordância
            0.3 * retention_rate +          # Proporção de inliers
            0.2 * (1 - min(1.0, variance * 10)) +  # Baixa variância
            0.1 * convergence               # Convergência
        )
        
        refined_boxes.append(final_box)
        stability_scores.append(stability)
        refined_labels.append(labels[i])
    
    return refined_boxes, stability_scores, refined_labels

def process_with_iterative_refinement(annotation_files, iou_threshold=0.5, 
                                     min_stability=0.3, max_iterations=3):
    """
    Processa anotações com refinamento iterativo
    
    Args:
        annotation_files: lista de arquivos
        iou_threshold: threshold de IoU
        min_stability: score mínimo de estabilidade
        max_iterations: número máximo de iterações de refinamento
    """
    # Ler todas as anotações
    all_boxes, all_labels = [], []
    
    for ann_file in annotation_files:
        boxes, labels = read_yolo_annotations(ann_file)
        all_boxes.extend(boxes)
        all_labels.extend(labels)
    
    if not all_boxes:
        return [], [], []
    
    # Aplicar refinamento iterativo
    refined_boxes, stability_scores, refined_labels = iterative_refinement(
        all_boxes, all_labels, iou_threshold, max_iterations
    )
    
    # Filtrar por estabilidade mínima
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    
    for box, score, label in zip(refined_boxes, stability_scores, refined_labels):
        if score >= min_stability:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)
    
    return filtered_boxes, filtered_scores, filtered_labels

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
            line += f" # stability: {scores[i]:.3f}"
        lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

def analyze_refinement(annotation_files, iou_threshold=0.5):
    """Analisa processo de refinamento"""
    all_boxes = []
    for f in annotation_files:
        boxes, _ = read_yolo_annotations(f)
        all_boxes.extend(boxes)
    
    boxes, scores, labels = process_with_iterative_refinement(
        annotation_files, iou_threshold
    )
    
    print(f"\n=== ANÁLISE DE REFINAMENTO ===")
    print(f"Anotações originais: {len(all_boxes)}")
    print(f"Após refinamento: {len(boxes)}")
    print(f"Redução: {(1 - len(boxes)/len(all_boxes))*100:.1f}%")
    print(f"\nEstabilidade média: {np.mean(scores):.3f}")
    print(f"Estabilidade mínima: {np.min(scores):.3f}")
    print(f"Estabilidade máxima: {np.max(scores):.3f}")
    
    # Distribuição
    high = sum(1 for s in scores if s >= 0.7)
    medium = sum(1 for s in scores if 0.4 <= s < 0.7)
    low = sum(1 for s in scores if s < 0.4)
    
    print(f"\nDistribuição de estabilidade:")
    print(f"  Alta (≥0.7): {high}")
    print(f"  Média (0.4-0.7): {medium}")
    print(f"  Baixa (<0.4): {low}")
    
    return boxes, scores, labels

if __name__ == "__main__":
    files = ['file1.txt', 'file2.txt', 'file3.txt']
    boxes, scores, labels = analyze_refinement(files)
