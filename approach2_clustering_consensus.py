"""
ABORDAGEM 2: Clustering + Consenso por Votação
- Agrupa boxes similares (IoU > threshold) da mesma classe
- Calcula box média do cluster
- Score = proporção de anotadores que concordam
- VANTAGEM: Mais interpretável, score = % de consenso
"""

import numpy as np
from collections import defaultdict
import os

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

def cluster_boxes_by_similarity(boxes, labels, iou_threshold=0.5):
    """
    Agrupa boxes similares em clusters
    Retorna: lista de clusters, cada cluster é lista de índices
    """
    n = len(boxes)
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Iniciar novo cluster
        cluster = [i]
        visited[i] = True
        
        # Encontrar todas as boxes similares
        for j in range(i + 1, n):
            if visited[j] or labels[i] != labels[j]:
                continue
            
            # Verificar IoU com qualquer box do cluster
            for idx in cluster:
                if calculate_iou(boxes[idx], boxes[j]) > iou_threshold:
                    cluster.append(j)
                    visited[j] = True
                    break
        
        clusters.append(cluster)
    
    return clusters

def calculate_cluster_consensus(cluster_indices, boxes, labels, n_annotators):
    """
    Calcula box consenso e score para um cluster
    Score = proporção de anotadores que concordam
    """
    cluster_boxes = [boxes[i] for i in cluster_indices]
    cluster_label = labels[cluster_indices[0]]
    
    # Calcular box média (consenso)
    avg_box = np.mean(cluster_boxes, axis=0).tolist()
    
    # Score = proporção de concordância
    # Se 3 de 5 anotadores concordam, score = 0.6
    consensus_score = len(cluster_indices) / n_annotators
    
    # Ajustar score baseado na variância do cluster
    # Menor variância = maior confiança
    if len(cluster_indices) > 1:
        variance = np.var(cluster_boxes, axis=0).mean()
        variance_penalty = min(0.2, variance * 10)  # Penalidade por alta variância
        consensus_score = max(0.1, consensus_score - variance_penalty)
    
    return avg_box, consensus_score, cluster_label

def process_with_clustering(annotation_files, iou_threshold=0.5, min_consensus=0.2):
    """
    Processa anotações usando clustering e consenso
    
    Args:
        annotation_files: lista de arquivos de anotação
        iou_threshold: threshold para considerar boxes similares
        min_consensus: score mínimo para manter uma box (ex: 0.2 = 20% dos anotadores)
    """
    # Ler todas as anotações
    all_boxes, all_labels = [], []
    
    for ann_file in annotation_files:
        boxes, labels = read_yolo_annotations(ann_file)
        all_boxes.extend(boxes)
        all_labels.extend(labels)
    
    if not all_boxes:
        return [], [], []
    
    n_annotators = len(annotation_files)
    
    # Agrupar boxes similares
    clusters = cluster_boxes_by_similarity(all_boxes, all_labels, iou_threshold)
    
    # Calcular consenso para cada cluster
    final_boxes, final_scores, final_labels = [], [], []
    
    for cluster in clusters:
        box, score, label = calculate_cluster_consensus(
            cluster, all_boxes, all_labels, n_annotators
        )
        
        # Filtrar por consenso mínimo
        if score >= min_consensus:
            final_boxes.append(box)
            final_scores.append(score)
            final_labels.append(label)
    
    return final_boxes, final_scores, final_labels

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
            line += f" # consensus: {scores[i]:.2%}"
        lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

def analyze_consensus(annotation_files, iou_threshold=0.5):
    """Analisa estatísticas de consenso"""
    boxes, scores, labels = process_with_clustering(annotation_files, iou_threshold)
    
    print(f"\n=== ANÁLISE DE CONSENSO ===")
    print(f"Total de anotações originais: {sum(len(read_yolo_annotations(f)[0]) for f in annotation_files)}")
    print(f"Total após consenso: {len(boxes)}")
    print(f"Score médio: {np.mean(scores):.2%}")
    print(f"Score mínimo: {np.min(scores):.2%}")
    print(f"Score máximo: {np.max(scores):.2%}")
    
    # Distribuição de scores
    high_consensus = sum(1 for s in scores if s >= 0.6)
    medium_consensus = sum(1 for s in scores if 0.3 <= s < 0.6)
    low_consensus = sum(1 for s in scores if s < 0.3)
    
    print(f"\nDistribuição:")
    print(f"  Alto consenso (≥60%): {high_consensus}")
    print(f"  Médio consenso (30-60%): {medium_consensus}")
    print(f"  Baixo consenso (<30%): {low_consensus}")
    
    return boxes, scores, labels

if __name__ == "__main__":
    # Exemplo de uso
    files = ['file1.txt', 'file2.txt', 'file3.txt']
    boxes, scores, labels = analyze_consensus(files)
