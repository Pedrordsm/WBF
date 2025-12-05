"""
PROCESSADOR EM BATCH
Processa todos os arquivos de anotação e compara as 3 abordagens
"""

import os
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

# Importar as 3 abordagens
from approach1_wbf_confidence import process_with_wbf
from approach2_clustering_consensus import process_with_clustering
from approach3_iterative_refinement import process_with_iterative_refinement

def group_annotations_by_image(label_folders):
    """
    Agrupa arquivos de anotação por imagem
    Assume que arquivos com mesmo nome são da mesma imagem
    """
    image_annotations = defaultdict(list)
    
    for folder in label_folders:
        if not os.path.exists(folder):
            continue
        
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                image_id = filename.replace('.txt', '')
                filepath = os.path.join(folder, filename)
                image_annotations[image_id].append(filepath)
    
    return image_annotations

def process_all_images(image_annotations, output_dir, approach='all'):
    """
    Processa todas as imagens com a(s) abordagem(ns) escolhida(s)
    
    Args:
        image_annotations: dict {image_id: [list of annotation files]}
        output_dir: diretório de saída
        approach: 'wbf', 'clustering', 'iterative', ou 'all'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'wbf': {},
        'clustering': {},
        'iterative': {}
    }
    
    total_images = len(image_annotations)
    
    for idx, (image_id, ann_files) in enumerate(image_annotations.items(), 1):
        print(f"\rProcessando {idx}/{total_images}: {image_id}", end='')
        
        # Pular se só tem 1 anotação (sem redundância)
        if len(ann_files) < 2:
            continue
        
        # Abordagem 1: WBF
        if approach in ['wbf', 'all']:
            try:
                boxes, scores, labels = process_with_wbf(ann_files)
                results['wbf'][image_id] = {
                    'n_boxes': len(boxes),
                    'avg_score': float(np.mean(scores)) if scores else 0,
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
                
                # Salvar
                output_path = os.path.join(output_dir, 'wbf', f'{image_id}.txt')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_results(output_path, boxes, labels, scores)
            except Exception as e:
                print(f"\nErro WBF em {image_id}: {e}")
        
        # Abordagem 2: Clustering
        if approach in ['clustering', 'all']:
            try:
                boxes, scores, labels = process_with_clustering(ann_files)
                results['clustering'][image_id] = {
                    'n_boxes': len(boxes),
                    'avg_score': float(np.mean(scores)) if scores else 0,
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
                
                output_path = os.path.join(output_dir, 'clustering', f'{image_id}.txt')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_results(output_path, boxes, labels, scores)
            except Exception as e:
                print(f"\nErro Clustering em {image_id}: {e}")
        
        # Abordagem 3: Iterative
        if approach in ['iterative', 'all']:
            try:
                boxes, scores, labels = process_with_iterative_refinement(ann_files)
                results['iterative'][image_id] = {
                    'n_boxes': len(boxes),
                    'avg_score': float(np.mean(scores)) if scores else 0,
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
                
                output_path = os.path.join(output_dir, 'iterative', f'{image_id}.txt')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_results(output_path, boxes, labels, scores)
            except Exception as e:
                print(f"\nErro Iterative em {image_id}: {e}")
    
    print("\n\nProcessamento concluído!")
    return results

def save_results(output_path, boxes, labels, scores):
    """Salva resultados no formato YOLO"""
    lines = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        line = f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

def compare_approaches(results):
    """Compara estatísticas das 3 abordagens"""
    print("\n" + "="*60)
    print("COMPARAÇÃO DAS ABORDAGENS")
    print("="*60)
    
    for approach_name, approach_results in results.items():
        if not approach_results:
            continue
        
        total_boxes = sum(r['n_boxes'] for r in approach_results.values())
        avg_boxes_per_image = total_boxes / len(approach_results)
        avg_score = np.mean([r['avg_score'] for r in approach_results.values()])
        
        print(f"\n{approach_name.upper()}:")
        print(f"  Imagens processadas: {len(approach_results)}")
        print(f"  Total de boxes: {total_boxes}")
        print(f"  Média de boxes por imagem: {avg_boxes_per_image:.2f}")
        print(f"  Score médio: {avg_score:.3f}")
    
    # Salvar estatísticas
    stats = {
        approach: {
            'n_images': len(results[approach]),
            'total_boxes': sum(r['n_boxes'] for r in results[approach].values()),
            'avg_score': float(np.mean([r['avg_score'] for r in results[approach].values()]))
        }
        for approach in results if results[approach]
    }
    
    with open('comparison_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nEstatísticas salvas em 'comparison_stats.json'")

if __name__ == "__main__":
    # Configuração
    # Exemplo: se você tem múltiplas pastas com anotações diferentes
    label_folders = [
        'labels/labels/test',
        # Adicione outras pastas se tiver múltiplos anotadores
    ]
    
    output_dir = 'processed_annotations'
    
    # Agrupar anotações por imagem
    print("Agrupando anotações por imagem...")
    image_annotations = group_annotations_by_image(label_folders)
    print(f"Encontradas {len(image_annotations)} imagens")
    
    # Processar com todas as abordagens
    print("\nProcessando com todas as abordagens...")
    results = process_all_images(image_annotations, output_dir, approach='all')
    
    # Comparar resultados
    compare_approaches(results)
    
    print("\n✓ Processamento completo!")
    print(f"Resultados salvos em: {output_dir}/")
