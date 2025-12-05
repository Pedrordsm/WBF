"""
EXEMPLO PRÁTICO DE USO
Demonstra como usar cada abordagem com seus dados reais
"""

import os
import numpy as np
from pathlib import Path

# Importar as abordagens
from approach1_wbf_confidence import process_with_wbf, save_yolo_format as save_wbf
from approach2_clustering_consensus import process_with_clustering, analyze_consensus
from approach3_iterative_refinement import process_with_iterative_refinement, analyze_refinement

def example_single_image():
    """
    Exemplo 1: Processar uma única imagem com múltiplas anotações
    """
    print("\n" + "="*60)
    print("EXEMPLO 1: Processando uma única imagem")
    print("="*60)
    
    # Suponha que você tem 3 arquivos de anotação da mesma imagem
    # feitos por 3 pessoas diferentes
    annotation_files = [
        'labels/labels/test/002a34c58c5b758217ed1f584ccbcfe9.txt',
        # Se você tiver mais arquivos da mesma imagem, adicione aqui
        # 'path/to/same_image_annotator2.txt',
        # 'path/to/same_image_annotator3.txt',
    ]
    
    # Verificar se arquivos existem
    existing_files = [f for f in annotation_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️  Nenhum arquivo encontrado. Ajuste os caminhos.")
        return
    
    print(f"\nArquivos encontrados: {len(existing_files)}")
    
    # Testar Abordagem 1: WBF
    print("\n--- Abordagem 1: WBF ---")
    try:
        boxes1, scores1, labels1 = process_with_wbf(existing_files)
        print(f"✓ Boxes resultantes: {len(boxes1)}")
        print(f"  Score médio: {np.mean(scores1):.3f}")
        print(f"  Scores: {[f'{s:.3f}' for s in scores1[:5]]}")  # Primeiros 5
    except Exception as e:
        print(f"✗ Erro: {e}")
    
    # Testar Abordagem 2: Clustering
    print("\n--- Abordagem 2: Clustering ---")
    try:
        boxes2, scores2, labels2 = process_with_clustering(existing_files)
        print(f"✓ Boxes resultantes: {len(boxes2)}")
        print(f"  Consenso médio: {np.mean(scores2):.1%}")
        print(f"  Consensos: {[f'{s:.1%}' for s in scores2[:5]]}")
    except Exception as e:
        print(f"✗ Erro: {e}")
    
    # Testar Abordagem 3: Iterativo
    print("\n--- Abordagem 3: Iterativo ---")
    try:
        boxes3, scores3, labels3 = process_with_iterative_refinement(existing_files)
        print(f"✓ Boxes resultantes: {len(boxes3)}")
        print(f"  Estabilidade média: {np.mean(scores3):.3f}")
        print(f"  Estabilidades: {[f'{s:.3f}' for s in scores3[:5]]}")
    except Exception as e:
        print(f"✗ Erro: {e}")

def example_batch_processing():
    """
    Exemplo 2: Processar múltiplas imagens em batch
    """
    print("\n" + "="*60)
    print("EXEMPLO 2: Processamento em Batch")
    print("="*60)
    
    # Diretório com suas anotações
    label_dir = 'labels/labels/test'
    
    if not os.path.exists(label_dir):
        print(f"⚠️  Diretório não encontrado: {label_dir}")
        return
    
    # Listar todos os arquivos
    all_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"\nEncontrados {len(all_files)} arquivos de anotação")
    
    # Processar primeiros 5 arquivos como exemplo
    sample_files = all_files[:5]
    
    print("\nProcessando amostra de 5 arquivos...")
    
    results = {
        'wbf': [],
        'clustering': [],
        'iterative': []
    }
    
    for filename in sample_files:
        filepath = os.path.join(label_dir, filename)
        
        # Para este exemplo, cada arquivo é tratado individualmente
        # Na prática, você agruparia arquivos da mesma imagem
        
        try:
            # WBF
            boxes, scores, labels = process_with_wbf([filepath])
            results['wbf'].append(len(boxes))
            
            # Clustering
            boxes, scores, labels = process_with_clustering([filepath])
            results['clustering'].append(len(boxes))
            
            # Iterativo
            boxes, scores, labels = process_with_iterative_refinement([filepath])
            results['iterative'].append(len(boxes))
            
        except Exception as e:
            print(f"Erro em {filename}: {e}")
    
    # Estatísticas
    print("\n--- Estatísticas ---")
    for approach, counts in results.items():
        if counts:
            print(f"{approach.upper()}:")
            print(f"  Média de boxes por imagem: {np.mean(counts):.1f}")
            print(f"  Total de boxes: {sum(counts)}")

def example_with_visualization():
    """
    Exemplo 3: Processar e preparar para visualização
    """
    print("\n" + "="*60)
    print("EXEMPLO 3: Preparando para Visualização")
    print("="*60)
    
    annotation_files = [
        'labels/labels/test/002a34c58c5b758217ed1f584ccbcfe9.txt',
    ]
    
    existing_files = [f for f in annotation_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️  Nenhum arquivo encontrado.")
        return
    
    # Processar com Abordagem 2 (mais interpretável)
    boxes, scores, labels = process_with_clustering(existing_files)
    
    print(f"\nProcessado: {len(boxes)} boxes")
    
    # Criar arquivo de saída com informações extras
    output_lines = []
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Formato YOLO padrão
        line = f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
        # Adicionar comentário com informações extras
        confidence_level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
        line += f"  # Consensus: {score:.1%} ({confidence_level})"
        
        output_lines.append(line)
    
    # Salvar
    output_path = 'example_output_with_info.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Salvo em: {output_path}")
    print("\nPrimeiras 3 linhas:")
    for line in output_lines[:3]:
        print(f"  {line}")

def example_parameter_tuning():
    """
    Exemplo 4: Testando diferentes parâmetros
    """
    print("\n" + "="*60)
    print("EXEMPLO 4: Ajustando Parâmetros")
    print("="*60)
    
    annotation_files = [
        'labels/labels/test/002a34c58c5b758217ed1f584ccbcfe9.txt',
    ]
    
    existing_files = [f for f in annotation_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️  Nenhum arquivo encontrado.")
        return
    
    # Testar diferentes thresholds de IoU
    iou_thresholds = [0.3, 0.5, 0.7]
    
    print("\nTestando diferentes IoU thresholds:")
    print("-" * 40)
    
    for iou_thr in iou_thresholds:
        boxes, scores, labels = process_with_clustering(
            existing_files,
            iou_threshold=iou_thr
        )
        
        print(f"\nIoU = {iou_thr}:")
        print(f"  Boxes: {len(boxes)}")
        if scores:
            print(f"  Score médio: {np.mean(scores):.3f}")
            print(f"  Score min/max: {np.min(scores):.3f} / {np.max(scores):.3f}")

def example_quality_analysis():
    """
    Exemplo 5: Análise de qualidade das anotações
    """
    print("\n" + "="*60)
    print("EXEMPLO 5: Análise de Qualidade")
    print("="*60)
    
    annotation_files = [
        'labels/labels/test/002a34c58c5b758217ed1f584ccbcfe9.txt',
    ]
    
    existing_files = [f for f in annotation_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️  Nenhum arquivo encontrado.")
        return
    
    # Usar Abordagem 2 para análise de consenso
    boxes, scores, labels = process_with_clustering(existing_files)
    
    if not scores:
        print("Nenhuma box processada.")
        return
    
    # Análise detalhada
    print("\n--- Análise de Qualidade ---")
    
    high_quality = [s for s in scores if s >= 0.7]
    medium_quality = [s for s in scores if 0.4 <= s < 0.7]
    low_quality = [s for s in scores if s < 0.4]
    
    print(f"\nDistribuição de Qualidade:")
    print(f"  Alta (≥70% consenso): {len(high_quality)} boxes ({len(high_quality)/len(scores)*100:.1f}%)")
    print(f"  Média (40-70%): {len(medium_quality)} boxes ({len(medium_quality)/len(scores)*100:.1f}%)")
    print(f"  Baixa (<40%): {len(low_quality)} boxes ({len(low_quality)/len(scores)*100:.1f}%)")
    
    if low_quality:
        print(f"\n⚠️  {len(low_quality)} boxes com baixo consenso detectadas!")
        print("  Considere revisar essas anotações manualmente.")
    
    # Estatísticas por classe
    from collections import Counter
    class_counts = Counter(labels)
    
    print(f"\nDistribuição por Classe:")
    for class_id, count in class_counts.most_common():
        class_boxes = [s for l, s in zip(labels, scores) if l == class_id]
        avg_score = np.mean(class_boxes)
        print(f"  Classe {class_id}: {count} boxes (consenso médio: {avg_score:.1%})")

def main():
    """Executa todos os exemplos"""
    print("\n" + "="*70)
    print(" "*20 + "EXEMPLOS DE USO")
    print("="*70)
    
    # Executar exemplos
    example_single_image()
    example_batch_processing()
    example_with_visualization()
    example_parameter_tuning()
    example_quality_analysis()
    
    print("\n" + "="*70)
    print("✓ Todos os exemplos executados!")
    print("="*70)
    print("\nPróximos passos:")
    print("1. Ajuste os caminhos dos arquivos para seus dados")
    print("2. Escolha a abordagem que melhor se adequa")
    print("3. Use batch_processor.py para processar tudo")
    print("4. Valide os resultados visualmente")

if __name__ == "__main__":
    main()
