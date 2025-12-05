"""
UTILITÁRIOS DE VISUALIZAÇÃO
Funções para visualizar e comparar resultados das diferentes abordagens
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter
import os

def plot_boxes_comparison(image_path, original_boxes, processed_boxes, 
                          original_labels, processed_labels, 
                          processed_scores=None, title="Comparação"):
    """
    Plota comparação entre boxes originais e processadas
    
    Args:
        image_path: caminho para imagem (ou None para plot sem imagem)
        original_boxes: lista de boxes originais [[x1,y1,x2,y2], ...]
        processed_boxes: lista de boxes processadas
        original_labels: labels originais
        processed_labels: labels processadas
        processed_scores: scores das boxes processadas (opcional)
        title: título do plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Cores por classe
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Boxes Originais
    ax1.set_title(f'Original ({len(original_boxes)} boxes)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)  # Inverter Y
    ax1.set_aspect('equal')
    
    for box, label in zip(original_boxes, original_labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=colors[label % 10],
            facecolor='none', alpha=0.7
        )
        ax1.add_patch(rect)
        
        # Label
        ax1.text(x1, y1-0.01, f'C{label}', 
                color=colors[label % 10], fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot 2: Boxes Processadas
    ax2.set_title(f'Processado ({len(processed_boxes)} boxes)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0)
    ax2.set_aspect('equal')
    
    for i, (box, label) in enumerate(zip(processed_boxes, processed_labels)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Cor baseada no score se disponível
        if processed_scores:
            score = processed_scores[i]
            alpha = 0.5 + (score * 0.5)  # Alpha baseado no score
            linewidth = 1 + (score * 2)   # Espessura baseada no score
        else:
            alpha = 0.7
            linewidth = 2
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth, edgecolor=colors[label % 10],
            facecolor='none', alpha=alpha
        )
        ax2.add_patch(rect)
        
        # Label com score
        if processed_scores:
            label_text = f'C{label} ({processed_scores[i]:.2f})'
        else:
            label_text = f'C{label}'
        
        ax2.text(x1, y1-0.01, label_text,
                color=colors[label % 10], fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_score_distribution(scores, approach_name, bins=20):
    """
    Plota distribuição de scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(scores), color='red', linestyle='--', 
               linewidth=2, label=f'Média: {np.mean(scores):.3f}')
    ax.axvline(np.median(scores), color='green', linestyle='--',
               linewidth=2, label=f'Mediana: {np.median(scores):.3f}')
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.set_title(f'Distribuição de Scores - {approach_name}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_class_distribution(labels, scores=None, title="Distribuição por Classe"):
    """
    Plota distribuição de boxes por classe
    """
    class_counts = Counter(labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    bars = ax.bar(classes, counts, edgecolor='black', alpha=0.7)
    
    # Colorir barras baseado no score médio se disponível
    if scores:
        for i, class_id in enumerate(classes):
            class_scores = [s for l, s in zip(labels, scores) if l == class_id]
            avg_score = np.mean(class_scores)
            
            # Cor baseada no score
            color = plt.cm.RdYlGn(avg_score)
            bars[i].set_facecolor(color)
            
            # Adicionar score médio no topo da barra
            ax.text(class_id, counts[i], f'{avg_score:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Classe', fontsize=12)
    ax.set_ylabel('Quantidade de Boxes', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig

def compare_approaches_plot(results_dict):
    """
    Compara resultados de múltiplas abordagens
    
    Args:
        results_dict: {
            'approach_name': {
                'boxes': [...],
                'scores': [...],
                'labels': [...]
            }
        }
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    approach_names = list(results_dict.keys())
    
    # 1. Quantidade de boxes
    ax = axes[0, 0]
    n_boxes = [len(results_dict[name]['boxes']) for name in approach_names]
    bars = ax.bar(approach_names, n_boxes, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Número de Boxes', fontsize=11)
    ax.set_title('Quantidade de Boxes por Abordagem', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, val in zip(bars, n_boxes):
        ax.text(bar.get_x() + bar.get_width()/2, val,
               f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Score médio
    ax = axes[0, 1]
    avg_scores = [np.mean(results_dict[name]['scores']) 
                  if results_dict[name]['scores'] else 0 
                  for name in approach_names]
    bars = ax.bar(approach_names, avg_scores, edgecolor='black', alpha=0.7,
                  color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Score Médio', fontsize=11)
    ax.set_title('Score Médio por Abordagem', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Distribuição de scores (boxplot)
    ax = axes[1, 0]
    scores_data = [results_dict[name]['scores'] for name in approach_names]
    bp = ax.boxplot(scores_data, labels=approach_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Distribuição de Scores', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Redução percentual
    ax = axes[1, 1]
    
    # Assumir que a primeira abordagem tem o total original
    # (ou você pode passar o total original separadamente)
    if len(approach_names) > 0:
        baseline = n_boxes[0]
        reductions = [(baseline - n) / baseline * 100 for n in n_boxes]
        
        bars = ax.bar(approach_names, reductions, edgecolor='black', alpha=0.7,
                     color=['#95a5a6', '#3498db', '#e74c3c'])
        ax.set_ylabel('Redução (%)', fontsize=11)
        ax.set_title('Redução de Boxes vs Baseline', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, reductions):
            ax.text(bar.get_x() + bar.get_width()/2, val,
                   f'{val:.1f}%', ha='center', 
                   va='bottom' if val >= 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def save_comparison_report(results_dict, output_path='comparison_report.png'):
    """
    Gera e salva relatório visual completo
    """
    fig = compare_approaches_plot(results_dict)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Relatório salvo em: {output_path}")
    return fig

def print_statistics(boxes, scores, labels, approach_name):
    """
    Imprime estatísticas detalhadas
    """
    print(f"\n{'='*60}")
    print(f"ESTATÍSTICAS - {approach_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\nGeral:")
    print(f"  Total de boxes: {len(boxes)}")
    
    if scores:
        print(f"  Score médio: {np.mean(scores):.3f}")
        print(f"  Score mediano: {np.median(scores):.3f}")
        print(f"  Score mín/máx: {np.min(scores):.3f} / {np.max(scores):.3f}")
        print(f"  Desvio padrão: {np.std(scores):.3f}")
    
    # Por classe
    class_counts = Counter(labels)
    print(f"\nPor Classe:")
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        
        if scores:
            class_scores = [s for l, s in zip(labels, scores) if l == class_id]
            avg_score = np.mean(class_scores)
            print(f"  Classe {class_id}: {count} boxes (score médio: {avg_score:.3f})")
        else:
            print(f"  Classe {class_id}: {count} boxes")
    
    # Distribuição de qualidade
    if scores:
        high = sum(1 for s in scores if s >= 0.7)
        medium = sum(1 for s in scores if 0.4 <= s < 0.7)
        low = sum(1 for s in scores if s < 0.4)
        
        print(f"\nDistribuição de Qualidade:")
        print(f"  Alta (≥0.7): {high} ({high/len(scores)*100:.1f}%)")
        print(f"  Média (0.4-0.7): {medium} ({medium/len(scores)*100:.1f}%)")
        print(f"  Baixa (<0.4): {low} ({low/len(scores)*100:.1f}%)")

if __name__ == "__main__":
    print("Utilitários de visualização carregados!")
    print("\nFunções disponíveis:")
    print("  - plot_boxes_comparison()")
    print("  - plot_score_distribution()")
    print("  - plot_class_distribution()")
    print("  - compare_approaches_plot()")
    print("  - save_comparison_report()")
    print("  - print_statistics()")
