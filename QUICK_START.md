# 🚀 Guia de Início Rápido

## Instalação

```bash
# Dependências básicas (obrigatório)
pip install numpy

# Para Abordagem 1 (WBF) - opcional
pip install ensemble-boxes

# Para visualização - opcional
pip install matplotlib
```

## Uso Rápido (3 passos)

### 1️⃣ Teste Simples (uma imagem)

```python
from approach2_clustering_consensus import process_with_clustering

# Seus arquivos de anotação da mesma imagem
files = [
    'labels/labels/test/002a34c58c5b758217ed1f584ccbcfe9.txt',
    # adicione mais se tiver múltiplos anotadores
]

# Processar
boxes, scores, labels = process_with_clustering(files)

print(f"Resultado: {len(boxes)} boxes")
print(f"Consenso médio: {sum(scores)/len(scores):.1%}")
```

### 2️⃣ Processar Tudo em Batch

```python
python batch_processor.py
```

Isso vai:
- ✅ Processar todas as imagens
- ✅ Testar as 3 abordagens
- ✅ Gerar estatísticas comparativas
- ✅ Salvar resultados em `processed_annotations/`

### 3️⃣ Ver Exemplos Práticos

```python
python example_usage.py
```

## Estrutura dos Arquivos

```
📁 Seu Projeto
├── 📄 approach1_wbf_confidence.py      # Abordagem 1: WBF
├── 📄 approach2_clustering_consensus.py # Abordagem 2: Clustering ⭐ RECOMENDADO
├── 📄 approach3_iterative_refinement.py # Abordagem 3: Iterativo
├── 📄 batch_processor.py               # Processar tudo
├── 📄 example_usage.py                 # Exemplos práticos
├── 📄 visualization_utils.py           # Visualização
├── 📄 README_APPROACHES.md             # Documentação completa
└── 📄 QUICK_START.md                   # Este arquivo
```

## Qual Abordagem Usar?

### 🎯 Recomendação Rápida

**Comece com Abordagem 2 (Clustering)** - é a mais simples e interpretável!

```python
from approach2_clustering_consensus import process_with_clustering

boxes, scores, labels = process_with_clustering(
    annotation_files,
    iou_threshold=0.5,    # Ajuste se necessário
    min_consensus=0.3     # Mínimo 30% de consenso
)
```

### 📊 Quando Usar Cada Uma

| Abordagem | Use Quando... | Score Significa |
|-----------|---------------|-----------------|
| **1. WBF** | Quer máxima precisão | Confiança combinada |
| **2. Clustering** ⭐ | Quer simplicidade | % de consenso |
| **3. Iterativo** | Tem outliers/ruído | Estabilidade |

## Ajuste de Parâmetros

### IoU Threshold (quão próximas as boxes devem estar)

```python
iou_threshold=0.3  # Mais permissivo (agrupa boxes mais distantes)
iou_threshold=0.5  # Padrão balanceado ⭐
iou_threshold=0.7  # Mais restritivo (só boxes muito próximas)
```

### Score Mínimo (filtro de qualidade)

```python
min_consensus=0.2  # Mantém mais boxes (recall alto)
min_consensus=0.4  # Balanceado ⭐
min_consensus=0.6  # Só alto consenso (precision alta)
```

## Exemplo Completo

```python
# 1. Importar
from approach2_clustering_consensus import process_with_clustering, analyze_consensus
from visualization_utils import print_statistics

# 2. Seus arquivos
annotation_files = [
    'path/to/annotator1.txt',
    'path/to/annotator2.txt',
    'path/to/annotator3.txt',
]

# 3. Processar
boxes, scores, labels = process_with_clustering(
    annotation_files,
    iou_threshold=0.5,
    min_consensus=0.3
)

# 4. Ver estatísticas
print_statistics(boxes, scores, labels, "Meu Resultado")

# 5. Salvar
from approach2_clustering_consensus import save_yolo_format
save_yolo_format('output.txt', boxes, labels, scores)
```

## Interpretando Resultados

### Scores da Abordagem 2 (Clustering)

- **0.8-1.0**: 80-100% dos anotadores concordam ✅ Excelente!
- **0.5-0.8**: 50-80% concordam ✅ Bom
- **0.3-0.5**: 30-50% concordam ⚠️ Revisar
- **<0.3**: Menos de 30% concordam ❌ Suspeito!

### Exemplo de Saída

```
=== ANÁLISE DE CONSENSO ===
Total de anotações originais: 150
Total após consenso: 45
Score médio: 72%

Distribuição:
  Alto consenso (≥60%): 30 boxes
  Médio consenso (30-60%): 12 boxes
  Baixo consenso (<30%): 3 boxes
```

## Troubleshooting

### ❌ "No module named 'ensemble_boxes'"

```bash
pip install ensemble-boxes
# ou use Abordagem 2 ou 3 (não precisam dessa lib)
```

### ❌ "Nenhuma box processada"

- Verifique se os arquivos existem
- Verifique formato YOLO: `class x_center y_center width height`
- Tente diminuir `min_consensus` ou `iou_threshold`

### ❌ "Muitas boxes ainda"

- Aumente `iou_threshold` (ex: 0.7)
- Aumente `min_consensus` (ex: 0.5)
- Use Abordagem 3 (mais agressiva)

### ❌ "Poucas boxes"

- Diminua `iou_threshold` (ex: 0.3)
- Diminua `min_consensus` (ex: 0.2)
- Verifique se não está filtrando demais

## Próximos Passos

1. ✅ Rode `example_usage.py` para ver exemplos
2. ✅ Ajuste parâmetros para seus dados
3. ✅ Rode `batch_processor.py` para processar tudo
4. ✅ Compare as 3 abordagens
5. ✅ Escolha a melhor para seu caso
6. ✅ Valide resultados visualmente

## Dicas Importantes

💡 **Sempre valide visualmente** alguns resultados antes de processar tudo

💡 **Comece com subset pequeno** (5-10 imagens) para testar parâmetros

💡 **Compare as 3 abordagens** - cada uma tem vantagens

💡 **Documente seus parâmetros** - você vai querer reproduzir depois

## Suporte

Leia a documentação completa em `README_APPROACHES.md` para:
- Explicação detalhada de cada abordagem
- Comparação técnica
- Casos de uso específicos
- FAQ

## Checklist Rápido

- [ ] Instalei dependências (`pip install numpy`)
- [ ] Testei com uma imagem (`example_usage.py`)
- [ ] Ajustei parâmetros para meus dados
- [ ] Processei em batch (`batch_processor.py`)
- [ ] Comparei as 3 abordagens
- [ ] Validei resultados visualmente
- [ ] Escolhi a melhor abordagem
- [ ] Documentei meus parâmetros

---

**Pronto para começar? Execute:**

```bash
python example_usage.py
```

🎉 **Boa sorte com seu projeto!**
