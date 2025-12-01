import os
import json
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

# --- CONFIGURAÇÕES ---
# Arquivo JSON gerado no passo anterior
ARQUIVO_JSON_ENTRADA = "dados_para_wbf.json"

# Pasta onde ficarão os TXTs finais (limpos, sem redundância, formato YOLO padrão)
PASTA_SAIDA_FINAL = "dataset_yolo_final_limpo"

# Parâmetros do WBF
IOU_THRESHOLD = 0.55  # Limite para considerar fusão final
SKIP_BOX_THR = 0.001  # Descarta caixas com confiança extremamente baixa (ruído)

# --- Função Auxiliar ---
def x1y1x2y2_to_yolo(box):
    """
    Converte coordenadas normalizadas (x1, y1, x2, y2) 
    para formato YOLO (xc, yc, w, h).
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

# --- Execução Principal ---

def generate_clean_yolo_files(json_path, output_folder):
    if not os.path.exists(json_path):
        print(f"Erro: O arquivo {json_path} não existe.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Lendo dados de {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Gerando arquivos YOLO limpos em: {output_folder}")

    # Itera sobre cada imagem
    for image_id, content in tqdm(data.items(), desc="Processando WBF e Salvando"):
        
        boxes = content['boxes']   # [x1, y1, x2, y2] normalizado
        scores = content['scores'] # O score calculado anteriormente (usado só aqui)
        labels = content['labels']

        # Se não houver anotações, cria arquivo vazio e pula
        if len(boxes) == 0:
            open(os.path.join(output_folder, f"{image_id}.txt"), 'w').close()
            continue

        # --- APLICAÇÃO DO WBF ---
        # Prepara listas de listas (formato exigido pela lib)
        boxes_list = [boxes]
        scores_list = [scores]
        labels_list = [labels]
        weights = [1] 

        try:
            # O WBF usa o 'scores' para ponderar a fusão, mas retorna scores_final
            boxes_final, scores_final, labels_final = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=IOU_THRESHOLD,
                skip_box_thr=SKIP_BOX_THR,
                conf_type='avg'
            )
        except Exception as e:
            print(f"Erro na imagem {image_id}: {e}")
            continue

        # --- SALVAMENTO LIMPO (SEM SCORE, SEM REDUNDÂNCIA) ---
        output_path = os.path.join(output_folder, f"{image_id}.txt")
        
        with open(output_path, 'w') as f_out:
            # Itera sobre as caixas resultantes do WBF
            for b, s, l in zip(boxes_final, scores_final, labels_final):
                
                # 1. Converte para geometria YOLO (xc, yc, w, h)
                yolo_box = x1y1x2y2_to_yolo(b)
                
                # 2. Formata a string APENAS com as 5 colunas padrão
                # int(l) garante que a classe seja inteiro
                # :.6f garante precisão nas coordenadas
                line = f"{int(l)} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                
                f_out.write(line)

if __name__ == "__main__":
    generate_clean_yolo_files(ARQUIVO_JSON_ENTRADA, PASTA_SAIDA_FINAL)
    print("\nProcesso concluído!")
    print(f"Verifique a pasta '{PASTA_SAIDA_FINAL}'. Os arquivos devem conter apenas 5 colunas.")