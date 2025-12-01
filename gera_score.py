import os
import json
import glob
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

# --- Configurações ---
ARQUIVO_JSON_ENTRADA = "dados_para_wbf.json"  # O arquivo gerado no script anterior
PASTA_SAIDA_FINAL = "resultado_final_wbf"     # Onde os txts finais serão salvos

# Parâmetros do WBF
IOU_THRESHOLD = 0.55  # Se as caixas médias tiverem IoU > 0.55, elas serão fundidas novamente
SKIP_BOX_THR = 0.001  # Ignora caixas com score muito baixo (quase zero)

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

def run_wbf_and_save(json_path, output_folder):
    # Verifica se o arquivo JSON existe
    if not os.path.exists(json_path):
        print(f"Erro: O arquivo {json_path} não foi encontrado.")
        return

    # Cria a pasta de saída
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Carregando dados do JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_images = len(data)
    print(f"Iniciando WBF em {total_images} imagens...")

    # Itera sobre cada imagem no JSON
    for image_id, content in tqdm(data.items(), desc="Aplicando WBF"):
        
        # Extrai as listas do JSON
        boxes = content['boxes']   # Já estão em [x1, y1, x2, y2] normalizado
        scores = content['scores']
        labels = content['labels']

        # Se não houver caixas para esta imagem, cria um txt vazio e pula
        if len(boxes) == 0:
            open(os.path.join(output_folder, f"{image_id}.txt"), 'w').close()
            continue

        # --- PREPARAÇÃO PARA WBF ---
        # A biblioteca espera uma lista de listas (uma lista para cada modelo).
        # Como consolidamos tudo em um único "modelo mestre", encapsulamos em []:
        boxes_list = [boxes]
        scores_list = [scores]
        labels_list = [labels]
        weights = [1] # Peso 1, pois é o único input

        # --- RODA O WBF ---
        try:
            boxes_final, scores_final, labels_final = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=IOU_THRESHOLD,
                skip_box_thr=SKIP_BOX_THR,
                conf_type='avg' # Se houver fusão, tira a média dos scores
            )
        except Exception as e:
            print(f"Erro ao processar imagem {image_id}: {e}")
            continue

        # --- SALVA O RESULTADO ---
        output_path = os.path.join(output_folder, f"{image_id}.txt")
        
        with open(output_path, 'w') as f_out:
            for b, s, l in zip(boxes_final, scores_final, labels_final):
                # Converte de volta para YOLO (xc, yc, w, h) para salvar
                yolo_box = x1y1x2y2_to_yolo(b)
                
                # Formato final: class score xc yc w h
                # Usamos 6 casas decimais para precisão
                line = f"{int(l)} {s:.6f} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                f_out.write(line)

if __name__ == "__main__":
    run_wbf_and_save(ARQUIVO_JSON_ENTRADA, PASTA_SAIDA_FINAL)
    print(f"\nSucesso! Arquivos finais gerados em: {PASTA_SAIDA_FINAL}")