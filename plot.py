import cv2
import os

def plot_yolo_bboxes(img_path, txt_path, class_names=None, show_conf=False):
    """
    Plota bounding boxes no formato YOLO em uma imagem.

    Args:
        img_path (str): Caminho para o arquivo de imagem.
        txt_path (str): Caminho para o arquivo .txt com as anotações YOLO.
        class_names (list): Lista opcional com nomes das classes (ex: ['gato', 'cachorro']).
        show_conf (bool): Se o txt tiver confiança (6ª coluna), mostrar ela.
    """
    
    # 1. Verificar se arquivos existem
    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print("Erro: Imagem ou arquivo de texto não encontrados.")
        return

    # 2. Carregar imagem
    img = cv2.imread(img_path)
    if img is None:
        print("Erro: Não foi possível ler a imagem.")
        return

    # Dimensões da imagem (Altura, Largura)
    h_img, w_img, _ = img.shape

    # 3. Ler o arquivo de coordenadas
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Cores para as classes (B, G, R)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    print(f"Encontrados {len(lines)} objetos.")

    # 4. Processar cada linha
    for line in lines:
        parts = line.strip().split()
        
        # O formato YOLO é: class_id center_x center_y width height [conf]
        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        # 5. Converter de Normalizado (0-1) para Pixels Absolutos
        # O YOLO dá o centro do objeto, precisamos do canto superior esquerdo para desenhar
        x_center = int(x_center_norm * w_img)
        y_center = int(y_center_norm * h_img)
        box_w = int(width_norm * w_img)
        box_h = int(height_norm * h_img)

        # Canto superior esquerdo (x1, y1) e inferior direito (x2, y2)
        x1 = int(x_center - (box_w / 2))
        y1 = int(y_center - (box_h / 2))
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Escolher cor baseada no ID da classe
        color = colors[class_id % len(colors)]

        # 6. Desenhar o Retângulo
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 7. Adicionar o texto (Nome da classe ou ID)
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"ID {class_id}"
        
        # Adicionar fundo no texto para facilitar leitura
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1) # Fundo preenchido
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 8. Mostrar o resultado
    cv2.imshow("YOLO Bounding Boxes", img)
    
    # Pressione 'q' para fechar ou espere indefinidamente
    print("Pressione qualquer tecla na janela da imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Opcional: Salvar a imagem processada
    # cv2.imwrite("resultado_yolo.jpg", img)

# --- CONFIGURAÇÃO E USO ---
if __name__ == "__main__":
    # Coloque aqui os nomes dos seus arquivos
    imagem = "exemplo.jpg" 
    arquivo_txt = "exemplo.txt"
    
    # Se você souber os nomes das classes (ex: 0 é 'pessoa', 1 é 'carro')
    nomes_classes = ["classe0", "classe1", "classe2"] 

    plot_yolo_bboxes(imagem, arquivo_txt, nomes_classes)