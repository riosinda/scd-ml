import os
import json
import numpy as np
from PIL import Image
# IMPORTANTE: Necesitas pycocotools instalado
import pycocotools.mask as mask_util


def create_coco_annotations(data_dir, output_json_path):
    """
    Convierte las máscaras binarias de un dataset personalizado 
    (ej. HAM10000) al formato COCO JSON.
    """
    
    # --- DEFINICIÓN DE RUTAS ---
    IMAGE_DIR = os.path.join(data_dir, "images")
    MASK_DIR = os.path.join(data_dir, "masks")
    
    # Asumimos que tienes una lista de nombres de archivos que coinciden
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".png")])
    
    # --- ESTRUCTURA COCO ---
    coco_json = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "lesion", "supercategory": "skin"}], # Única categoría
        "images": [],
        "annotations": []
    }
    
    annotation_id = 0
    
    for image_id, filename in enumerate(mask_files):
        
        # 1. CARGAR MÁSCARA
        mask_path = os.path.join(MASK_DIR, filename)
        
        try:
            # Usamos PIL para leer la máscara. La convertimos a escala de grises.
            mask = Image.open(mask_path).convert("L") 
            mask_np = np.array(mask)
        except Exception as e:
            print(f"Error al cargar la máscara {filename}: {e}")
            continue
        
        # --- 2. PROCESAR MÁSCARA Y ANOTACIONES ---
        # Si la máscara tiene píxeles de lesión (valor > 0):
        if mask_np.max() > 0:
            
            # Aseguramos que el array sea binario (0s y 1s) para la codificación RLE
            binary_mask = (mask_np > 0).astype(np.uint8)
            
            # Codificación RLE (Run-Length Encoding), estándar de COCO.
            # np.asfortranarray() es necesario para el correcto encodificado.
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8') # Decodificar para JSON serialization
            
            # Cálculo de BBox y Área a partir de la RLE
            bbox = mask_util.toBbox(rle).tolist()
            area = float(mask_util.area(rle))
            
            # Crear la entrada de anotación
            annotation = {
                "id": annotation_id,
                "image_id": image_id, # ¡El índice numérico clave!
                "category_id": 1, 
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0 # 0 para anotaciones de objeto singular
            }
            coco_json["annotations"].append(annotation)
            annotation_id += 1
        
        # --- 3. AGREGAR ENTRADA DE IMAGEN ---
        # Asumimos que las imágenes originales son JPG y tienen el mismo nombre base
        img_filename = filename.replace('.png', '.jpg') 
        img_path = os.path.join(IMAGE_DIR, img_filename) 

        try:
            img = Image.open(img_path)
            width, height = img.size
        except FileNotFoundError:
            # Si la imagen no está, usamos dimensiones de la máscara
            height, width = mask_np.shape[:2]
        
        image_info = {
            "id": image_id, # ¡Mismo ID que en la anotación!
            "file_name": img_filename,
            "width": width,
            "height": height
        }
        coco_json["images"].append(image_info)


    # --- 4. GUARDAR JSON ---
    with open(output_json_path, "w") as f:
        json.dump(coco_json, f, indent=4)
        
    print(f"\n--- GENERACIÓN EXITOSA ---")
    print(f"Archivo COCO generado en: {output_json_path}")
    print(f"Total de imágenes registradas: {len(coco_json['images'])}")
    print(f"Total de anotaciones (lesiones): {len(coco_json['annotations'])}")


