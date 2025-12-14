import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import distance
import os

# =========================================================
# 1. Config & Data Parsing (เหมือนเดิม)
# =========================================================
def load_yaml_manual(yaml_content):
    id_map = {}
    lines = yaml_content.split('\n')
    is_names_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('names:'):
            is_names_section = True
            continue
        if is_names_section and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    class_id = int(parts[0].strip())
                    name_val = parts[1].strip().replace("'", "").replace('"', "")
                    id_map[class_id] = name_val
                except ValueError:
                    continue
    return id_map

yaml_raw = """
nc: 32
names:
  0: '11'
  1: '12'
  2: '13'
  3: '14'
  4: '15'
  5: '16'
  6: '17'
  7: '18'
  8: '21'
  9: '22'
  10: '23'
  11: '24'
  12: '25'
  13: '26'
  14: '27'
  15: '28'
  16: '31'
  17: '32'
  18: '33'
  19: '34'
  20: '35'
  21: '36'
  22: '37'
  23: '38'
  24: '41'
  25: '42'
  26: '43'
  27: '44'
  28: '45'
  29: '46'
  30: '47'
  31: '48'
"""
CLASS_TO_FDI = load_yaml_manual(yaml_raw)

def parse_yolo_polygon_file(file_path):
    teeth_data = []
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        points = np.array(coords).reshape(-1, 2)
        
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        fdi_str = CLASS_TO_FDI.get(class_id, str(class_id))
        if fdi_str.isdigit():
            quadrant = int(fdi_str[0])
            jaw = "UPPER" if quadrant in [1, 2] else "LOWER" if quadrant in [3, 4] else "UNKNOWN"
        else:
            jaw = "UNKNOWN"
            quadrant = 0

        teeth_data.append({
            "id": fdi_str, "class_id": class_id, "pos": (centroid_x, centroid_y),
            "area": area, "jaw": jaw, "quadrant": quadrant
        })
    return teeth_data

def build_dental_graph(teeth_data):
    G = nx.Graph()
    upper_teeth = [t for t in teeth_data if t['jaw'] == 'UPPER']
    lower_teeth = [t for t in teeth_data if t['jaw'] == 'LOWER']
    
    for t in teeth_data:
        G.add_node(t['id'], jaw=t['jaw'],
                   abstract_pos=(t['pos'][0], 1 - t['pos'][1]), 
                   true_pos=t['pos'], area=t['area'])

    def connect_horizontal_strict(teeth_list):
        sorted_t = sorted(teeth_list, key=lambda k: k['pos'][0])
        for i in range(len(sorted_t) - 1):
            curr_t = sorted_t[i]
            next_t = sorted_t[i+1]
            dist = distance.euclidean(curr_t['pos'], next_t['pos'])
            y_diff = abs(curr_t['pos'][1] - next_t['pos'][1])
            if curr_t['jaw'] != next_t['jaw']: continue
            if y_diff > 0.1: continue 
            if dist < 0.15:
                G.add_edge(curr_t['id'], next_t['id'], relation="IS_ADJACENT_TO", weight=dist, color='#00FF00')
            else:
                G.add_edge(curr_t['id'], next_t['id'], relation="HAS_GAP_WITH", weight=dist, color='#FF0000')

    connect_horizontal_strict(upper_teeth)
    connect_horizontal_strict(lower_teeth)
    
    for u_tooth in upper_teeth:
        for l_tooth in lower_teeth:
            x_dist = abs(u_tooth['pos'][0] - l_tooth['pos'][0])
            y_dist = abs(u_tooth['pos'][1] - l_tooth['pos'][1])
            if x_dist < 0.05 and y_dist < 0.45:
                G.add_edge(u_tooth['id'], l_tooth['id'], relation="IS_OPPOSING", weight=x_dist, color='#0000FF')
    return G

# =========================================================
# 2. Visualization Functions (ปรับปรุงให้รับ ax)
# =========================================================

def plot_abstract_graph(G, ax):
    """ Plot Abstract Graph ลงใน ax ที่กำหนด """
    pos = nx.get_node_attributes(G, 'abstract_pos')
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['orange' if jaws[n] == 'UPPER' else 'skyblue' for n in G.nodes()]
    
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    styles = ['dashed' if G[u][v]['relation'] == "HAS_GAP_WITH" 
              else 'dotted' if G[u][v]['relation'] == "IS_OPPOSING" 
              else 'solid' for u,v in edges]

    # สังเกตการใส่ ax=ax ลงในฟังก์ชันวาด
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=colors, style=styles, width=2, ax=ax)
    
    ax.set_title("1. Abstract Schema")
    ax.axis('off')

def overlay_graph_on_image(G, image_path, ax):
    """ Plot Overlay Graph ลงใน ax ที่กำหนด """
    if not os.path.exists(image_path):
        ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
        return

    # Load Image & Handle Grayscale
    img = mpimg.imread(image_path)
    if len(img.shape) == 2: # Grayscale
        img_h, img_w = img.shape
        ax.imshow(img, cmap='gray')
    else: # RGB
        img_h, img_w = img.shape[:2]
        ax.imshow(img)
    
    # Map Coordinates
    pos_pixel = {}
    for node in G.nodes():
        norm_pos = G.nodes[node]['true_pos']
        pos_pixel[node] = (norm_pos[0] * img_w, norm_pos[1] * img_h)
    
    # Draw
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos_pixel, node_size=300, node_color=node_colors, alpha=0.7, edgecolors='white', ax=ax)
    nx.draw_networkx_labels(G, pos_pixel, font_size=8, font_color='white', font_weight='bold', ax=ax)
    
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u,v in edges]
    styles = ['dashed' if G[u][v]['relation'] == "HAS_GAP_WITH" 
              else 'dotted' if G[u][v]['relation'] == "IS_OPPOSING" 
              else 'solid' for u,v in edges]
    
    nx.draw_networkx_edges(G, pos_pixel, edge_color=edge_colors, style=styles, width=2, alpha=0.8, ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00FF00', lw=2, label='Adjacent'),
        Line2D([0], [0], color='#FF0000', lw=2, linestyle='--', label='Gap'),
        Line2D([0], [0], color='#0000FF', lw=2, linestyle=':', label='Opposing')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    ax.set_title(f"2. Overlay on {os.path.basename(image_path)}")
    ax.axis('off')

# =========================================================
# Main Execution
# =========================================================

txt_filename = '5.jpg.txt'
img_filename = '5.jpg'

data = parse_yolo_polygon_file(txt_filename)

if data:
    graph = build_dental_graph(data)

    # --- สร้าง Figure ใหญ่ 1 อัน แล้วแบ่งเป็น 2 ช่อง (1 แถว 2 คอลัมน์) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # เรียกฟังก์ชันวาด โดยส่ง ax เข้าไป
    plot_abstract_graph(graph, ax=ax1)            # วาดลงช่องซ้าย
    overlay_graph_on_image(graph, img_filename, ax=ax2) # วาดลงช่องขวา
    
    plt.tight_layout() # จัดระเบียบไม่ให้ซ้อนกัน
    plt.show()
else:
    print("No data parsed.")