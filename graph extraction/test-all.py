import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import distance
import os
import json
import math
import glob
from neo4j import GraphDatabase # pip install neo4j

# =========================================================
# 0. Configuration
# =========================================================
DATA_FOLDER = 'img_ann'      
OUTPUT_DIR = 'output_results' 

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678" # <--- แก้รหัสผ่านตรงนี้

# =========================================================
# 1. Helper Functions & Data Parsing
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
    if not os.path.exists(file_path): return []
    with open(file_path, 'r') as f: lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        points = np.array(coords).reshape(-1, 2)
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        x = points[:, 0]; y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        fdi_str = CLASS_TO_FDI.get(class_id, str(class_id))
        
        if fdi_str.isdigit():
            quadrant = int(fdi_str[0])
            jaw = "UPPER" if quadrant in [1, 2] else "LOWER" if quadrant in [3, 4] else "UNKNOWN"
        else: jaw = "UNKNOWN"; quadrant = 0
            
        teeth_data.append({
            "id": fdi_str, "class_id": class_id, "pos": (centroid_x, centroid_y), "area": area, "jaw": jaw, "quadrant": quadrant
        })
    return teeth_data

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_fdi_value(fdi_str):
    if fdi_str.isdigit(): return int(fdi_str)
    return 0

def is_primary_tooth(fdi_val):
    return 50 <= fdi_val <= 85

# =========================================================
# 2. Graph Building Logic
# =========================================================
def build_dental_graph(teeth_data, case_id="unknown_case"):
    G = nx.Graph()
    G.graph['case_id'] = case_id
    
    # Create Nodes
    for i, t in enumerate(teeth_data):
        unique_id = f"tooth_{i}" 
        G.add_node(unique_id, fdi=t['id'], jaw=t['jaw'], abstract_pos=(t['pos'][0], 1 - t['pos'][1]), true_pos=t['pos'], area=t['area'])

    nodes_data = list(G.nodes(data=True))
    upper_nodes = [n for n in nodes_data if n[1]['jaw'] == 'UPPER']
    lower_nodes = [n for n in nodes_data if n[1]['jaw'] == 'LOWER']

    def connect_horizontal_smart(node_list):
        sorted_nodes = sorted(node_list, key=lambda x: x[1]['true_pos'][0])
        
        for i in range(len(sorted_nodes) - 1):
            curr_id, curr_props = sorted_nodes[i]
            next_id, next_props = sorted_nodes[i+1]
            
            dist = distance.euclidean(curr_props['true_pos'], next_props['true_pos'])
            angle = calculate_angle(curr_props['true_pos'], next_props['true_pos'])
            
            dx = abs(curr_props['true_pos'][0] - next_props['true_pos'][0])
            dy = abs(curr_props['true_pos'][1] - next_props['true_pos'][1])

            curr_fdi = get_fdi_value(curr_props['fdi'])
            next_fdi = get_fdi_value(next_props['fdi'])
            curr_quad = curr_fdi // 10
            next_quad = next_fdi // 10
            
            is_gap = False
            is_vertical_stack = dy > dx
            
            if is_vertical_stack:
                is_gap = False # ซ้อนแนวตั้ง = Next
            else:
                # เช็คฟันผสม (Mixed Dentition)
                is_mixed_pair = is_primary_tooth(curr_fdi) != is_primary_tooth(next_fdi)
                
                if is_mixed_pair:
                    is_gap = False # ฟันผสม = Next
                else:
                    # Logic ปกติ
                    if curr_quad != next_quad:
                        # Midline Rule: ต้องเป็น 1-1 เท่านั้นถึงจะ Next
                        if (curr_fdi % 10 == 1) and (next_fdi % 10 == 1):
                            is_gap = False 
                        else:
                            is_gap = True # ข้าม Midline ผิดคู่ = Gap
                    else:
                        is_fdi_jump = abs(curr_fdi - next_fdi) > 1
                        is_physical_gap = dist >= 0.15
                        if is_fdi_jump or is_physical_gap:
                            is_gap = True

            rel_type = "IS_ADJACENT_TO"
            color = '#00FF00'
            if is_gap:
                rel_type = "HAS_GAP_WITH"
                color = '#FF0000'
            
            G.add_edge(curr_id, next_id, relation=rel_type, weight=dist, angle=angle, color=color)

    connect_horizontal_smart(upper_nodes)
    connect_horizontal_smart(lower_nodes)
    
    # Connect Opposing
    for u_id, u_props in upper_nodes:
        for l_id, l_props in lower_nodes:
            x_dist = abs(u_props['true_pos'][0] - l_props['true_pos'][0])
            y_dist = abs(u_props['true_pos'][1] - l_props['true_pos'][1])
            if x_dist < 0.05 and y_dist < 0.45:
                G.add_edge(u_id, l_id, relation="IS_OPPOSING", weight=x_dist, color='#0000FF')
    return G

def save_graph_as_json(G, output_filename):
    case_id = G.graph.get('case_id', 'unknown')
    
    # --- [UPDATED] เพิ่ม Properties ให้ Case Info ---
    case_properties = {
        "radiograph_type": "Panoramic Radiograph",
        "dataset_source": "img_ann" 
    }
    
    export_data = {
        "case_info": {
            "id": case_id, 
            "type": "Case",
            "properties": case_properties # <--- ใส่ตรงนี้
        }, 
        "nodes": [], 
        "relationships": []
    }
    existing_classes = set()
    
    for n_id, attr in G.nodes(data=True):
        fdi_class = attr['fdi']
        instance_id = f"{case_id}_{n_id}" 
        
        export_data["nodes"].append({
            "id": instance_id, "labels": ["Tooth"], 
            "properties": {"fdi": fdi_class, "area": attr['area'], "jaw": attr['jaw'], "pos_x": attr['true_pos'][0], "pos_y": attr['true_pos'][1]}
        })
        export_data["relationships"].append({"start": case_id, "end": instance_id, "type": "HAS", "properties": {}})
        
        if fdi_class not in existing_classes:
            export_data["nodes"].append({"id": fdi_class, "labels": ["ToothClass"], "properties": {"fdi": fdi_class}})
            existing_classes.add(fdi_class)
        export_data["relationships"].append({"start": instance_id, "end": fdi_class, "type": "IS_A", "properties": {}})

    for u, v, attr in G.edges(data=True):
        u_instance = f"{case_id}_{u}"; v_instance = f"{case_id}_{v}"
        rel_type = "NEXT_TO" if "ADJACENT" in attr['relation'] or "GAP" in attr['relation'] else "OPPOSING"
        props = {"dist": attr['weight']}
        if 'angle' in attr: props['angle'] = attr['angle']
        if "GAP" in attr['relation']: props['is_gap'] = True
        export_data["relationships"].append({"start": u_instance, "end": v_instance, "type": rel_type, "properties": props, "original_relation_tag": attr['relation']})
    
    with open(output_filename, 'w') as f: json.dump(export_data, f, indent=4)
    return export_data 

# =========================================================
# 3. Visualization
# =========================================================
def overlay_graph_on_image(G, image_path, ax):
    if not os.path.exists(image_path): ax.text(0.5, 0.5, "Image not found", ha='center', va='center'); return
    img = mpimg.imread(image_path)
    if len(img.shape) == 2: img_h, img_w = img.shape; ax.imshow(img, cmap='gray')
    else: img_h, img_w = img.shape[:2]; ax.imshow(img)
    pos_pixel = {}; labels = {}
    for node, attr in G.nodes(data=True):
        norm_pos = attr['true_pos']
        pos_pixel[node] = (norm_pos[0] * img_w, norm_pos[1] * img_h)
        labels[node] = attr['fdi'] 
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos_pixel, node_size=300, node_color=node_colors, alpha=0.7, edgecolors='white', ax=ax)
    nx.draw_networkx_labels(G, pos_pixel, labels=labels, font_size=8, font_color='white', font_weight='bold', ax=ax)
    edges = G.edges(); edge_colors = [G[u][v]['color'] for u,v in edges]
    styles = ['dashed' if G[u][v]['relation'] == "HAS_GAP_WITH" else 'dotted' if G[u][v]['relation'] == "IS_OPPOSING" else 'solid' for u,v in edges]
    nx.draw_networkx_edges(G, pos_pixel, edge_color=edge_colors, style=styles, width=2, alpha=0.8, ax=ax)
    ax.set_title(f"1. Real Overlay ({os.path.basename(image_path)})"); ax.axis('off')

def plot_spatial_with_attributes(G, ax):
    pos = nx.get_node_attributes(G, 'abstract_pos')
    labels = {n: G.nodes[n]['fdi'] for n in G.nodes()} 
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', ax=ax)
    edges = G.edges(data=True)
    colors = [d['color'] for u,v,d in edges]
    styles = ['dashed' if d['relation'] == "HAS_GAP_WITH" else 'dotted' if d['relation'] == "IS_OPPOSING" else 'solid' for u,v,d in edges]
    nx.draw_networkx_edges(G, pos, edge_color=colors, style=styles, width=2, ax=ax)
    edge_labels = { (u,v): f"D:{d['weight']:.2f}" for u, v, d in edges if 'angle' in d }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax, label_pos=0.5)
    ax.set_title("2. Extracted Graph Data"); ax.axis('off')

def plot_neo4j_schema_concept(G, ax):
    demo_G = nx.DiGraph(); case_id = G.graph.get('case_id', 'Case_001')
    demo_G.add_node(case_id, type='Case', color='#FF6B6B', label='CASE')
    all_nodes_sorted = sorted(list(G.nodes(data=True)), key=lambda x: int(x[1]['fdi']) if x[1]['fdi'].isdigit() else 999)
    sample_nodes = all_nodes_sorted[:5]; sample_ids = [n[0] for n in sample_nodes]
    for n_id, attr in sample_nodes:
        fdi = attr['fdi']; instance_id = f"{case_id}_{n_id}"; class_id = f"Class_{fdi}"
        demo_G.add_node(instance_id, type='Instance', color='#4ECDC4', label=f"{fdi}")
        demo_G.add_node(class_id, type='Class', color='#FFE66D', label=f"Std\n{fdi}")
        demo_G.add_edge(case_id, instance_id, label='HAS', color='gray', style='solid')
        demo_G.add_edge(instance_id, class_id, label='IS_A', color='gray', style='solid')
    for u, v, d in G.edges(data=True):
        if u in sample_ids and v in sample_ids:
            u_inst = f"{case_id}_{u}"; v_inst = f"{case_id}_{v}"
            lbl = 'NEXT'; col = 'gray'; sty = 'solid'
            if 'relation' in d:
                if 'ADJACENT' in d['relation']: lbl = 'NEXT'; col = '#00CC00'; sty = 'solid'
                elif 'GAP' in d['relation']: lbl = 'GAP'; col = '#FF0000'; sty = 'dashed'
                elif 'OPPOSING' in d['relation']: lbl = 'OPP'; col = '#0000FF'; sty = 'dotted'
            demo_G.add_edge(u_inst, v_inst, label=lbl, color=col, style=sty)
    pos = nx.spring_layout(demo_G, seed=42, k=0.5)
    colors = [nx.get_node_attributes(demo_G, 'color')[n] for n in demo_G.nodes()]
    labels = nx.get_node_attributes(demo_G, 'label')
    edge_colors = [demo_G[u][v]['color'] for u,v in demo_G.edges()]
    edge_styles = [demo_G[u][v]['style'] for u,v in demo_G.edges()]
    nx.draw_networkx_nodes(demo_G, pos, node_size=1200, node_color=colors, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(demo_G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(demo_G, pos, edge_color=edge_colors, style=edge_styles, width=1.5, arrowstyle='->', arrowsize=15, ax=ax)
    nx.draw_networkx_edge_labels(demo_G, pos, edge_labels=nx.get_edge_attributes(demo_G, 'label'), font_size=7, ax=ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', label='Case'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', label='Instance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFE66D', label='Class'),
        Line2D([0], [0], color='#00CC00', lw=2, label='Next'),
        Line2D([0], [0], color='#FF0000', lw=2, linestyle='--', label='Gap')
    ], loc='lower right', fontsize='small')
    ax.set_title("3. Neo4j Architecture (Spring Layout)"); ax.axis('off')

# =========================================================
# 4. Neo4j Bulk Insert (Fixed Syntax Error + Props)
# =========================================================
def insert_into_neo4j_bulk(json_data_list, uri, user, password):
    print(f"🔌 Connecting to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password)); driver.verify_connectivity()
    except Exception as e: print(f"❌ Connection failed: {e}"); return

    def create_graph(tx, data):
        # 1. Case: [UPDATED] ใช้ SET c += $props เพื่อเพิ่ม property
        tx.run("MERGE (c:Case {id: $id}) SET c += $props", 
               id=data['case_info']['id'], 
               props=data['case_info']['properties'])
        
        # 2. Nodes
        for node in data['nodes']:
            labels = ":".join(node['labels'])
            tx.run(f"MERGE (n:{labels} {{id: $id}}) SET n += $props", id=node['id'], props=node['properties'])
        
        # 3. Relationships
        for rel in data['relationships']:
            rel_type = rel['type']
            part_match = "MATCH (a {id: $start_id}), (b {id: $end_id}) "
            part_merge = f"MERGE (a)-[r:{rel_type}]->(b) "
            part_set = "SET r += $props"
            query = part_match + part_merge + part_set
            tx.run(query, start_id=rel['start'], end_id=rel['end'], props=rel['properties'])

    success_count = 0
    with driver.session() as session:
        for json_data in json_data_list:
            try: session.execute_write(create_graph, json_data); success_count += 1
            except Exception as e: print(f"  ❌ Error inserting {json_data['case_info']['id']}: {e}")
    driver.close(); print(f"✅ Bulk Insert Completed! Success: {success_count}/{len(json_data_list)}")

# =========================================================
# 5. Main Loop
# =========================================================
def main():
    txt_files = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
    print(f"📂 Found {len(txt_files)} annotation files in '{DATA_FOLDER}'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_graph_json_data = []

    for txt_file in txt_files:
        base_name = os.path.basename(txt_file)
        img_file = txt_file.replace(".txt", "") 
        if not os.path.exists(img_file):
            if os.path.exists(txt_file.replace(".txt", ".png")): img_file = txt_file.replace(".txt", ".png")
            elif os.path.exists(txt_file.replace(".txt", ".jpeg")): img_file = txt_file.replace(".txt", ".jpeg")
            elif os.path.exists(txt_file.replace(".txt", ".jpg")): img_file = txt_file.replace(".txt", ".jpg")
            else: img_file = None

        print(f"Processing: {base_name}...")
        data = parse_yolo_polygon_file(txt_file)
        if not data: continue

        case_id = os.path.basename(img_file) if img_file else base_name
        graph = build_dental_graph(data, case_id=case_id)
        json_out_path = os.path.join(OUTPUT_DIR, f"{case_id}.json")
        graph_json = save_graph_as_json(graph, json_out_path)
        all_graph_json_data.append(graph_json)

        if img_file:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
            overlay_graph_on_image(graph, img_file, ax=ax1) 
            plot_spatial_with_attributes(graph, ax=ax2)         
            plot_neo4j_schema_concept(graph, ax=ax3) 
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{case_id}_viz.png"), dpi=150)
            plt.close(fig)

    print("="*50)
    if all_graph_json_data:
        insert_into_neo4j_bulk(all_graph_json_data, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    else:
        print("❌ No valid data to insert.")

if __name__ == "__main__":
    main()