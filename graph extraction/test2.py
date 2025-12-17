import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import distance
import os
import json
import math
from neo4j import GraphDatabase # ต้องลง pip install neo4j

# =========================================================
# 1. Config & Data Parsing
# =========================================================
# ... (ส่วน load_yaml_manual และ parse_yolo_polygon_file เหมือนเดิม) ...
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
        teeth_data.append({"id": fdi_str, "class_id": class_id, "pos": (centroid_x, centroid_y), "area": area, "jaw": jaw, "quadrant": quadrant})
    return teeth_data

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def build_dental_graph(teeth_data, case_id="unknown_case"):
    G = nx.Graph(); G.graph['case_id'] = case_id
    upper_teeth = [t for t in teeth_data if t['jaw'] == 'UPPER']
    lower_teeth = [t for t in teeth_data if t['jaw'] == 'LOWER']
    for t in teeth_data: G.add_node(t['id'], jaw=t['jaw'], abstract_pos=(t['pos'][0], 1 - t['pos'][1]), true_pos=t['pos'], area=t['area'])
    def connect_horizontal_strict(teeth_list):
        sorted_t = sorted(teeth_list, key=lambda k: k['pos'][0])
        for i in range(len(sorted_t) - 1):
            curr_t = sorted_t[i]; next_t = sorted_t[i+1]
            dist = distance.euclidean(curr_t['pos'], next_t['pos'])
            y_diff = abs(curr_t['pos'][1] - next_t['pos'][1])
            angle = calculate_angle(curr_t['pos'], next_t['pos'])
            if curr_t['jaw'] != next_t['jaw'] or y_diff > 0.1: continue 
            rel = "IS_ADJACENT_TO" if dist < 0.15 else "HAS_GAP_WITH"
            col = '#00FF00' if dist < 0.15 else '#FF0000'
            G.add_edge(curr_t['id'], next_t['id'], relation=rel, weight=dist, angle=angle, color=col)
    connect_horizontal_strict(upper_teeth); connect_horizontal_strict(lower_teeth)
    for u in upper_teeth:
        for l in lower_teeth:
            if abs(u['pos'][0]-l['pos'][0]) < 0.05 and abs(u['pos'][1]-l['pos'][1]) < 0.45:
                G.add_edge(u['id'], l['id'], relation="IS_OPPOSING", weight=abs(u['pos'][0]-l['pos'][0]), color='#0000FF')
    return G

def save_graph_as_json(G, output_filename):
    case_id = G.graph.get('case_id', 'unknown')
    export_data = {"case_info": {"id": case_id, "type": "Case"}, "nodes": [], "relationships": []}
    existing_classes = set()
    for n, attr in G.nodes(data=True):
        fdi_class = n; instance_id = f"{case_id}_{fdi_class}"
        export_data["nodes"].append({"id": instance_id, "labels": ["Tooth"], "properties": {"fdi": fdi_class, "area": attr['area'], "jaw": attr['jaw'], "pos_x": attr['true_pos'][0], "pos_y": attr['true_pos'][1]}})
        export_data["relationships"].append({"start": case_id, "end": instance_id, "type": "HAS", "properties": {}})
        if fdi_class not in existing_classes:
            export_data["nodes"].append({"id": fdi_class, "labels": ["ToothClass"], "properties": {"fdi": fdi_class}})
            existing_classes.add(fdi_class)
        export_data["relationships"].append({"start": instance_id, "end": fdi_class, "type": "IS_A", "properties": {}})
    for u, v, attr in G.edges(data=True):
        u_instance, v_instance = f"{case_id}_{u}", f"{case_id}_{v}"
        rel_type = "NEXT_TO" if "ADJACENT" in attr['relation'] or "GAP" in attr['relation'] else "OPPOSING"
        props = {"dist": attr['weight']}; 
        if 'angle' in attr: props['angle'] = attr['angle']
        export_data["relationships"].append({"start": u_instance, "end": v_instance, "type": rel_type, "properties": props, "original_relation_tag": attr['relation']})
    
    with open(output_filename, 'w') as f: json.dump(export_data, f, indent=4)
    print(f"✅ JSON saved to {output_filename}")
    return export_data # Return data for Neo4j insertion

# =========================================================
# VIZ 1, 2, 3 (เหมือนเดิม แต่ Viz 3 ใช้ Spring Layout)
# =========================================================
def overlay_graph_on_image(G, image_path, ax):
    if not os.path.exists(image_path): ax.text(0.5, 0.5, "Image not found", ha='center', va='center'); return
    img = mpimg.imread(image_path)
    if len(img.shape) == 2: img_h, img_w = img.shape; ax.imshow(img, cmap='gray')
    else: img_h, img_w = img.shape[:2]; ax.imshow(img)
    pos_pixel = {}
    for node in G.nodes():
        norm_pos = G.nodes[node]['true_pos']
        pos_pixel[node] = (norm_pos[0] * img_w, norm_pos[1] * img_h)
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos_pixel, node_size=300, node_color=node_colors, alpha=0.7, edgecolors='white', ax=ax)
    nx.draw_networkx_labels(G, pos_pixel, font_size=8, font_color='white', font_weight='bold', ax=ax)
    edges = G.edges(); edge_colors = [G[u][v]['color'] for u,v in edges]
    styles = ['dashed' if G[u][v]['relation'] == "HAS_GAP_WITH" else 'dotted' if G[u][v]['relation'] == "IS_OPPOSING" else 'solid' for u,v in edges]
    nx.draw_networkx_edges(G, pos_pixel, edge_color=edge_colors, style=styles, width=2, alpha=0.8, ax=ax)
    ax.set_title(f"1. Real Overlay ({os.path.basename(image_path)})"); ax.axis('off')

def plot_spatial_with_attributes(G, ax):
    pos = nx.get_node_attributes(G, 'abstract_pos')
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    edges = G.edges(data=True)
    colors = [d['color'] for u,v,d in edges]
    styles = ['dashed' if d['relation'] == "HAS_GAP_WITH" else 'dotted' if d['relation'] == "IS_OPPOSING" else 'solid' for u,v,d in edges]
    nx.draw_networkx_edges(G, pos, edge_color=colors, style=styles, width=2, ax=ax)
    edge_labels = { (u,v): f"D:{d['weight']:.2f}" for u, v, d in edges if 'angle' in d }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax, label_pos=0.5)
    ax.set_title("2. Extracted Graph Data"); ax.axis('off')

def plot_neo4j_schema_concept(G, ax):
    demo_G = nx.DiGraph()
    case_id = G.graph.get('case_id', 'Case_001')
    demo_G.add_node(case_id, type='Case', color='#FF6B6B', label='CASE')
    all_nodes_sorted = sorted(list(G.nodes()), key=lambda x: int(x) if x.isdigit() else 999)
    sample_nodes = all_nodes_sorted[:5] # Subset 5
    for n in sample_nodes:
        instance_id = f"{case_id}_{n}"; class_id = f"Class_{n}"
        demo_G.add_node(instance_id, type='Instance', color='#4ECDC4', label=f"{n}")
        demo_G.add_node(class_id, type='Class', color='#FFE66D', label=f"Std\n{n}")
        demo_G.add_edge(case_id, instance_id, label='HAS', color='gray', style='solid')
        demo_G.add_edge(instance_id, class_id, label='IS_A', color='gray', style='solid')
    for u, v, d in G.edges(data=True):
        if u in sample_nodes and v in sample_nodes:
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
    edge_lbls = nx.get_edge_attributes(demo_G, 'label')
    nx.draw_networkx_edge_labels(demo_G, pos, edge_labels=edge_lbls, font_size=7, ax=ax)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', label='Case', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', label='Instance', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFE66D', label='Class', markersize=10),
        Line2D([0], [0], color='#00CC00', lw=2, label='Next'),
        Line2D([0], [0], color='#FF0000', lw=2, linestyle='--', label='Gap')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
    ax.set_title("3. Neo4j Architecture (Spring Layout)"); ax.axis('off')

# =========================================================
# 4. NEO4J INSERTION FUNCTION (NEW!) 🚀
# =========================================================
def insert_into_neo4j(json_data, uri, user, password):
    """
    Connect to Neo4j and insert the Case, Nodes, and Relationships.
    Compatible with Neo4j Python Driver v5.x+
    """
    # ตรวจสอบการเชื่อมต่อ
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("🔌 Connected to Neo4j successfully.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    def create_graph(tx, data):
        # 1. Create/Merge Case Node
        case_info = data['case_info']
        tx.run("MERGE (c:Case {id: $id})", id=case_info['id'])

        # 2. Create Nodes (Tooth & ToothClass)
        for node in data['nodes']:
            labels = ":".join(node['labels']) # e.g. "Tooth" or "ToothClass"
            # ใช้ MERGE เพื่อกันข้อมูลซ้ำ
            query = f"MERGE (n:{labels} {{id: $id}}) SET n += $props"
            tx.run(query, id=node['id'], props=node['properties'])

        # 3. Create Relationships
        for rel in data['relationships']:
            rel_type = rel['type']
            # สร้าง Relationship (ต้องมี Node ปลายทางอยู่ก่อนแล้ว ซึ่งเราทำใน step 2 แล้ว)
            query = (
                "MATCH (a {id: $start_id}), (b {id: $end_id}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += $props"
            )
            tx.run(query, start_id=rel['start'], end_id=rel['end'], props=rel['properties'])

    try:
        with driver.session() as session:
            # ✅ แก้ตรงนี้: เปลี่ยนจาก write_transaction เป็น execute_write
            session.execute_write(create_graph, json_data)
            
        print(f"✅ Data inserted into Neo4j successfully for Case: {json_data['case_info']['id']}")
    except Exception as e:
        print(f"❌ Error inserting into Neo4j: {e}")
    finally:
        driver.close()

# =========================================================
# Main Execution
# =========================================================
txt_filename = '5.jpg.txt'
img_filename = '5.jpg'
json_filename = '5.json'
output_image_filename = '5_graph_viz.png' # ชื่อไฟล์รูปที่จะบันทึก

# Neo4j Config (ใส่ของคุณเอง)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

data = parse_yolo_polygon_file(txt_filename)

if data:
    # 1. Build Graph
    graph = build_dental_graph(data, case_id=img_filename)
    
    # 2. Save JSON & Get Data
    graph_json = save_graph_as_json(graph, json_filename)

    # 3. Plot & Save Image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    overlay_graph_on_image(graph, img_filename, ax=ax1) 
    plot_spatial_with_attributes(graph, ax=ax2)         
    plot_neo4j_schema_concept(graph, ax=ax3) 
    
    plt.tight_layout()
    plt.savefig(output_image_filename, dpi=300) # [NEW] บันทึกภาพ
    print(f"✅ Visualization saved to {output_image_filename}")
    plt.show()

    # 4. Insert into Neo4j (Uncomment เพื่อใช้งาน)
    insert_into_neo4j(graph_json, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

else:
    print("No data parsed.")