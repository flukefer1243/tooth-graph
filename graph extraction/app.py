import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from PIL import Image
import io

# =========================================================
# 1. Config & Utility Functions
# =========================================================

def get_fdi_mapping():
    """ Returns the dictionary mapping YOLO class ID to FDI Number """
    mapping = {
        0: '11', 1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18',
        8: '21', 9: '22', 10: '23', 11: '24', 12: '25', 13: '26', 14: '27', 15: '28',
        16: '31', 17: '32', 18: '33', 19: '34', 20: '35', 21: '36', 22: '37', 23: '38',
        24: '41', 25: '42', 26: '43', 27: '44', 28: '45', 29: '46', 30: '47', 31: '48'
    }
    return mapping

CLASS_TO_FDI = get_fdi_mapping()

def parse_yolo_data(txt_content):
    teeth_data = []
    lines = txt_content.split('\n')
        
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        points = np.array(coords).reshape(-1, 2)
        
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        fdi_str = CLASS_TO_FDI.get(class_id, str(class_id))
        
        if fdi_str.isdigit():
            quadrant = int(fdi_str[0])
            if quadrant in [1, 2]:
                jaw = "UPPER"
            elif quadrant in [3, 4]:
                jaw = "LOWER"
            else:
                jaw = "UNKNOWN"
        else:
            jaw = "UNKNOWN"
            quadrant = 0

        teeth_data.append({
            "id": fdi_str,
            "class_id": class_id,
            "pos": (centroid_x, centroid_y),
            "area": area,
            "jaw": jaw,
            "quadrant": quadrant
        })
        
    return teeth_data

def build_dental_graph(teeth_data):
    G = nx.Graph()
    
    upper_teeth = [t for t in teeth_data if t['jaw'] == 'UPPER']
    lower_teeth = [t for t in teeth_data if t['jaw'] == 'LOWER']
    
    for t in teeth_data:
        G.add_node(t['id'], 
                   jaw=t['jaw'],
                   abstract_pos=(t['pos'][0], 1 - t['pos'][1]), 
                   true_pos=t['pos'],
                   area=t['area'])

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
# 2. Plotting Functions
# =========================================================

def plot_abstract_graph(G, ax):
    pos = nx.get_node_attributes(G, 'abstract_pos')
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    styles = ['dashed' if G[u][v]['relation'] == "HAS_GAP_WITH" 
              else 'dotted' if G[u][v]['relation'] == "IS_OPPOSING" 
              else 'solid' for u,v in edges]

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=colors, style=styles, width=2, ax=ax)
    
    # Custom Legend for Abstract Graph
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00FF00', lw=2, label='Adjacent'),
        Line2D([0], [0], color='#FF0000', lw=2, linestyle='--', label='Gap'),
        Line2D([0], [0], color='#0000FF', lw=2, linestyle=':', label='Opposing'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=10, label='Upper Jaw'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=10, label='Lower Jaw')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize='small')
    ax.set_title("Abstract Knowledge Graph Schema", fontsize=14, fontweight='bold')
    ax.axis('off')

def overlay_graph_on_image(G, img_array, ax):
    # Handle Grayscale vs RGB
    if len(img_array.shape) == 2:
        img_h, img_w = img_array.shape
        ax.imshow(img_array, cmap='gray')
    else:
        img_h, img_w = img_array.shape[:2]
        ax.imshow(img_array)
    
    pos_pixel = {}
    for node in G.nodes():
        norm_pos = G.nodes[node]['true_pos']
        pos_pixel[node] = (norm_pos[0] * img_w, norm_pos[1] * img_h)
    
    jaws = nx.get_node_attributes(G, 'jaw')
    node_colors = ['#FFA500' if jaws[n] == 'UPPER' else '#87CEEB' for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos_pixel, node_size=400, node_color=node_colors, alpha=0.7, edgecolors='white', ax=ax)
    nx.draw_networkx_labels(G, pos_pixel, font_size=9, font_color='white', font_weight='bold', ax=ax)
    
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u,v in edges]
    styles = ['dashed' if G[u][v]['relation'] == "HAS_GAP_WITH" 
              else 'dotted' if G[u][v]['relation'] == "IS_OPPOSING" 
              else 'solid' for u,v in edges]
    
    nx.draw_networkx_edges(G, pos_pixel, edge_color=edge_colors, style=styles, width=2, alpha=0.8, ax=ax)

    ax.set_title("Overlay Graph on X-ray Image", fontsize=14, fontweight='bold')
    ax.axis('off')

# =========================================================
# 3. Streamlit App Layout
# =========================================================

st.set_page_config(page_title="Dental Knowledge Graph", layout="centered") 
# เปลี่ยน layout เป็น centered เพื่อให้ดูง่ายขึ้นสำหรับการเรียงแนวตั้ง

st.title("🦷 Dental Knowledge Graph Generator")
st.markdown("Upload your X-ray image and YOLO Label file to generate the graph.")

# Sidebar for Uploads
with st.sidebar:
    st.header("Upload Files")
    uploaded_img = st.file_uploader("1. Upload Image (jpg, png)", type=['jpg', 'jpeg', 'png'])
    uploaded_txt = st.file_uploader("2. Upload Label (.txt)", type=['txt'])

if uploaded_img and uploaded_txt:
    stringio = io.StringIO(uploaded_txt.getvalue().decode("utf-8"))
    txt_content = stringio.read()
    
    teeth_data = parse_yolo_data(txt_content)
    
    if not teeth_data:
        st.error("Could not parse data from the text file. Please check the format.")
    else:
        G = build_dental_graph(teeth_data)
        
        image = Image.open(uploaded_img)
        img_array = np.array(image)
        
        st.success(f"Graph generated successfully with {G.number_of_nodes()} teeth nodes.")
        
        # --- ROW 1: Abstract Graph ---
        st.markdown("---")
        st.header("1. Abstract Graph Structure")
        # ใช้ Figure แยกอิสระ เพื่อปรับขนาดให้เหมาะกับแนวนอน
        fig1, ax1 = plt.subplots(figsize=(10, 5)) 
        plot_abstract_graph(G, ax=ax1)
        st.pyplot(fig1)
        
        # --- ROW 2: Overlay Graph ---
        st.markdown("---")
        st.header("2. Image Overlay Visualization")
        # ใช้ Figure แยกอิสระ ปรับขนาดให้เป็นสี่เหลี่ยมจัตุรัสหรือตามรูป
        fig2, ax2 = plt.subplots(figsize=(10, 10)) 
        overlay_graph_on_image(G, img_array, ax=ax2)
        st.pyplot(fig2)
        
        # --- Data Table ---
        with st.expander("See Graph Schema Dump"):
            # สร้างตัวแปร list เพื่อเก็บเกี่ยวบรรทัดข้อความ
            lines = []
            
            # --- Header ---
            lines.append("="*50)
            lines.append("      KNOWLEDGE GRAPH SCHEMA DUMP (FDI)")
            lines.append("="*50)
            
            # --- NODES SECTION ---
            # เรียงลำดับ Node ตามตัวเลข (เช่น 11, 12, ..., 48)
            sorted_nodes = sorted(G.nodes(), key=lambda x: int(x) if x.isdigit() else 999)
            
            lines.append(f"\n[NODES] Total: {G.number_of_nodes()}")
            lines.append(f"{'FDI':<6} | {'JAW':<6} | {'POS (Norm)'}")
            
            for n in sorted_nodes:
                data = G.nodes[n]
                pos_str = f"({data['true_pos'][0]:.2f}, {data['true_pos'][1]:.2f})"
                lines.append(f"{n:<6} | {data['jaw']:<6} | {pos_str}")
            
            # --- EDGES SECTION ---
            # เรียงลำดับ Edge ตามโหนดต้นทาง -> ปลายทาง
            sorted_edges = sorted(G.edges(data=True), 
                                key=lambda x: (int(x[0]) if x[0].isdigit() else 999, 
                                                int(x[1]) if x[1].isdigit() else 999))
            
            lines.append(f"\n[EDGES] Total: {G.number_of_edges()}")
            lines.append(f"{'Source':<7} -> {'Target':<7}| {'RELATION'}")
            
            for u, v, data in sorted_edges:
                lines.append(f"{u:<7} -> {v:<7}| {data['relation']}")
            
            lines.append("="*50)
            
            # รวมทุกบรรทัดเป็นข้อความเดียว
            report_text = "\n".join(lines)
            
            # แสดงผลแบบ Code block เพื่อรักษา font แบบ monospaced (ให้ตารางตรงกัน)
            st.text(report_text)

else:
    st.info("Please upload both Image and Label files to proceed.")