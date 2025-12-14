import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib as mpl

# ==========================================
# 1. แก้ไขปัญหาภาษาไทย (เพิ่มตรงนี้)
# ==========================================
# ถ้าใช้ Windows ให้ใช้ 'Tahoma' หรือ 'Microsoft Sans Serif'
mpl.rcParams['font.family'] = 'Tahoma' 
# หมายเหตุ: ถ้าใช้ Mac ให้เปลี่ยนเป็น 'Thonburi'

# --- 1. ฟังก์ชันอ่านไฟล์ (คงเดิม) ---
def get_class_data_from_file(filename, target_class_id):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == str(target_class_id):
                coords = np.array([float(p) for p in parts[1:]])
                if len(coords) % 2 != 0: coords = coords[:-1]
                points = coords.reshape(-1, 2) * 1000 
                return points
        return None
    except: return None

# --- 2. ฟังก์ชันคำนวณ PCA (คงเดิม) ---
def calculate_pca(points):
    mean_vec = np.mean(points, axis=0)
    centered_points = points - mean_vec
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return mean_vec, eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

# --- 3. ฟังก์ชันหมุนภาพ (คงเดิม) ---
def rotate_points(points, angle_degrees, center):
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    return (points - center).dot(rotation_matrix) + center

# ================= เริ่มทำงาน =================
target_class = 24 
filename = '5.jpg.txt'
points_A = get_class_data_from_file(filename, target_class)

if points_A is not None:
    mean_A, vals_A, vecs_A = calculate_pca(points_A)
    points_B = rotate_points(points_A, 30, mean_A)
    mean_B, vals_B, vecs_B = calculate_pca(points_B)
    cosine_sim = abs(np.dot(vecs_A[:, 0], vecs_B[:, 0]))

    # ================= วาดกราฟ =================
    # ปรับขนาดรูปให้สูงขึ้น (figsize=(20, 9)) เพื่อให้มีที่เขียนคำอธิบายด้านล่าง
    fig, axes = plt.subplots(1, 3, figsize=(20, 9))
    
    # ปรับระยะห่าง: bottom=0.25 (เว้นที่ด้านล่าง 25%), wspace=0.3 (เว้นระยะระหว่างรูป)
    plt.subplots_adjust(bottom=0.3, wspace=0.3) 

    for ax in axes: 
        ax.set_aspect('equal')
        ax.invert_yaxis() 

    # --- Graph 1: Vectors from Center ---
    ax1 = axes[0]
    ax1.set_title(f"Class {target_class}: Vectors (Showing 80% of points)")
    
    origin = mean_A 
    ax1.plot(origin[0], origin[1], 'ko', markersize=5, label='Centroid (จุดศูนย์กลาง)')
    
    count = 0
    for i in range(len(points_A)):
        if i % 5 == 0: continue # Skip 20%
        p = points_A[i]
        ax1.annotate("", xy=(p[0], p[1]), xytext=(origin[0], origin[1]),
                    arrowprops=dict(arrowstyle="-", color="purple", alpha=0.3, linewidth=0.5))
        ax1.plot(p[0], p[1], '.', color='purple', markersize=5)
        count += 1
    
    # Label หลอกๆ เพื่อให้ Legend แสดงว่าลูกศรคืออะไร
    ax1.plot([], [], '-', color='purple', alpha=0.5, label='Vectors (ลูกศร)')

    poly = Polygon(points_A, closed=True, fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
    ax1.add_patch(poly)
    
    # Legend 1: ย้ายลงมาข้างล่างกราฟ
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
    
    # Description 1: เขียนคำอธิบายใต้ Legend
    desc1 = (f"แสดงให้เห็นว่า Mask 1 อัน ประกอบขึ้นจาก\n"
             f"Vector จำนวนมาก ({count} เส้น)\n"
             f"ที่พุ่งออกจากจุดศูนย์กลางไปยังขอบฟัน")
    ax1.text(0.5, -0.35, desc1, transform=ax1.transAxes, ha='center', va='top', fontsize=11, color='#333333')


    # --- Graph 2: PCA Axes ---
    ax2 = axes[1]
    ax2.set_title(f"PCA: Principal Directions")
    poly2 = Polygon(points_A, closed=True, facecolor='violet', alpha=0.3)
    ax2.add_patch(poly2)
    ax2.scatter(points_A[:, 0], points_A[:, 1], c='purple', s=2, alpha=0.3)
    
    scale = np.sqrt(vals_A[0]) * 2.5
    scale2 = np.sqrt(vals_A[1]) * 2.5
    
    ax2.arrow(mean_A[0], mean_A[1], vecs_A[0,0]*scale, vecs_A[1,0]*scale, 
              width=scale*0.05, color='red', zorder=5, label='Eigenvector 1 (แกนหลัก)')
    ax2.arrow(mean_A[0], mean_A[1], vecs_A[0,1]*scale2, vecs_A[1,1]*scale2, 
              width=scale*0.02, color='green', zorder=5, label='Eigenvector 2 (แกนรอง)')
    ax2.plot(mean_A[0], mean_A[1], 'ko', markersize=5)
    
    # Legend 2
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
    
    # Description 2
    desc2 = ("PCA ช่วยคำนวณหา 'แกนหลัก' (สีแดง)\n"
             "ซึ่งชี้ไปในทิศทางที่ข้อมูลมีความยาวมากที่สุด\n"
             "ใช้เป็นตัวแทน 'ทิศทางของฟัน' ได้")
    ax2.text(0.5, -0.35, desc2, transform=ax2.transAxes, ha='center', va='top', fontsize=11, color='#333333')

    # --- Graph 3: Comparison ---
    ax3 = axes[2]
    ax3.set_title(f"Comparison Result")
    poly_A = Polygon(points_A, closed=True, facecolor='gray', alpha=0.3, label='Original Mask (ฟันต้นแบบ)')
    ax3.add_patch(poly_A)
    ax3.plot([mean_A[0], mean_A[0]+vecs_A[0,0]*scale], [mean_A[1], mean_A[1]+vecs_A[1,0]*scale], 'r--', linewidth=2)
             
    poly_B = Polygon(points_B, closed=True, facecolor='orange', alpha=0.5, label='Input Mask (ฟันที่ทำนายได้)')
    ax3.add_patch(poly_B)
    v1_B = vecs_B[:, 0] * scale
    ax3.plot([mean_B[0], mean_B[0]+v1_B[0]], [mean_B[1], mean_B[1]+v1_B[1]], 'r-', linewidth=3, label='Direction (ทิศทาง)')
    
    # Legend 3
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
    
    # Description 3
    desc3 = (f"เปรียบเทียบทิศทางของฟันทั้ง 2 ซี่\n"
             f"ด้วยค่า Cosine Similarity\n"
             f"Similarity = {cosine_sim:.4f} (ยิ่งใกล้ยิ่งเหมือน)")
    ax3.text(0.5, -0.35, desc3, transform=ax3.transAxes, ha='center', va='top', fontsize=11, color='#333333')
    
    plt.savefig('pca_final_explained.png', dpi=300, bbox_inches='tight') # เซฟแบบไม่ตกขอบ
    plt.show()
else:
    print("Error: Data not found")