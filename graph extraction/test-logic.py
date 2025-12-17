import numpy as np
from neo4j import GraphDatabase
import random
import copy

# =========================================================
# ⚙️ Configuration
# =========================================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678" # <--- แก้รหัสผ่านของคุณตรงนี้

# =========================================================
# 1. CLASS: Knowledge Base (สมองที่ดึงความรู้จาก Graph)
# =========================================================
class DentalKnowledgeBase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # --- Move 1 Support: หาเพื่อนบ้าน / เติม Gap ---
    def find_missing_teeth_between(self, fdi_start, fdi_end):
        # ถาม Graph จาก Instance จริง: (Start)-[:NEXT_TO]-(Middle)-[:NEXT_TO]-(End)
        query = """
        MATCH (start:Tooth {fdi: $fdi_a})-[:NEXT_TO]-(middle:Tooth)-[:NEXT_TO]-(end:Tooth {fdi: $fdi_b})
        MATCH (case:Case)-[:HAS]->(start)
        MATCH (case)-[:HAS]->(middle)
        MATCH (case)-[:HAS]->(end)
        
        RETURN middle.fdi as suggested_fdi, count(distinct case) as frequency
        ORDER BY frequency DESC
        LIMIT 1
        """
        with self.driver.session() as session:
            # ต้องแปลงเป็น str() ก่อนส่งเข้า DB เพราะเราเก็บ fdi เป็น string
            result = session.run(query, fdi_a=str(fdi_start), fdi_b=str(fdi_end))
            record = result.single()
            
            if record:
                print(f"      📊 Stats: Found pattern {fdi_start}-{record['suggested_fdi']}-{fdi_end} in {record['frequency']} cases.")
                return [int(record["suggested_fdi"])]
            else:
                return []

    def suggest_correct_neighbor(self, fdi_prev):
        # ถาม Graph: ต่อจาก prev ปกติคือเลขอะไร?
        query = """
        MATCH (prev:Tooth {fdi: $fdi_prev})-[:NEXT_TO]-(next:Tooth)
        WHERE next.fdi > prev.fdi
        RETURN next.fdi as suggested_fdi, count(*) as freq
        ORDER BY freq DESC
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, fdi_prev=str(fdi_prev))
            record = result.single()
            if record:
                print(f"      📊 Stats: Neighbor {fdi_prev}->{record['suggested_fdi']} found {record['freq']} times.")
                return int(record["suggested_fdi"])
            return None

    # --- Move 2 Support: สถิติตำแหน่ง (Spatial) ---
    def get_spatial_stats(self, fdi):
        # ถาม Graph: ปกติซี่นี้ (fdi) อยู่สูงต่ำแค่ไหน (pos_y)?
        query = """
        MATCH (t:Tooth {fdi: $fdi})
        RETURN avg(t.pos_y) as mean_y, stdev(t.pos_y) as std_y, count(*) as n
        """
        with self.driver.session() as session:
            result = session.run(query, fdi=str(fdi))
            record = result.single()
            if record and record['mean_y'] is not None:
                std_y = record['std_y']
                # ถ้าข้อมูลน้อยเกินไปจน std=0 ให้ใช้ค่า default กัน Error หารศูนย์
                if std_y is None or std_y == 0: 
                    std_y = 0.05 
                return {'mean_y': record['mean_y'], 'std_y': std_y}
            return None

    # --- Move 3 Support: ความน่าจะเป็นของ Path (Ranking) ---
    def validate_triplet_probability(self, fdi_1, fdi_2, fdi_3):
        # ถาม Graph: ลำดับ A -> B -> C นี้เคยเกิดขึ้นกี่ครั้ง?
        query = """
        MATCH (t1:Tooth {fdi: $f1})-[:NEXT_TO]-(t2:Tooth {fdi: $f2})-[:NEXT_TO]-(t3:Tooth {fdi: $f3})
        MATCH (c:Case)-[:HAS]->(t1), (c)-[:HAS]->(t2), (c)-[:HAS]->(t3)
        RETURN count(distinct c) as freq
        """
        with self.driver.session() as session:
            result = session.run(query, f1=str(fdi_1), f2=str(fdi_2), f3=str(fdi_3))
            record = result.single()
            return record['freq'] if record else 0

# =========================================================
# 2. CLASS: Simulator (ตัวจำลองข้อมูลผิด)
# =========================================================
class PredictionSimulator:
    def __init__(self, ground_truth):
        self.ground_truth = sorted(ground_truth, key=lambda x: x['pos_x'])

    def simulate_missing_tooth(self, target_fdi):
        """ ลบฟันออก 1 ซี่ (False Negative) """
        print(f"🎭 SIMULATION: Deleting tooth {target_fdi}...")
        corrupted = [t for t in self.ground_truth if t['fdi'] != target_fdi]
        return corrupted

    def simulate_spatial_error(self, target_fdi, wrong_y):
        """ เปลี่ยนตำแหน่งฟันให้ผิดปกติ (Spatial Error) """
        print(f"🎭 SIMULATION: Moving tooth {target_fdi} to y={wrong_y}...")
        corrupted = copy.deepcopy(self.ground_truth)
        for t in corrupted:
            if t['fdi'] == target_fdi:
                t['pos_y'] = wrong_y
        return sorted(corrupted, key=lambda x: x['pos_x'])
    
    def simulate_bad_path(self, index, wrong_fdi):
        """ เปลี่ยนเลขฟันให้ลำดับมั่ว (Path Error) """
        print(f"🎭 SIMULATION: Changing index {index} to {wrong_fdi}...")
        corrupted = copy.deepcopy(self.ground_truth)
        corrupted[index]['fdi'] = wrong_fdi
        return sorted(corrupted, key=lambda x: x['pos_x'])

# =========================================================
# 3. CLASS: GraphCorrector (ระบบตรวจสอบ 3 ท่า)
# =========================================================
class GraphCorrector:
    def __init__(self, kb):
        self.kb = kb

    # --- Move 1: Neighbor Check (เติมฟัน/แก้เลขซ้ำ) ---
    def move1_gap_filling_and_labeling(self, teeth_list):
        print("\n🏃 MOVE 1: Neighbor Check (Gap Filling & Labeling)...")
        # 1.1 แก้เลขซ้ำ (Duplicate)
        sorted_teeth = sorted(teeth_list, key=lambda x: x['pos_x'])
        for i in range(len(sorted_teeth) - 1):
            if sorted_teeth[i]['fdi'] == sorted_teeth[i+1]['fdi']:
                print(f"   ⚠️ Detect Duplicate: {sorted_teeth[i]['fdi']}")
                if i > 0:
                    prev = sorted_teeth[i-1]['fdi']
                    suggestion = self.kb.suggest_correct_neighbor(prev)
                    if suggestion and suggestion != sorted_teeth[i]['fdi']:
                        print(f"   ✅ Action: Corrected to {suggestion}")
                        sorted_teeth[i]['fdi'] = suggestion
                        sorted_teeth[i]['status'] = 'CORRECTED'

        # 1.2 เติม Gap
        corrected_list = []
        for i in range(len(sorted_teeth) - 1):
            curr_t = sorted_teeth[i]
            next_t = sorted_teeth[i+1]
            corrected_list.append(curr_t)

            fdi_diff = abs(curr_t['fdi'] - next_t['fdi'])
            same_quadrant = (curr_t['fdi'] // 10 == next_t['fdi'] // 10)
            
            if fdi_diff > 1 and same_quadrant:
                print(f"   ⚠️ Detect Gap: {curr_t['fdi']} -> {next_t['fdi']}")
                missing = self.kb.find_missing_teeth_between(curr_t['fdi'], next_t['fdi'])
                if missing:
                    for m_fdi in missing:
                        # สร้าง Virtual Node
                        mid_x = (curr_t['pos_x'] + next_t['pos_x']) / 2
                        mid_y = (curr_t['pos_y'] + next_t['pos_y']) / 2
                        virtual_tooth = {'fdi': m_fdi, 'pos_x': mid_x, 'pos_y': mid_y, 'status': 'VIRTUAL'}
                        corrected_list.append(virtual_tooth)
                        print(f"   ✅ Action: Inserted {m_fdi}")
                else:
                    print("   🧠 KB: Normal Gap (No pattern found)")
        
        corrected_list.append(sorted_teeth[-1])
        return corrected_list

    # --- Move 2: Spatial Constraint Check ---
    def move2_spatial_constraints(self, teeth_list):
        print("\n🏃 MOVE 2: Spatial Constraint Check (Z-Score)...")
        checked_list = []
        for t in teeth_list:
            stats = self.kb.get_spatial_stats(t['fdi'])
            is_valid = True
            
            if stats:
                # คำนวณ Z-score
                z_score = abs(t['pos_y'] - stats['mean_y']) / stats['std_y']
                
                # ถ้าหลุด 3 SD ถือว่าผิดปกติมาก
                if z_score > 3.0:
                    print(f"   ⚠️ Spatial Alert: Tooth {t['fdi']} at y={t['pos_y']:.2f} (Avg={stats['mean_y']:.2f}, Z={z_score:.2f})")
                    t['status'] = 'OUTLIER'
                    # ในระบบจริงอาจจะลบทิ้ง หรือ Flag ไว้ให้หมอดู
                    is_valid = False 
            
            if not is_valid:
                print(f"   ❌ Action: Flagged {t['fdi']} as Spatial Outlier")
            
            checked_list.append(t)
        return checked_list

    # --- Move 3: Path Ranking ---
    def move3_path_ranking(self, teeth_list):
        print("\n🏃 MOVE 3: Path Ranking (Triplet Consistency)...")
        # เช็คทีละ 3 ตัวเรียงกัน
        for i in range(len(teeth_list) - 2):
            t1, t2, t3 = teeth_list[i], teeth_list[i+1], teeth_list[i+2]
            
            freq = self.kb.validate_triplet_probability(t1['fdi'], t2['fdi'], t3['fdi'])
            print(f"   🔍 Checking Path [{t1['fdi']}->{t2['fdi']}->{t3['fdi']}] : Found {freq} times")
            
            if freq == 0:
                print(f"   ⚠️ Low Probability Path!")
                # ลองถาม KB ว่าถ้าไม่ใช่ t2 ควรเป็นอะไร?
                suggestion = self.kb.find_missing_teeth_between(t1['fdi'], t3['fdi'])
                if suggestion:
                    print(f"   💡 Suggestion: Graph suggests {suggestion[0]} instead of {t2['fdi']}")
                    # Action: แก้ไขค่า (ถ้ามั่นใจ)
                    # t2['fdi'] = suggestion[0] 
        
        return teeth_list

# =========================================================
# 4. MAIN EXECUTION (The Pipeline)
# =========================================================
def main():
    # 0. Mock Data (จำลอง Ground Truth)
    # สมมติว่าเป็นฟันบน (y ~ 0.2)
    ground_truth_case = [
        {'fdi': 11, 'pos_x': 0.63, 'pos_y': 0.36}, 
        {'fdi': 12, 'pos_x': 0.67, 'pos_y': 0.35},
        {'fdi': 13, 'pos_x': 0.71, 'pos_y': 0.35},
        {'fdi': 14, 'pos_x': 0.75, 'pos_y': 0.52}
    ]

    kb = DentalKnowledgeBase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    simulator = PredictionSimulator(ground_truth_case)
    corrector = GraphCorrector(kb)

    print("="*60)
    print("🦷 DENTAL AI: NEURO-SYMBOLIC REASONING ENGINE (3 MOVES)")
    print("="*60)

    # --- SCENARIO A: ฟันหาย (Test Move 1) ---
    print("\n--- [CASE A] Missing Tooth ---")
    input_a = simulator.simulate_missing_tooth(12)
    print(f"Input: {[t['fdi'] for t in input_a]}")
    
    step1_a = corrector.move1_gap_filling_and_labeling(input_a)
    print(f"Result A: {[t['fdi'] for t in step1_a]}")


    # --- SCENARIO B: ตำแหน่งผิดปกติ (Test Move 2) ---
    print("\n--- [CASE B] Spatial Outlier ---")
    # แกล้งย้าย 12 ไปอยู่ข้างล่าง (y=0.8) ซึ่งผิดปกติสำหรับฟันเบอร์ 12
    input_b = simulator.simulate_spatial_error(12, 0.8)
    print(f"Input: {[t['fdi'] for t in input_b]} with y={[t['pos_y'] for t in input_b]}")
    
    step1_b = corrector.move1_gap_filling_and_labeling(input_b)
    step2_b = corrector.move2_spatial_constraints(input_b)


    # --- SCENARIO C: ลำดับผิดปกติ (Test Move 3) ---
    print("\n--- [CASE C] Bad Path Sequence ---")
    # เปลี่ยน 12 เป็น 15 (ลำดับกลายเป็น 11->15->13->14 ซึ่งเป็นไปไม่ได้)
    input_c = simulator.simulate_bad_path(1, 15)
    # ต้อง sort ใหม่ตาม x เพราะ simulator แค่แก้ค่า
    input_c = sorted(input_c, key=lambda x: x['pos_x']) 
    print(f"Input: {[t['fdi'] for t in input_c]}")

    step1_c = corrector.move1_gap_filling_and_labeling(input_c) # Move 1 ผ่านเพราะไม่มี Gap ใหญ่
    step2_c = corrector.move2_spatial_constraints(input_c)      # Move 2 ผ่านเพราะ y ปกติ
    step3_c = corrector.move3_path_ranking(input_c)             # Move 3 ควรจับผิดได้

    kb.close()

if __name__ == "__main__":
    main()