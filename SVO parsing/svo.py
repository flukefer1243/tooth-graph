import pdfplumber
import spacy
import re
import json
from neo4j import GraphDatabase

# --- 1. CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7689" # แก้เป็น URI ของคุณ
NEO4J_USER = "neo4j"                # แก้เป็น Username ของคุณ
NEO4J_PASSWORD = "12345678"         # แก้เป็น Password ของคุณ
PDF_FILENAME = "ohse1.pdf" # ชื่อไฟล์ PDF ของคุณ

print("⏳ Loading NLP Model...")
try:
    nlp = spacy.load("en_core_web_lg")
except:
    print("❌ Model not found. Please run: python -m spacy download en_core_web_lg")
    exit()

# ==========================================
# PART 1: INTELLIGENT EXTRACTOR
# ==========================================
class PDFExtractor:
    def __init__(self):
        self.results = []
        # คำสรรพนามที่ "ห้าม" เป็น Head node (เพราะไม่รู้ว่าคือใคร)
        self.BANNED_HEADS = {"it", "this", "that", "which", "who", "he", "she", "they", "you", "we", "i", "what"}

    # --- A. Cleaning Function ---
    def clean_text_advanced(self, text):
        if not text: return ""
        # ลบคำว่า A R C H I V E D ที่หัวกระดาษ
        text = re.sub(r'a\s?r\s?c\s?h\s?i\s?v\s?e\s?d', '', text, flags=re.IGNORECASE)
        # ลบเลขหน้าและขีดกั้น (เช่น "| 11", "8 |")
        text = re.sub(r'\|\s?\d+', '', text)
        text = re.sub(r'\d+\s?\|', '', text)
        # ลบ Bullet points
        text = re.sub(r'[•●▪-]', '', text)
        # ลบ Newline แล้วเชื่อมด้วย Space
        text = text.replace('\n', ' ')
        # ลบ Space ซ้ำๆ
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # --- B. SVO Logic (Smart Filtering) ---
    def extract_svo_from_text(self, text, page_num):
        cleaned_text = self.clean_text_advanced(text)
        doc = nlp(cleaned_text)
        triples = []
        
        def get_full_phrase(token):
            # ดึงมาทั้งยวง แต่กรองเอาเฉพาะตัวหนังสือ
            phrase = "".join([t.text_with_ws for t in token.subtree]).strip()
            return re.sub(r'[()",;]', '', phrase).strip()

        for token in doc:
            if token.pos_ == "VERB":
                subj_phrase = None
                obj_phrase = None
                
                # 1. Subject Logic
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        # กรอง Pronoun ทิ้ง
                        if child.lemma_.lower() in self.BANNED_HEADS: continue
                        if child.pos_ == "PRON": continue
                        subj_phrase = get_full_phrase(child)
                
                # 2. Object Logic
                for child in token.children:
                    if child.dep_ in ("dobj", "attr", "acomp"):
                        obj_phrase = get_full_phrase(child)
                    elif child.dep_ == "prep":
                        for grand_child in child.children:
                            if grand_child.dep_ == "pobj":
                                obj_phrase = child.text + " " + get_full_phrase(grand_child)

                # 3. Validation & Construction
                if subj_phrase and obj_phrase:
                    # กรองคำที่สั้นเกินไป
                    if len(subj_phrase) < 3 or len(obj_phrase) < 2: continue

                    # ใช้รากศัพท์ (Lemma) เป็นชื่อ Relation (แก้ 'RE -> BE)
                    relation = token.lemma_.upper()
                    
                    # กรอง Relation ที่เป็นสัญลักษณ์ขยะ
                    if not re.match(r'^[A-Z]+$', relation): continue

                    triples.append({
                        "head": subj_phrase,
                        "relation": relation,
                        "tail": obj_phrase,
                        "meta": {
                            "source": "text",
                            "page": page_num,
                            "context": token.sent.text.strip()[:200]
                        }
                    })
        return triples

    # --- C. Table Logic (Fixed) ---
    def extract_triples_from_table(self, table, page_num):
        triples = []
        if not table or len(table) < 2: return []

        headers = table[0]
        data_rows = table[1:]
        
        for row in data_rows:
            if not row or row[0] is None: continue
            
            subject = str(row[0]).replace('\n', ' ').strip()
            if not subject: continue

            for i in range(1, len(row)):
                if i < len(headers) and row[i]:
                    header_val = headers[i]
                    if not header_val or str(header_val).strip() == "": continue
                    
                    # Clean Header Name
                    raw_header = str(header_val).replace('\n', '_').strip()
                    clean_header = re.sub(r'[^a-zA-Z0-9_]', '', raw_header)
                    relation = f"HAS_{clean_header.upper()}" 

                    obj = str(row[i]).replace('\n', ' ').strip()
                    if obj:
                        triples.append({
                            "head": subject,
                            "relation": relation,
                            "tail": obj,
                            "meta": {
                                "source": "table",
                                "page": page_num
                            }
                        })
        return triples

    # --- D. Main Processor ---
    def process_pdf(self, pdf_path):
        print(f"📂 Reading PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # 1. Tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        self.results.extend(self.extract_triples_from_table(table, page_num))

                # 2. Text (ส่ง raw text ไปให้ clean_text_advanced จัดการ)
                raw_text = page.extract_text()
                if raw_text:
                    self.results.extend(self.extract_svo_from_text(raw_text, page_num))

        return self.results

# ==========================================
# PART 2: NEO4J IMPORTER
# ==========================================
class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_data(self, data_list):
        print(f"\n🚀 Connecting to Neo4j... (Importing {len(data_list)} relationships)")
        
        with self.driver.session() as session:
            count = 0
            for item in data_list:
                # 1. Prepare the clean_rel variable
                clean_rel = re.sub(r'[^A-Z0-9_]', '_', item['relation']).strip('_')
                if not clean_rel: clean_rel = "RELATED_TO"

                meta = item.get('meta', {})
                context = meta.get('context', 'Table Data')
                page = meta.get('page', 0)
                source = meta.get('source', 'unknown')

                # 2. THE UPDATED QUERY
                # Note the use of {{ }} for Cypher properties
                # We use $action_rel as a placeholder for the variable
                query = f"""
                MERGE (h:Entity {{name: $head}})
                MERGE (t:Entity {{name: $tail}})
                MERGE (h)-[r:RELATED_TO {{action: $action_rel}}]->(t)
                SET r.source = $source,
                    r.page = $page,
                    r.context = $context
                """
                
                try:
                    # 3. PASS THE VARIABLE HERE
                    session.run(query, 
                                head=item['head'], 
                                tail=item['tail'],
                                action_rel=clean_rel,  # <--- Helper variable connects to $action_rel
                                source=source,
                                page=page,
                                context=context)
                    count += 1
                except Exception as e:
                    print(f"⚠️ Error importing item: {item['head']} -> {e}")
                    
        print(f"✅ Import Finished! Successfully saved {count} relationships.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Extract Data
    extractor = PDFExtractor()
    try:
        final_data = extractor.process_pdf(PDF_FILENAME)
        print(f"✅ Extraction Complete! Found {len(final_data)} quality triples.")
        
        # 2. Save JSON
        json_filename = "cleaned_knowledge_graph.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved JSON to '{json_filename}'")

        # 3. Import to Neo4j
        if len(final_data) > 0:
            importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            importer.import_data(final_data)
            importer.close()
        else:
            print("⚠️ No data found to import.")
            
    except Exception as e:
        print(f"❌ Error: {e}")