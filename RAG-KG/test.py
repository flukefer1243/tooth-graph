from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase # [NEW] ต้อง import ตัวนี้เพิ่ม

import openai
import time
import json

# --- 1. เตรียมข้อมูล PDF ---
file_path = "ohse1.pdf"
loader = PDFPlumberLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(docs)
print(f"จำนวน Chunks ทั้งหมด: {len(chunks)}\n")

# --- 2. ตั้งค่าการเชื่อมต่อ (Setup) ---

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# [NEW] สร้าง Driver เตรียมไว้ (ทำแค่นอกลูปพอ เพื่อประหยัด Resource)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

client = openai.OpenAI(
    api_key="sk-gsFUgxu77dZO94CoOjq3WESR74U9ETGz8gTtdK8tMH60dyq4", # อย่าลืมใส่ API Key ของคุณ
    base_url="https://api.opentyphoon.ai/v1"
)

# [NEW] ฟังก์ชันสำหรับบันทึกข้อมูลลง Neo4j (ใช้ Cypher Query)
def save_to_neo4j(tx, data_list):
    # Query นี้ใช้ UNWIND เพื่อรับ List เข้าไปทีเดียว (เร็วกว่าวนลูป insert ทีละบรรทัด)
    query = """
    UNWIND $batch AS row
    WITH row WHERE row.Entity2 IS NOT NULL  // กรองข้อมูลที่ไม่มีปลายทางทิ้ง
    MERGE (e1:Entity {name: row.Entity1})
    MERGE (e2:Entity {name: row.Entity2})
    MERGE (e1)-[r:RELATED_TO {action: row.Relationship}]->(e2)
    """
    tx.run(query, batch=data_list)

# --- 3. เริ่มวนลูปประมวลผล ---

try: # [NEW] ครอบ try-finally เพื่อให้มั่นใจว่า driver จะถูกปิดเสมอเมื่อจบงาน
    for i, chunk in enumerate(chunks):
        print(f"--- Processing Chunk {i+1}/{len(chunks)} ---")
        
        try:
            response = client.chat.completions.create(
                model="typhoon-v2.1-12b-instruct",
                messages=[
                    {"role": "system", "content": "ช่วยวิเคราะห์ข้อความต่อไปนี้ แล้วสกัดออกมาเป็นรูปแบบ (Subject, Predicate, Object) หรือ (Entity1, Relationship, Entity2) ในรูปแบบ JSON Array เท่านั้น ไม่ต้องมีคำอธิบายอื่น"},
                    {"role": "user", "content": chunk.page_content}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            typhoon_response_string = response.choices[0].message.content
            print(typhoon_response_string) # debug ดูดิบๆ

            # 1. Cleaning
            cleaned_string = typhoon_response_string.replace("```json", "").replace("```", "").strip()

            # 2. Parsing to Python List
            data_list = json.loads(cleaned_string)
            print(data_list) # debug ดูดิบๆ
            
            # ตรวจสอบเบื้องต้นว่าได้ List จริงไหม
            if isinstance(data_list, list) and len(data_list) > 0:
                print(f"สกัดได้ {len(data_list)} ความสัมพันธ์ -> กำลังบันทึกลง Neo4j...")
                
                # [NEW] 3. เรียกใช้ Driver เพื่อบันทึกข้อมูล
                with driver.session() as session:
                    session.execute_write(save_to_neo4j, data_list)
                print("✅ บันทึกสำเร็จ!")
            else:
                print("⚠️ ผลลัพธ์ว่างเปล่า หรือรูปแบบไม่ถูกต้อง")

        except json.JSONDecodeError as e:
            print(f"❌ JSON Error: แปลงข้อมูลไม่ได้ ข้าม Chunk นี้ไป")
        except Exception as e:
            print(f"❌ Error อื่นๆ: {e}")

        # พักหายใจ (ป้องกัน Rate Limit)
        time.sleep(5) 
        print("--------------------------------------------------\n")

finally:
    # [NEW] ปิดการเชื่อมต่อเมื่อจบการทำงานทั้งหมด
    driver.close()
    print("ปิดการเชื่อมต่อ Neo4j เรียบร้อย")