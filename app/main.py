import pandas as pd
from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import datetime
from sentence_transformers import SentenceTransformer
import os
import re
import traceback
import json

# --- 모델 및 클라이언트 초기화 ---
embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
VECTOR_DIMENSIONS = 1024

# --- FastAPI 및 Elasticsearch 초기화 ---
app = FastAPI()
es = Elasticsearch("http://localhost:9200")


def create_index_if_not_exists(index_name: str):
    """
    Elasticsearch에 특정 인덱스가 존재하지 않으면, '사용자' 단위의 nested 구조가 적용된 매핑으로 새 인덱스를 생성합니다.
    """
    if not es.indices.exists(index=index_name):
        print(f"✨ '{index_name}' 인덱스가 없어 새로 생성합니다.")

        # --- 매핑 구조 변경 ---
        # 최상위 레벨에는 사용자 정보만 두고, 모든 질문/응답 관련 정보는 nested 필드 안으로 이동합니다.
        mappings = {
            "properties": {
                "user_id": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "qa_pairs": {  # Nested 필드 이름 변경 (Question-Answer pairs)
                    "type": "nested",
                    "properties": {
                        "q_code": {"type": "keyword"},
                        "q_text": {"type": "text", "analyzer": "nori"},
                        "q_type": {"type": "keyword"},
                        "answer_text": {"type": "text", "analyzer": "nori"},
                        "embedding_text": {
                            "type": "text",
                            "analyzer": "nori",
                            "index": False,
                        },
                        "answer_vector": {
                            "type": "dense_vector",
                            "dims": VECTOR_DIMENSIONS,
                        },
                    },
                },
            }
        }
        try:
            es.indices.create(index=index_name, mappings=mappings)
            print(f"👍 '{index_name}' 인덱스 생성 완료 (사용자 단위 Nested 구조 적용).")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"'{index_name}' 인덱스 생성 실패: {e}"
            )

def parse_question_metadata(file_path: str) -> dict:
    metadata = {}
    current_q_code = None
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([a-zA-Z0-9_]+),([^,]+),([^,]+)$", line)
            if match:
                q_code, q_text, q_type = match.groups()
                current_q_code = q_code.strip()
                metadata[current_q_code] = {
                    "text": q_text.strip(),
                    "type": q_type.strip(),
                    "options": {},
                }
            elif current_q_code and re.match(r"^\d+,", line):
                parts = line.split(",", 2)
                option_code = parts[0].strip()
                option_text = parts[1].strip()
                if option_code:
                    metadata[current_q_code]["options"][option_code] = option_text
    return metadata


@app.get("/")
def read_root():
    return {"message": "설문 데이터 색인 API가 Elasticsearch와 함께 실행 중입니다!"}


@app.post("/index-survey-data") 
def index_survey_data_by_user():
    question_file = "./data/question_list.csv"
    response_file = "./data/response_list_300.csv"
    index_name = "survey_responses" # 새로운 데이터 구조를 위한 새 인덱스

    try:
        if not es.ping():
            raise HTTPException(
                status_code=503, detail="Elasticsearch 서버에 연결할 수 없습니다."
            )

        print("\n--- 🚀 데이터 색인 작업을 시작합니다 (사용자 단위 Nested 구조) ---")
        print(f"- 대상 인덱스: {index_name}\n")

        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"🗑️  기존 '{index_name}' 인덱스를 삭제했습니다.")
        create_index_if_not_exists(index_name)

        questions_meta = parse_question_metadata(question_file)
        print("✅ 질문 메타데이터 파싱 완료.")

        df_responses = pd.read_csv(response_file, encoding="utf-8-sig")
        df_responses = df_responses.astype(object).where(pd.notnull(df_responses), None)
        print(f"✅ 응답 데이터 로드 완료. (총 {len(df_responses)}명)")

        actions = []
        user_count = 0
        total_users = len(df_responses)

        # 사용자 한 명당 하나의 문서를 생성
        for _, row in df_responses.iterrows():
            user_count += 1
            if user_count % 50 == 0 or user_count == 1 or user_count == total_users:
                print(f"🔄 사용자 데이터 처리 중... ({user_count}/{total_users})")

            user_id = row.get("mb_sn")
            if not user_id:
                continue

            # 해당 사용자의 모든 질문-응답 쌍을 담을 리스트
            all_qa_pairs_for_user = []

            # 사용자의 각 응답(컬럼)을 순회
            for q_code, raw_answer in row.items():
                if q_code == "mb_sn" or raw_answer is None:
                    continue
                q_info = questions_meta.get(q_code)
                if not q_info:
                    continue

                q_text, q_type = q_info["text"], q_info["type"]
                answers_text_list = []

                if q_type == "MULTI":
                    answer_codes = str(raw_answer).split(",")
                    for code in answer_codes:
                        code = code.strip()
                        if code:
                            answers_text_list.append(
                                q_info["options"].get(code, f"알 수 없는 코드: {code}")
                            )
                elif q_type == "SINGLE":
                    answers_text_list.append(
                        q_info["options"].get(str(raw_answer).strip(), raw_answer)
                    )
                else:
                    answers_text_list.append(str(raw_answer))

                # 각 답변 텍스트를 nested 객체로 변환
                for answer_text in answers_text_list:
                    if answer_text is None or str(answer_text).strip() == "":
                        continue
                    
                    embedding_text = f"{q_text} 문항에 '{answer_text}'라고 응답"
                    vector = embedding_model.encode(embedding_text).tolist()

                    # Nested 객체 생성
                    qa_pair_doc = {
                        "q_code": q_code,
                        "q_text": q_text,
                        "q_type": q_type,
                        "answer_text": answer_text,
                        "embedding_text": embedding_text,
                        "answer_vector": vector,
                    }
                    all_qa_pairs_for_user.append(qa_pair_doc)

            # 처리된 질문-응답 쌍이 있을 경우에만 최종 사용자 문서를 생성
            if all_qa_pairs_for_user:
                final_user_document = {
                    "user_id": user_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "qa_pairs": all_qa_pairs_for_user,
                }
                actions.append(
                    {"_index": index_name, "_id": user_id, "_source": final_user_document}
                )

        if not actions:
            print("⚠️  처리할 문서가 없습니다. 작업을 중단합니다.")
            return {"message": "처리할 데이터가 없습니다."}

        print(
            f"\n✅ 총 {len(actions)}개의 문서를 생성했습니다. (사용자 단위로 그룹화)"
        )
        print("--- 📄 첫 번째 사용자 문서 샘플 ---")
        print(json.dumps(actions[0]["_source"], indent=2, ensure_ascii=False))
        print("--------------------------------\n")

        print("⏳ Elasticsearch에 데이터 대량 삽입(bulk)을 시작합니다...")
        success, failed = bulk(es, actions, raise_on_error=False, refresh=True)

        print(f"🎉 작업 완료! 성공: {success}, 실패: {len(failed)}")

        return {
            "message": "사용자 단위 설문 데이터 색인 작업이 완료되었습니다.",
            "성공": success,
            "실패": len(failed),
        }

    except Exception as e:
        print("💥💥💥 예상치 못한 오류 발생! 💥💥💥")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        )