from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import json

# --- 1. 설정 ---
# 모델과 Elasticsearch 클라이언트를 준비합니다.
model = SentenceTransformer("nlpai-lab/KURE-v1")
es = Elasticsearch("http://localhost:9200")
index_name = "survey_responses"

# --- 2. 검색어 벡터 생성 ---
# 테스트하고 싶은 자연어 검색어를 입력합니다.
search_sentence = "음주경험이 있는 갤럭시 보유자"
print(f"🔍 검색어: '{search_sentence}'")

# 검색어를 쿼리 벡터로 변환합니다.
query_vector = model.encode(search_sentence).tolist()

# --- 3. 하이브리드 검색 쿼리 구성 ---
# ✨ [수정] knn 쿼리 내부에 filter 절을 추가합니다.
knn_query = {
    "field": "embedding_vector",
    "query_vector": query_vector,
    "k": 10,
    "num_candidates": 100,
    "filter": {
        "bool": {
            "should": [
                { "wildcard": { "q_text.keyword": "*갤럭시*" } }
            ]
        }
    }
}

# 검색을 실행하고, 필요한 필드만 가져옵니다.
response = es.search(
    index=index_name,
    knn=knn_query,
    source=["user_id", "q_text", "answer"]
)

# --- 4. 결과 출력 ---
print("\n--- ✨ 하이브리드 검색 결과 ---")
for hit in response['hits']['hits']:
    score = hit['_score']
    source = hit['_source']
    print(f"- 유사도 점수: {score:.4f}")
    print(f"  내용: {source.get('q_text')} -> {source.get('answer')}")
    print(f"  (사용자 ID: {source.get('user_id')})\n")