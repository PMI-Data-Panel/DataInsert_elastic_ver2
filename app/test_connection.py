from elasticsearch import Elasticsearch, ConnectionError

# Docker Compose 설정에 따라 인증 정보 없이 연결합니다.
ES_HOST = "http://localhost:9200"

print(f"'{ES_HOST}'로 연결을 시도합니다...")

try:
    # 클라이언트 생성
    es_client = Elasticsearch(hosts=[ES_HOST])

    # ping() 메소드로 서버가 살아있는지 확인
    if es_client.ping():
        print("✅ 성공! Elasticsearch 서버에 연결되었습니다.")

        # 실제 정보도 가져와 봅니다.
        info = es_client.info()
        print("\n--- 서버 정보 ---")
        print(f"Cluster Name: {info['cluster_name']}")
        print(f"ES Version: {info['version']['number']}")
        print("-----------------")

    else:
        print("❌ 실패! 서버가 응답하지 않습니다. ping() 실패.")

except ConnectionError as e:
    print("\n💥 연결 오류 발생! 서버 주소나 네트워크를 확인하세요.")
    print(f"   자세한 오류: {e}")

except Exception as e:
    print(f"\n💥 예상치 못한 다른 오류 발생: {e}")