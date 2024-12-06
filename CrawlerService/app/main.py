from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import os
from CrawlerService.crawler_service import run_crawler

app = FastAPI()

# 요청 모델 정의
class CrawlRequest(BaseModel):
    query: str  # 검색어 쿼리 (쉼표로 구분)
    verbose: bool = False  # 터미널 출력 여부
    max_results: int = 3  # 크롤링 최대 결과 수
    output_filename: str = "naver_blog_crawling"  # 결과를 저장할 파일 이름 (확장자 제외)


@app.get("/")
def read_root():
    """
    기본 라우트: API 상태 확인용
    """
    return {"message": "Welcome to the Blog Crawler API"}


@app.post("/crawl/")
def crawl_data(request: CrawlRequest):
    """
    Scrapy 크롤러를 실행하고 결과 파일 경로를 반환합니다.
    """
    try:
        # Scrapy 프로젝트 루트 경로
        project_root = Path(__file__).parent.parent

        # 출력 디렉토리 설정
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)  # 출력 폴더가 없으면 생성
        
        # Scrapy 크롤러 실행
        result_file = run_crawler(
            query=request.query,
            verbose=request.verbose,
            max_results=request.max_results,
            output_filename=request.output_filename,
        )
                
        return {"status": "success", "file_path": result_file}
    except Exception as e:
        # 크롤러 실행 중 오류 처리
        print(f"Error in FastAPI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/csv/{csv_file}")
def download_csv_file(csv_file: str):
    """
    CSV 파일을 다운로드합니다.
    """
    file_path = os.path.join("output", csv_file)
    if os.path.exists(file_path):
        # 파일이 존재하면 다운로드 제공
        return FileResponse(file_path, media_type="application/csv", filename=csv_file)
    raise HTTPException(status_code=404, detail="CSV File not found")


@app.get("/download/json/{json_file}")
def download_json_file(json_file: str):
    """
    JSON 파일을 다운로드합니다.
    """
    file_path = os.path.join("output", json_file)
    if os.path.exists(file_path):
        # 파일이 존재하면 다운로드 제공
        return FileResponse(file_path, media_type="application/json", filename=json_file)
    raise HTTPException(status_code=404, detail="JSON File not found")
