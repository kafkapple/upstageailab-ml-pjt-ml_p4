import os
from datetime import datetime
# from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from CrawlerService.spiders.naver_blog_spider import NaverBlogSpider


def run_crawler(query, verbose, max_results, output_filename):
    """
    Scrapy 크롤러를 실행하고 결과를 지정된 경로에 저장합니다.
    
    Args:
        query (str): 검색어 쿼리. 쉼표로 여러 검색어를 구분.
        verbose (bool): 상세 로그 출력 여부.
        max_results (int): 크롤링 최대 결과 개수.
        output_filename (str): 결과를 저장할 파일 이름 (확장자 제외).
    
    Returns:
        str: 결과가 저장된 파일 이름 (확장자 제외).
    
    Raises:
        Exception: 크롤링 실행 중 발생한 에러.
    """
    try:
        # 로그 파일 설정
        log_file = "logs/naver_blog_crawling.log"
        os.makedirs("logs", exist_ok=True)  # logs 디렉토리가 없으면 생성

        # 로그 파일 초기화
        if os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("")  # 로그 파일 비우기

        # 출력 파일 경로 설정
        output_csv = f"output/{output_filename}.csv"
        output_json = f"output/{output_filename}.json"
        os.makedirs("output", exist_ok=True)  # output 디렉토리가 없으면 생성

        # 출력 파일 초기화
        for file_path in [output_csv, output_json]:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Scrapy 설정 가져오기
        settings = get_project_settings()
        settings.update({
            "LOG_FILE": log_file,
            "LOG_LEVEL": "INFO",
            "FEEDS": {
                output_csv: {"format": "csv", "encoding": "utf-8", "overwrite": True},  # 결과를 CSV로 저장
                output_json: {"format": "json", "encoding": "utf-8", "indent": 4, "overwrite": True},  # 결과를 JSON으로 저장                
            },
        })

        # Scrapy 프로세스 실행
        process = CrawlerProcess(settings)
        # process.crawl("naver_blog_spider", query=query, max_results=max_results, verbose=verbose)
        process.crawl(NaverBlogSpider, query=query, max_results=max_results, verbose=verbose)
        process.start()  # Blocking call, 크롤링이 끝날 때까지 대기
        return {"csv": output_csv, "json": output_json}
    except Exception as e:
        print(f"Error during crawling: {e}")
        raise
