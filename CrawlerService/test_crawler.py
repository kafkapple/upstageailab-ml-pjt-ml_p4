from CrawlerService.crawler_service import run_crawler

# 크롤링 테스트
output_filename = "naver_blog_crawling"
# result = run_crawler(query="일본,중국,베트남", verbose=True, max_results=2, output_filename=output_filename)
# result = run_crawler(query="프랑스,이탈리아,스위스", verbose=True, max_results=2, output_filename=output_filename)
# result = run_crawler(query="이재명,조국,한동훈", verbose=True, max_results=4, output_filename=output_filename)
result = run_crawler(query="일본 도쿄,한국 서울", verbose=True, max_results=4, output_filename=output_filename)
print(f"크롤링 결과 저장 (CSV file): {result}")
print(f"크롤링 결과 저장 (JSON file): {result}")
