# 표준 라이브러리
import time
import sys
import os
import re
from urllib.parse import unquote

# 외부 라이브러리
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Scrapy 관련 모듈
import scrapy

# 프로젝트 내부 모듈
from CrawlerService.items import NaverBlogCrawlerItem

# Set recursion limit for log outputs or deeply nested data
# Increase recursion limit for deeply nested data
sys.setrecursionlimit(10000)

class NaverBlogSpider(scrapy.Spider):
    name = "naver_blog_spider"
    allowed_domains = ["search.naver.com", "blog.naver.com"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = [q.strip() for q in kwargs.get("query", "").split(",") if q.strip()]  # 쉼표로 분리된 검색어 리스트
        # verbose 값을 처리할 때 str인지 bool인지 확인
        verbose = kwargs.get("verbose", "False")
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            self.verbose = verbose.lower() == "true"  # verbose 플래그 추가 (문자열을 부울로 변환)
        self.max_results = int(kwargs.get("max_results", 3))  # 하나의 query 당 최대 가져올 게시물 수
        self.driver = self.get_chrome_driver()  # Selenium WebDriver를 한 번만 초기화
        self.clear_output_files()

    def clear_output_files(self):
        """이전에 생성된 파일 삭제 및 필요한 파일 초기화"""
        files_to_clear = [
            # "logs/naver_blog_crawling.log",
            "output/naver_blog_crawling.json",
            "output/naver_blog_crawling.csv",
            "output/naver_blog_cleaned_data.json",
            "output/naver_blog_cleaned_data.csv",
        ]
        for file_path in files_to_clear:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Deleted file: {file_path}")
            else:
                self.logger.info(f"No file to delete: {file_path}")

        # 빈 로그 파일을 미리 생성
        log_file_path = "logs/naver_blog_crawling.log"
        if not os.path.exists(log_file_path):
            open(log_file_path, "w").close()
            self.logger.info(f"Initialized log file: {log_file_path}")

    def start_requests(self):
        """
        Selenium으로 데이터를 가져와 Scrapy로 처리.
        각 요청 URL에 동적으로 timestamp를 추가하여 크롤링 차단을 방지.
        """
        # Scrapy 명령어 옵션으로 query 전달받기
        # "query"가 전달되면 이를 사용하고, 그렇지 않으면 기본 리스트 사용
        # input_query = getattr(self, "query", None)
        input_query = getattr(self, "query", [])
        if input_query and input_query != [""]:
            # 유효한 검색어가 있는 경우만 처리
            # 입력된 query 문자열을 쉼표(,)로 구분하여 각 쿼리에 대해 URL을 동적으로 생성
            # query.strip()을 사용하여 각 쿼리의 앞뒤 공백을 제거
            # 생성된 URL은 base_urls 리스트에 저장되어 이후 크롤링에 사용됨
            # [예시] scrapy crawl naver_blog_spider -a verbose=True -a query=query1,query2,query3
            # input_query = [unquote(q) for q in input_query.split(",")]  # 디코딩 처리
            input_query = [unquote(q) for q in input_query]  # 디코딩 처리
            base_urls = [
                f"https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query={query.strip()}"
                for query in input_query  # 이미 리스트인 input_query를 순회
            ]
        else:
            # 기본 URL 리스트
            base_urls = [
                "https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query=대한항공",
                "https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query=아시아나항공",
                "https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query=제주항공",
            ]

        # 각 URL에 대해 작업 수행
        for base_url in base_urls:
            query = base_url.split("query=")[-1]
            # 동적으로 timestamp 추가
            timestamp = int(time.time() * 1000)  # 밀리초 단위의 현재 시간
            url_with_timestamp = f"{base_url}&timestamp={timestamp}"
            self.logger.info(f"Fetching posts for URL: {url_with_timestamp}")

            try:
                self.ensure_driver()  # WebDriver 상태 확인 및 재초기화
                self.driver.get(url_with_timestamp)  # WebDriver 재사용
                posts = self.extract_blog_links(self.driver)  # 블로그 링크 추출
                if posts:
                    self.logger.info(f"Extracted {len(posts)} posts from {url_with_timestamp}")
                    for post in posts:
                        # 각 블로그 게시물 URL에 대해 Scrapy Request 생성
                        if self.verbose:
                            print(f"[{query}] {post['link']}")  # 검색된 블로그 URL 출력
                        post["query"] = query
                        yield scrapy.Request(
                            url=post["link"],
                            callback=self.parse_post,
                            meta=post,  # 메타 데이터로 post 정보 전달
                            dont_filter=True,  # 중복 필터링 방지
                        )
                else:
                    self.logger.warning(f"No posts found for URL: {url_with_timestamp}")
            except Exception as e:
                self.logger.exception(f"Error processing {url_with_timestamp}")  # 스택 트레이스 로깅

    def get_chrome_driver(self):
        """Selenium ChromeDriver 초기화"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")  # 확장 프로그램 비활성화
        chrome_options.add_argument("--disable-infobars")  # 브라우저 인포바 비활성화
        chrome_options.add_argument("--disable-notifications")  # 알림 비활성화
        service = Service("/usr/bin/chromedriver")
        return webdriver.Chrome(service=service, options=chrome_options)

    def safe_get(self, url, retries=3, delay=2):
        """
        URL을 안전하게 로드하며 네트워크 오류 발생 시 재시도합니다.
        """
        for attempt in range(retries):
            try:
                self.driver.get(url)
                return  # 성공적으로 로드된 경우 종료
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for URL: {url}. Error: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)  # 재시도 전 대기
                else:
                    raise  # 재시도 초과 시 예외 발생

    def ensure_driver(self):
        """
        WebDriver 상태를 확인하고, None이거나 비정상 상태일 경우 새로 초기화합니다.
        """
        if self.driver is None:
            self.logger.info("Initializing a new WebDriver instance.")
            self.driver = self.get_chrome_driver()
        else:
            try:
                # WebDriver가 정상적으로 작동하는지 테스트
                self.driver.current_url  # 현재 URL 확인
            except Exception as e:
                self.logger.warning(f"WebDriver is in an invalid state: {e}")
                self.logger.info("Reinitializing WebDriver.")
                try:
                    self.driver.quit()  # 기존 드라이버 종료
                except Exception:
                    self.logger.warning("Failed to quit the existing WebDriver.")
                finally:
                    self.driver = self.get_chrome_driver()

    def closed(self, reason):
        """Spider 종료 시 WebDriver 종료"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                self.logger.warning("Error while closing WebDriver.")
            finally:
                self.driver = None

    def parse_post_data(self, post):
        """
        개별 블로그 게시물의 데이터를 추출.
        Args:
            post (WebElement): Selenium으로 가져온 블로그 게시물 요소
        Returns:
            dict: 블로그 게시물의 제목, 링크, 작성자, 미리보기, 날짜 정보를 포함한 딕셔너리
        """
        try:
            title = post.find_element(By.CSS_SELECTOR, "a.title_link").text.strip()
            link = post.find_element(By.CSS_SELECTOR, "a.title_link").get_attribute("href")
            author_name = post.find_element(By.CSS_SELECTOR, "div.user_info a.name").text.strip()
            date = post.find_element(By.CSS_SELECTOR, "span.sub").text.strip()
            preview = post.find_element(By.CSS_SELECTOR, "div.dsc_area a.dsc_link").text.strip()

            # 필수 정보 누락 시 예외 발생
            if not title or not link:
                self.logger.warning("Skipped post due to missing title or link.")
                return None

            return {
                "title": title,
                "link": link,
                "author_name": author_name,
                "date": date,
                "preview": preview,
            }
        except Exception as e:
            self.logger.warning(f"Error extracting data from post: {e}")
            return None

    def extract_blog_links(self, driver):
        """
        블로그 링크 추출 메서드.
        Args:
            driver (WebDriver): Selenium WebDriver 인스턴스
        Returns:
            블로그 게시물 정보 리스트
        """        
        results = []
        visited_links = set()
        try:
            posts = driver.find_elements(By.CSS_SELECTOR, "ul.lst_view li.bx")
            if not posts:
                self.logger.warning("No blog posts found on the page.")
                return results  # 빈 리스트 반환
            
            for post in posts[:self.max_results]:
                post_data = self.parse_post_data(post)
                if post_data and post_data["link"] not in visited_links:
                    visited_links.add(post_data["link"])
                    results.append(post_data)
        except Exception as e:
            self.logger.exception("Error extracting blog links")  # 스택 트레이스 로깅
        return results

    def extract_blog_content(self, driver):
        """iframe 내부의 본문 콘텐츠를 추출"""
        selectors = ["div.se-viewer", "div#post-area"]  # 다양한 본문 영역에 대한 선택자
        for selector in selectors:
            try:
                content = driver.find_element(By.CSS_SELECTOR, selector).text
                self.logger.debug(f"Selector {selector} succeeded for {driver.current_url}")
                return content.strip()
            except Exception as e:
                self.logger.debug(f"Selector {selector} failed for {driver.current_url}: {e}")
                continue
        self.logger.warning("Failed to extract blog content from available selectors.")
        return None


    def parse_post(self, response):
        """개별 블로그 포스트의 본문 데이터를 Selenium을 사용해 추출"""
        try:
            self.ensure_driver()  # WebDriver 상태 확인 및 재초기화
            self.safe_get(response.url)  # 블로그 URL로 이동

            # mainFrame으로 전환
            iframe = self.driver.find_element(By.ID, "mainFrame")
            self.driver.switch_to.frame(iframe)

            # iframe 내 본문 콘텐츠 추출
            content = self.extract_blog_content(self.driver) or "No Content Available"
            self.driver.switch_to.default_content()  # 최상위 프레임으로 복귀
        except Exception as e:
            self.logger.exception(f"Error parsing post {response.url}")  # 스택 트레이스 로깅
            content = "No Content Available"

        # 아이템 생성 및 필드 매핑
        item = NaverBlogCrawlerItem()
        item["title"] = response.meta.get("title", "No Title")
        item["url"] = response.url
        item["author_name"] = response.meta.get("author_name", "Unknown")
        item["date"] = response.meta.get("date", "Unknown Date")
        item["preview"] = response.meta.get("preview", "No Preview")
        item["content"] = content if content else "No Content Available"
        item['content'] = re.sub(r',', '', item['content'])  # comma( ,)를 제거
        item['content'] = re.sub(r'\n', ' ', item['content'])    # \n을 공백으로 대체

        # 디버깅 로그: 추출된 item 객체 전체를 디버깅 로그에 기록
        self.logger.debug(f"Scraped item: {item}")

        # 디버깅 로그: 추출된 content의 일부를 기록
        self.logger.debug(f"Extracted content for URL {response.url}: {content[:100] if content else 'No Content'}")

        yield item


if __name__ == "__main__":
    """
    스크래피 CLI를 사용하지 않고 직접 스파이더를 실행할 경우를 위한 옵션.
    주로 테스트 또는 별도 실행 환경에서 사용됩니다.
    """
    try:
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings

        process = CrawlerProcess(get_project_settings())
        process.crawl("naver_blog_spider")
        process.start()
    except Exception as e:
        print(f"Error running spider: {e}")
