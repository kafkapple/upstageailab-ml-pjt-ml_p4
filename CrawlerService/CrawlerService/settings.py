# Scrapy settings for NaverBlogCrawler project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html
import random

BOT_NAME = "CrawlerService"

SPIDER_MODULES = ["CrawlerService.spiders"]
NEWSPIDER_MODULE = "CrawlerService.spiders"

# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0',
]
USER_AGENT = random.choice(USER_AGENTS)

# Obey robots.txt rules
ROBOTSTXT_OBEY = True  # robots.txt 준수

# 중복 필터 디버그 설정
DUPEFILTER_DEBUG = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 4  # 동시 요청 수 제한 (기본값: 16)

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 1  # 요청 간 대기 시간[초] 
RANDOMIZE_DOWNLOAD_DELAY = True

# The download delay setting will honor only one of:
# 크롤링 대상이 동일한 도메인에서만 작동하는 경우 활성화
CONCURRENT_REQUESTS_PER_DOMAIN = 4  # 도메인별 최대 병렬 요청 수
# 대상이 여러 IP를 사용하는 서비스인 경우 활성화
# CONCURRENT_REQUESTS_PER_IP = 16

# 네트워크 요청 재시도 설정
RETRY_ENABLED = True  # 요청 재시도 활성화
# RETRY_TIMES = 5  # 요청 재시도 횟수 (기본값: 2)
RETRY_TIMES = 7
RETRY_HTTP_CODES = [500, 502, 503, 504, 408]  # 재시도 대상 HTTP 상태 코드

# 크롤링 종료 조건
CLOSESPIDER_ITEMCOUNT = 100  # 수집할 최대 아이템 개수
CLOSESPIDER_PAGECOUNT = 10   # 최대 크롤링 페이지 수

# Disable cookies (enabled by default)
# 대상 사이트가 쿠키를 사용하지 않거나, 쿠키 처리가 필요하지 않다면 비활성화
COOKIES_ENABLED = False  # 쿠키를 비활성화

# Disable Telnet Console (enabled by default)
# 디버깅이 필요하지 않다면 비활성화
TELNETCONSOLE_ENABLED = False  # 비활성화

# Override the default request headers:
# 크롤링 대상이 헤더 설정을 필요로 한다면 활성화
# 한국어로 설정하려면 Accept-Language 를 "ko"로 변경
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko",
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# 기본적으로 Scrapy는 내장된 미들웨어를 사용
# 사용자 정의 스파이더 미들웨어가 필요하다면 활성화하고 middlewares.py에서 구현
# 현재는 필요 없어 보이므로 비활성화 상태를 유지
# SPIDER_MIDDLEWARES = {
#     "NaverBlogCrawler.middlewares.NaverblogcrawlerSpiderMiddleware": 543,
# }

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
# 사용자 정의 다운로드 미들웨어가 필요하거나, 요청 또는 응답을 수정하려면 활성화
# 예: 프록시 설정, 헤더 조작, 데이터 변조 등
# 현재로서는 필요 없어 보이므로 비활성화 상태를 유지
# DOWNLOADER_MIDDLEWARES = {
#     "NaverBlogCrawler.middlewares.NaverblogcrawlerDownloaderMiddleware": 543,
# }

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# Telnet Console을 비활성화하려면 활성화
# 대부분의 프로젝트에서는 Telnet Console이 필요하지 않음.
EXTENSIONS = {
    "scrapy.extensions.telnet.TelnetConsole": None,
}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    "CrawlerService.pipelines.NaverblogcrawlerPipeline": 300,  # JSON,CSV 저장 파이프라인
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True  # 자동 속도 조절 활성화

# The initial download delay
AUTOTHROTTLE_START_DELAY = 2  # 0.5초 이하로 설정하면 서버 차단 위험이 높아질 수 있음.

# The maximum download delay to be set in case of high latencies
# AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_MAX_DELAY = 10

# The average number of requests Scrapy should be sending in parallel to
# each remote server
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0  # 5 이상으로 설정하면 서버 차단 위험이 있음.

# Enable showing throttling stats for every response received:
# AutoThrottle 설정이 활성화되어 있고, 디버깅 로그가 필요하다면 활성화
# 크롤링 속도와 관련된 디버깅을 확인할 수 있음.
# 디버깅 목적이 아니라면 비활성화 상태를 유지
# AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
# HTTP 캐시는 동일한 URL에 대해 반복적으로 요청하는 것을 방지하여, 네트워크 사용량을 줄이고 크롤링 속도를 높임.
# 개발 중이거나 자주 변경되지 않는 데이터를 크롤링할 때 유용함.
# 대상 사이트가 정적 콘텐츠를 제공하거나, 동일 데이터를 반복적으로 요청한다면 활성화
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600  # 캐시 만료 시간: 1시간
HTTPCACHE_DIR = "httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [500, 503, 504, 400, 403, 404]
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# Job directory configuration
JOBDIR = "crawls/naver_blog"

# Data export configuration
FEEDS = {
    "output/naver_blog_crawling.json": {"format": "json", "encoding": "utf-8", "indent": 4},
    "output/naver_blog_crawling.csv": {"format": "csv", "encoding": "utf-8"},
}

# Logging configuration
LOG_ENABLED = True  # 로그 기록을 활성화합니다.
LOG_STDOUT = False  # stdout 및 stderr 출력 포함
LOG_FILE = "logs/naver_blog_crawling.log"  # 로그 파일 저장 경로
LOG_FILE_MODE = "w"  # 파일을 덮어쓰기 모드로 설정
LOG_LEVEL = "INFO"  # 로그 레벨
# LOG_LEVEL = "DEBUG"  # 로그 레벨

import logging  # logging 모듈 임포트 추가
from scrapy.utils.project import get_project_settings
settings = get_project_settings()
# print(f"LOG_FILE_MODE: {settings.get('LOG_FILE_MODE')}")

# 로그 설정 직접 추가
logging.basicConfig(
    filename=LOG_FILE,
    filemode=LOG_FILE_MODE,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
