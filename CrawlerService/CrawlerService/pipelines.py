# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

# useful for handling different item types with a single interface
import json
import csv
from datetime import datetime, timedelta
from itemadapter import ItemAdapter


class NaverblogcrawlerPipeline:
    def __init__(self):
        self.seen_urls = set()  # 중복 제거를 위한 URL 저장
        self.json_file_path = "output/naver_blog_cleaned_data.json"  # 저장할 JSON 파일 경로
        self.csv_file_path = "output/naver_blog_cleaned_data.csv"  # 저장할 CSV 파일 경로

    def open_spider(self, spider):
        """크롤러 시작 시 호출, 파일 초기화"""
        spider.logger.info("Opening output files")

        # JSON 파일 초기화
        self.json_file = open(self.json_file_path, "w", encoding="utf-8")
        
        # CSV 파일 초기화
        self.csv_file = open(self.csv_file_path, "w", encoding="utf-8", newline="")
        self.csv_writer = None  # CSV writer 객체를 나중에 생성

    def close_spider(self, spider):
        """크롤러 종료 시 호출, 파일 닫기"""
        spider.logger.info("Closing output files")
        self.json_file.close()
        self.csv_file.close()

    def process_item(self, item, spider):
        """아이템 처리 로직"""
        spider.logger.info("Starting process_item")
        
        # 디버깅용 로그 추가
        spider.logger.debug(f"Raw item in pipeline: {item}")

        # 필수 필드 검증
        required_fields = ["title", "content", "url"]
        for field in required_fields:
            if not item.get(field):
                spider.logger.warning(f"Missing required field '{field}' in item: {item}")
                return None  # 필수 필드 누락 시 아이템 스킵

        # 중복 제거
        if item["url"] in self.seen_urls:
            spider.logger.warning(f"Duplicate item skipped: {item['url']}")
            return None
        self.seen_urls.add(item["url"])

        # 데이터 정제
        spider.logger.debug(f"Cleaning item: {item}")
        item = self.clean_item(item)

        # JSON 및 CSV 파일 저장
        spider.logger.debug(f"Saving item to files: {item}")
        self.save_to_json(item)
        self.save_to_csv(item)

        spider.logger.info("Processed item successfully")
        return item

    def clean_item(self, item):
        """아이템 데이터를 정제"""
        # 제목과 미리보기 텍스트의 앞뒤 공백 제거
        if item.get("title"):
            item["title"] = item["title"].strip()
        if item.get("preview"):
            item["preview"] = item["preview"].strip()

        # 날짜 형식 변환 (예: '3일 전' → ISO 8601 형식)
        if item.get("date"):
            item["date"] = self.parse_date(item["date"])

        return item

    def parse_date(self, date_str):
        """날짜 형식 변환"""
        try:
            if "일 전" in date_str:
                days_ago = int(date_str.replace("일 전", "").strip())
                return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            elif "시간 전" in date_str:
                hours_ago = int(date_str.replace("시간 전", "").strip())
                return (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return date_str  # 변환 실패 시 원래 값 반환
        return date_str

    def save_to_json(self, item):
        """JSON 파일에 아이템 저장"""
        line = json.dumps(ItemAdapter(item).asdict(), ensure_ascii=False) + "\n"
        self.json_file.write(line)

    def save_to_csv(self, item):
        """CSV 파일에 아이템 저장"""
        item_dict = ItemAdapter(item).asdict()
        
        # CSV 파일의 헤더 초기화
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=item_dict.keys())
            self.csv_writer.writeheader()  # 헤더 작성

        # 아이템 추가
        self.csv_writer.writerow(item_dict)
