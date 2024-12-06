# 사전 설정 : 네이버 메일 좌측 하단 "환경설정" -> POP3/IMAP 설정 -> 사용함 체크 -> 저장

from dotenv import load_dotenv; import os; 
envFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(f'{envFolder}/.env')
naverid, naverpw = os.getenv("naverid"), os.getenv("naverpw")

from email.header import Header
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication


recipient = "xxx@naver.com" # 수신자 이메일 적기
recipients = [recipient]

message = MIMEMultipart();
message['Subject'] = 'Streamit 메일 전송 테스트입니당'
message['From'] = f"{naverid}@naver.com"
message['To'] = ",".join(recipients)
message.set_charset('utf-8')

title = '메일.. 받으셨나요..?'
content = f"""
    <html>
    <body>
        <h2>{title}</h2>
        <p>안녕하세요 테스트입니다.</p>
    </body>
    </html>
"""

mimetext = MIMEText(content,'html')
message.attach(mimetext)

server = smtplib.SMTP('smtp.naver.com', 587)
server.ehlo()
server.starttls()

# print(naverid, naverpw)

server.login(naverid, naverpw)

server.sendmail(message['From'],recipients,message.as_string())
server.quit()

# # 파일 첨부
# with open("example.pdf", "rb") as file:
#     file_part = MIMEApplication(file.read(), Name="example.pdf")
#     file_part['Content-Disposition'] = 'attachment; filename="example.pdf"'
#     message.attach(file_part)