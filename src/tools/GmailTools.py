"""
Gmail Tools - 使用 Gmail API 和 OAuth2 认证

需要配置步骤：
1. 在 Google Cloud Console 创建项目并启用 Gmail API
2. 下载 OAuth2 客户端凭据 JSON 文件，保存为 credentials.json
3. 首次运行时会打开浏览器进行授权，生成 token.json
"""

import os
import re
import base64
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parsedate_to_datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.drafts'
]

# 从环境变量读取代理设置
from dotenv import load_dotenv

# 显式指定 .env 文件路径
load_dotenv("/Users/yhr/Agent/langgraph-email-automation/.env")


class GmailToolsClass:
    """Gmail 工具类，使用 Gmail API 和 OAuth2 认证"""

    def __init__(self):
        self.email = os.getenv('GMAIL_USER', os.getenv('MY_EMAIL', ''))
        self.credentials_path = os.getenv('GMAIL_CREDENTIALS_PATH', 'credentials.json')
        self.token_path = os.getenv('GMAIL_TOKEN_PATH', 'token.json')
        self.service = self._build_service()

    def _get_credentials(self) -> Credentials:
        """获取 OAuth2 凭据"""
        creds = None

        # 检查是否已有 token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        # 如果凭据无效或过期，刷新或重新获取
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"未找到 OAuth2 凭据文件: {self.credentials_path}\n"
                        "请从 Google Cloud Console 下载客户端凭据 JSON 文件并保存为 credentials.json"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # 保存凭据供下次使用
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        return creds

    def _build_service(self):
        """构建 Gmail API 服务"""
        creds = self._get_credentials()
        return build('gmail', 'v1', credentials=creds)

    def fetch_unanswered_emails(self, max_results=50):
        """
        获取未回复的邮件

        @param max_results: 最大获取数量
        @return: 邮件列表
        """
        import json
        import time
        
        for attempt in range(3):
            try:
                # 获取收件箱中最近8小时的邮件
                eight_hours_ago = datetime.now(timezone.utc) - timedelta(hours=8)
                date_str = eight_hours_ago.strftime("%Y-%m-%d")
                # 简化查询：先移除 -from:me 看是否能连接
                query = f'after:{date_str}'
                
                # #region debug log
                with open("/Users/yhr/Agent/langgraph-email-automation/.cursor/debug-32110b.log", "a") as f:
                    f.write(json.dumps({"sessionId":"32110b","runId":"debug","hypothesisId":"B","location":"GmailTools.py:fetch_unanswered_emails","message":"Gmail query attempt","data":{"attempt":attempt+1,"query":query},"timestamp":1700000000000}) + "\n")
                # #endregion
                
                results = self.service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=max_results
                ).execute()

                # #region debug log
                with open("/Users/yhr/Agent/langgraph-email-automation/.cursor/debug-32110b.log", "a") as f:
                    f.write(json.dumps({"sessionId":"32110b","runId":"debug","hypothesisId":"B","location":"GmailTools.py:fetch_unanswered_emails","message":"Gmail API success","data":{"result_keys":list(results.keys()),"messages_count":len(results.get('messages', []))},"timestamp":1700000000001}) + "\n")
                # #endregion

                messages = results.get('messages', [])
                if not messages:
                    return []

                emails = []
                my_email = self.email.lower()

                for msg_ref in messages:
                    msg_id = msg_ref['id']  # Gmail API 返回的是 'id' 不是 'message_id'
                    email_info = self._get_email_by_id(msg_id)

                    if email_info:
                        # 跳过自己发送的邮件
                        sender_email = self._extract_email_address(email_info.get('sender', ''))
                        if sender_email.lower() == my_email:
                            continue

                        # 检查是否已有回复（通过查找同一线程的已发送邮件）
                        if not self._has_replied(msg_id):
                            emails.append(email_info)

                return emails

            except Exception as e:
                # #region debug log
                with open("/Users/yhr/Agent/langgraph-email-automation/.cursor/debug-32110b.log", "a") as f:
                    f.write(json.dumps({"sessionId":"32110b","runId":"debug","hypothesisId":"B","location":"GmailTools.py:fetch_unanswered_emails:exception","message":"API call exception","data":{"attempt":attempt+1,"error_type":type(e).__name__,"error_message":str(e)},"timestamp":1700000000002}) + "\n")
                # #endregion
                
                if attempt < 2:
                    print(f"获取邮件超时，重试中... ({attempt+1}/3)")
                    time.sleep(2)  # 等待2秒后重试
                else:
                    print(f"获取未回复邮件失败: {e}")
                    return []

    def _get_email_by_id(self, msg_id):
        """根据邮件 ID 获取邮件详情"""
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()

            headers = message.get('payload', {}).get('headers', [])

            # 提取邮件头
            email_headers = {}
            for header in headers:
                name = header['name'].lower()
                email_headers[name] = header['value']

            # 获取邮件正文
            body = self._get_email_body(message.get('payload', {}))

            # 解析日期
            date_str = email_headers.get('date', '')
            occurred_at = self._parse_date(date_str)

            # 获取 References 和 In-Reply-To 用于线程追踪
            references = email_headers.get('references', '')
            message_id = email_headers.get('message-id', '')
            thread_id = message.get('threadId', '')

            return {
                "id": msg_id,
                "threadId": thread_id,
                "messageId": message_id,
                "references": references,
                "sender": email_headers.get('from', ''),
                "subject": email_headers.get('subject', ''),
                "body": body,
                "occurred_at": occurred_at,
            }

        except Exception as e:
            print(f"获取邮件详情失败: {e}")
            return None

    def _get_email_body(self, payload):
        """从邮件 payload 中提取正文"""
        body = ""

        # 递归查找正文
        def find_body(part):
            nonlocal body
            if body:
                return

            mime_type = part.get('mimeType', '')
            body_data = part.get('body', {})

            # 优先获取纯文本
            if mime_type == 'text/plain' and body_data.get('data'):
                body = base64.urlsafe_b64decode(body_data['data']).decode('utf-8', errors='replace')
                return

            # 如果没有纯文本，获取 HTML
            if mime_type == 'text/html' and not body and body_data.get('data'):
                body = base64.urlsafe_b64decode(body_data['data']).decode('utf-8', errors='replace')
                return

            # 遍历多部分邮件的子部分
            parts = part.get('parts', [])
            for p in parts:
                find_body(p)

        find_body(payload)

        # 清理 HTML 标签
        if body:
            body = self._clean_html(body)

        return body

    def _clean_html(self, text):
        """清理 HTML 内容"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        for tag in soup(['script', 'style', 'head', 'meta', 'title']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).strip()

    def _has_replied(self, thread_id):
        """检查是否已在该线程中回复过"""
        try:
            # 获取同一线程的所有消息
            thread = self.service.users().threads().get(
                userId='me',
                id=thread_id
            ).execute()

            messages = thread.get('messages', [])
            my_email = self.email.lower()

            # 检查是否发送过邮件
            for msg in messages:
                headers = msg.get('payload', {}).get('headers', [])
                for header in headers:
                    if header['name'].lower() == 'from':
                        sender = self._extract_email_address(header['value'])
                        if sender.lower() == my_email:
                            return True

            return False

        except Exception:
            return False

    def create_draft_reply(self, initial_email, reply_text):
        """
        创建草稿回复

        @param initial_email: 原始邮件对象
        @param reply_text: 回复内容
        @return: 包含草稿路径的字典
        """
        try:
            message = self._create_reply_message(initial_email, reply_text)

            # 创建草稿
            draft = self.service.users().drafts().create(
                userId='me',
                body={
                    'message': {
                        'raw': base64.urlsafe_b64encode(
                            message.as_bytes()
                        ).decode('utf-8')
                    }
                }
            ).execute()

            draft_id = draft.get('id', '')
            print(f"草稿已创建: {draft_id}")

            return {"draft_id": draft_id, "draft_path": f"https://mail.google.com/mail/#drafts/{draft_id}"}

        except Exception as e:
            print(f"创建草稿失败: {e}")
            return None

    def send_reply(self, initial_email, reply_text):
        """
        发送回复邮件

        @param initial_email: 原始邮件对象
        @param reply_text: 回复内容
        @return: 发送结果
        """
        try:
            message = self._create_reply_message(initial_email, reply_text, send=True)

            send_result = self.service.users().messages().send(
                userId='me',
                body={
                    'raw': base64.urlsafe_b64encode(
                        message.as_bytes()
                    ).decode('utf-8')
                }
            ).execute()

            print(f"邮件已发送: {send_result.get('id', '')}")
            return {"status": "sent", "message_id": send_result.get('id', '')}

        except Exception as e:
            print(f"发送邮件失败: {e}")
            return None

    def _create_reply_message(self, email_data, reply_text, send=False):
        """创建回复邮件"""
        # 获取原始邮件信息
        sender = email_data.get('sender', '')
        subject = email_data.get('subject', 'No Subject')
        if not subject.startswith('Re: '):
            subject = f'Re: {subject}'

        # 创建邮件
        message = MIMEMultipart('alternative')
        message['To'] = sender
        message['Subject'] = subject

        # 设置回复头
        in_reply_to = email_data.get('messageId', '')
        if in_reply_to:
            message['In-Reply-To'] = in_reply_to
            references = email_data.get('references', '')
            message['References'] = f"{references} {in_reply_to}".strip()

        # 生成 Message-ID
        import uuid
        message['Message-ID'] = f"<{uuid.uuid4()}@{self.email.split('@')[1]}>"

        # 纯文本内容
        text_part = MIMEText(reply_text, 'plain', 'utf-8')
        message.attach(text_part)

        # HTML 内容
        html_text = reply_text.replace('\n', '<br>')
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>{html_text}</body>
        </html>
        """
        html_part = MIMEText(html_content, 'html', 'utf-8')
        message.attach(html_part)

        return message

    @staticmethod
    def _extract_email_address(from_header):
        """从 From 头中提取邮箱地址"""
        if not from_header:
            return ""

        match = re.search(r'<([^>]+)>', from_header)
        if match:
            return match.group(1)

        match = re.search(r'[\w\.-]+@[\w\.-]+', from_header)
        if match:
            return match.group(0)

        return from_header

    @staticmethod
    def _parse_date(date_str):
        """解析邮件日期"""
        if not date_str:
            return datetime.now(timezone.utc).isoformat()

        try:
            dt = parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()
