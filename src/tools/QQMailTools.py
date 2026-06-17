import os
import re
import uuid
import imaplib
import smtplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# 显式指定 .env 文件路径
load_dotenv()


class QQMailTools:
    """QQ邮箱工具类，替代GmailTools使用IMAP/SMTP协议"""

    def __init__(self):
        self.email = os.environ.get('MY_EMAIL', '')
        self.password = os.environ.get('QQ_EMAIL_AUTH_CODE', '')  # QQ邮箱使用授权码
        self.smtp_server = os.environ.get('QQ_SMTP_SERVER', 'smtp.qq.com')
        self.smtp_port = int(os.environ.get('QQ_SMTP_PORT', 465))
        self.imap_server = os.environ.get('QQ_IMAP_SERVER', 'imap.qq.com')
        self.imap_port = int(os.environ.get('QQ_IMAP_PORT', 993))

        self._imap_conn = None
        self._smtp_conn = None

    def _get_imap_connection(self):
        """获取IMAP连接"""
        if self._imap_conn is None:
            self._imap_conn = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            self._imap_conn.login(self.email, self.password)
        return self._imap_conn

    def _get_smtp_connection(self):
        """获取SMTP连接"""
        if self._smtp_conn is None:
            self._smtp_conn = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            self._smtp_conn.login(self.email, self.password)
        return self._smtp_conn

    def fetch_unanswered_emails(self, max_results=50):
        """
        获取未回复的邮件（来自未回复的会话）

        @param max_results: 最大获取数量
        @return: 邮件列表
        """
        try:
            mail = self._get_imap_connection()
            mail.select('INBOX')

            # 搜索最近8小时内的邮件
            since_date = (datetime.now() - timedelta(hours=8)).strftime('%d-%b-%Y')

            # 搜索不在已发送邮件中的邮件（简化版，实际需要更复杂的逻辑）
            status, messages = mail.search(None, f'SINCE {since_date} UNSEEN')

            if status != 'OK':
                return []

            email_ids = messages[0].split()
            if not email_ids:
                return []

            # 获取最近的邮件
            recent_ids = email_ids[-min(max_results, len(email_ids)):]
            unanswered_emails = []

            # 获取已发送文件夹中所有发件人
            sent_senders = self._get_sent_senders()

            for msg_id in reversed(recent_ids):
                email_info = self._get_email_info_by_id(msg_id)

                # 跳过自己发送的邮件
                if self._should_skip_email(email_info):
                    continue

                # 简化逻辑：如果发件人不在已发送列表中，认为是未回复的
                if email_info['sender'] not in sent_senders:
                    unanswered_emails.append(email_info)

            return unanswered_emails

        except Exception as e:
            print(f"获取未回复邮件失败: {e}")
            return []

    def _get_sent_senders(self):
        """获取已发送邮件的发件人列表（用于判断是否已回复）"""
        sent_senders = set()
        try:
            mail = self._get_imap_connection()
            # 尝试访问已发送文件夹
            for folder in ['"已发送"', '"Sent"', '[Gmail]/Sent']:
                try:
                    mail.select(folder)
                    status, messages = mail.search(None, 'ALL')
                    if status == 'OK' and messages[0]:
                        for msg_id in messages[0].split()[-100:]:  # 只检查最近100封
                            _, data = mail.fetch(msg_id, '(ENVELOPE)')
                            # 简单处理：记录当前用户邮箱
                            sent_senders.add(self.email)
                    break
                except:
                    continue
            mail.select('INBOX')
        except:
            pass
        return sent_senders

    def fetch_recent_emails(self, max_results=50):
        """获取最近收到的邮件"""
        try:
            mail = self._get_imap_connection()
            mail.select('INBOX')

            # 搜索最近8小时内的未读邮件
            since_date = (datetime.now() - timedelta(hours=8)).strftime('%d-%b-%Y')
            status, messages = mail.search(None, f'SINCE {since_date} UNSEEN')

            if status != 'OK':
                return []

            email_ids = messages[0].split()
            if not email_ids:
                return []

            # 返回邮件ID列表
            return [{'id': msg_id.decode()} for msg_id in email_ids[-max_results:]]

        except Exception as error:
            print(f"获取邮件失败: {error}")
            return []

    def fetch_draft_replies(self):
        """
        获取草稿箱中的邮件
        QQ邮箱草稿箱中的邮件
        """
        try:
            mail = self._get_imap_connection()
            mail.select('"草稿箱"')  # QQ邮箱草稿箱

            status, messages = mail.search(None, 'ALL')
            if status != 'OK':
                return []

            drafts = []
            for msg_id in messages[0].split():
                _, data = mail.fetch(msg_id, '(RFC822)')
                msg = email.message_from_bytes(data[0][1])

                # 获取主题用于识别是回复邮件
                subject = self._decode_header_value(msg.get('Subject', ''))

                drafts.append({
                    "draft_id": msg_id.decode(),
                    "threadId": msg.get('Thread-Index', ''),
                    "id": msg_id.decode(),
                    "subject": subject
                })

            return drafts

        except Exception as error:
            print(f"获取草稿失败: {error}")
            return []

    def create_draft_reply(self, initial_email, reply_text):
        """创建草稿回复"""
        try:
            # 获取发送人邮箱（兼容 dict 和 Pydantic 模型）
            sender = initial_email.get('sender') if isinstance(initial_email, dict) else getattr(initial_email, 'sender', '')
            
            # 创建回复邮件
            message = self._create_reply_message(initial_email, reply_text)

            # 保存到草稿箱
            smtp = self._get_smtp_connection()
            smtp.sendmail(self.email, [sender], message.as_bytes())

            return True

        except Exception as error:
            print(f"创建草稿失败: {error}")
            return None

    def send_reply(self, initial_email, reply_text):
        """发送回复邮件"""
        try:
            # 获取收件人邮箱（兼容 dict 和 Pydantic 模型）
            sender = initial_email.get('sender', '') if isinstance(initial_email, dict) else getattr(initial_email, 'sender', '')
            
            # 创建回复邮件
            message = self._create_reply_message(initial_email, reply_text, send=True)

            # 获取收件人
            recipient = self._extract_email_address(sender)

            # 发送邮件
            smtp = self._get_smtp_connection()
            smtp.sendmail(self.email, [recipient], message.as_bytes())

            return True

        except Exception as error:
            print(f"发送邮件失败: {error}")
            return None

    def _create_reply_message(self, email_data, reply_text, send=False):
        """创建回复邮件"""
        # 获取原始邮件信息（兼容 dict 和 Pydantic 模型）
        if isinstance(email_data, dict):
            subject = email_data.get('subject', 'No Subject')
            sender = email_data.get('sender', '')
            message_id = email_data.get('messageId', '')
            references = email_data.get('references', '')
        else:
            subject = getattr(email_data, 'subject', 'No Subject')
            sender = getattr(email_data, 'sender', '')
            message_id = getattr(email_data, 'messageId', '')
            references = getattr(email_data, 'references', '')
            
        if not subject.startswith('Re: '):
            subject = f'Re: {subject}'

        # 创建邮件
        message = MIMEMultipart("alternative")
        message["From"] = self.email
        message["To"] = sender
        message["Subject"] = subject

        # 设置回复头
        in_reply_to = message_id
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = f"{references} {in_reply_to}".strip()

        # 生成Message-ID
        message["Message-ID"] = f"<{uuid.uuid4()}@{self.email.split('@')[1]}>"

        # HTML内容
        html_text = reply_text.replace("\n", "<br>")
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

        html_part = MIMEText(html_content, "html", "utf-8")
        message.attach(html_part)

        return message

    def _get_email_info_by_id(self, msg_id):
        """根据邮件ID获取邮件信息"""
        try:
            mail = self._get_imap_connection()
            _, data = mail.fetch(msg_id, '(RFC822)')

            if data is None or data[0] is None:
                return self._empty_email_info()

            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            sender = self._decode_header_value(msg.get('From', ''))
            subject = self._decode_header_value(msg.get('Subject', ''))
            message_id = msg.get('Message-ID', '')
            references = msg.get('References', '')
            date_str = msg.get('Date', '')

            # 获取邮件正文
            body = self._get_email_body(msg)

            # 生成threadId（QQ邮箱使用Message-ID作为线索）
            thread_id = msg.get('Thread-Index', message_id)

            return {
                "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                "threadId": thread_id,
                "messageId": message_id,
                "references": references,
                "sender": sender,
                "subject": subject,
                "body": body,
                "occurred_at": self._parse_date(date_str),
            }

        except Exception as e:
            print(f"获取邮件信息失败: {e}")
            return self._empty_email_info()

    def _empty_email_info(self):
        """返回空的邮件信息字典"""
        return {
            "id": "",
            "threadId": "",
            "messageId": "",
            "references": "",
            "sender": "",
            "subject": "",
            "body": "",
            "occurred_at": ""
        }

    def _should_skip_email(self, email_info):
        """判断是否跳过该邮件（跳过自己发送的邮件）"""
        my_email = os.environ.get('MY_EMAIL', '').lower()
        sender = email_info.get('sender', '').lower()
        return my_email in sender

    def _get_email_body(self, msg):
        """从邮件中提取正文"""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # 优先获取纯文本
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='replace')
                        break
                # 其次获取HTML
                elif content_type == "text/html" and not body and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        html_content = payload.decode(charset, errors='replace')
                        body = self._extract_main_content_from_html(html_content)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                content_type = msg.get_content_type()
                if content_type == "text/html":
                    body = self._extract_main_content_from_html(payload.decode(charset, errors='replace'))
                else:
                    body = payload.decode(charset, errors='replace')

        return self._clean_body_text(body)

    def _extract_main_content_from_html(self, html_content):
        """从HTML中提取主要内容"""
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style', 'head', 'meta', 'title']):
            tag.decompose()
        return soup.get_text(separator='\n', strip=True)

    def _clean_body_text(self, text):
        """清理邮件正文"""
        return re.sub(r'\s+', ' ', text.replace('\r', '').replace('\n', ' ')).strip()

    def _decode_header_value(self, value):
        """解码邮件头中的编码字段"""
        if not value:
            return ""

        decoded_parts = []
        try:
            parts = decode_header(value)
            for content, charset in parts:
                if isinstance(content, bytes):
                    charset = charset or 'utf-8'
                    try:
                        decoded_parts.append(content.decode(charset, errors='replace'))
                    except:
                        decoded_parts.append(content.decode('utf-8', errors='replace'))
                else:
                    decoded_parts.append(content)
        except:
            return value

        return ''.join(decoded_parts)

    def _extract_email_address(self, from_header):
        """从From头中提取邮箱地址"""
        if not from_header:
            return ""

        # 匹配邮箱地址
        match = re.search(r'<([^>]+)>', from_header)
        if match:
            return match.group(1)

        # 如果没有<>，直接检查是否是邮箱格式
        match = re.search(r'[\w\.-]+@[\w\.-]+', from_header)
        if match:
            return match.group(0)

        return from_header

    def _parse_date(self, date_str):
        """解析邮件日期"""
        if not date_str:
            return ""

        try:
            # 尝试解析多种日期格式
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%d %b %Y %H:%M:%S',
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip(), fmt)
                    return dt.isoformat()
                except:
                    continue

            # 尝试email.utils解析
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt.isoformat()

        except:
            return ""

    def close(self):
        """关闭连接"""
        if self._imap_conn:
            try:
                self._imap_conn.close()
                self._imap_conn.logout()
            except:
                pass
            self._imap_conn = None

        if self._smtp_conn:
            try:
                self._smtp_conn.quit()
            except:
                pass
            self._smtp_conn = None

    def mark_as_read(self, email_id: str) -> bool:
        """
        将指定邮件标记为已读

        @param email_id: 邮件ID
        @return: 是否成功
        """
        try:
            mail = self._get_imap_connection()
            mail.select('INBOX')
            # 使用 \\Seen flag 标记为已读
            status, _ = mail.store(email_id, '+FLAGS', '\\Seen')
            return status == 'OK'
        except Exception as e:
            print(f"[DEBUG] Mark as read failed: {e}")
            return False
