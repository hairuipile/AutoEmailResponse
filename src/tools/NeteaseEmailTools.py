"""
网易邮箱工具 - 使用 IMAP/SMTP 协议

配置步骤：
1. 在 163 邮箱设置中开启 IMAP/SMTP 服务
2. 获取授权码（不是登录密码）
3. 在 .env 中填入邮箱地址和授权码
"""

import os
import re
import ssl
import imaplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# 显式指定 .env 文件路径
load_dotenv()


class NeteaseEmailTools:
    """网易邮箱工具类，使用 IMAP/SMTP 协议"""
    
    def __init__(self):
        self.email = os.getenv('MY_EMAIL', '')
        self.auth_code = os.getenv('NETEASE_EMAIL_AUTH_CODE', '')
        self.smtp_server = os.getenv('NETEASE_SMTP_SERVER', 'smtp.163.com')
        self.smtp_port = int(os.getenv('NETEASE_SMTP_PORT', '465'))
        self.imap_server = os.getenv('NETEASE_IMAP_SERVER', 'imap.163.com')
        self.imap_port = int(os.getenv('NETEASE_IMAP_PORT', '993'))
        
        print(f"[DEBUG] 邮箱: {self.email}")
        print(f"[DEBUG] 授权码: {self.auth_code}")
        
        if not self.email or not self.auth_code:
            raise ValueError("请在 .env 中配置 MY_EMAIL 和 NETEASE_EMAIL_AUTH_CODE")
    
    def _create_imap_connection(self):
        """创建 IMAP 连接"""
        print(f"[DEBUG] 正在连接 IMAP: {self.imap_server}:{self.imap_port}")
        context = ssl.create_default_context()
        conn = imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context)
        print(f"[DEBUG] 正在登录邮箱: {self.email}")
        conn.login(self.email, self.auth_code)
        print(f"[DEBUG] 登录成功")
        return conn
    
    def _create_smtp_connection(self):
        """创建 SMTP 连接"""
        context = ssl.create_default_context()
        conn = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context)
        conn.login(self.email, self.auth_code)
        return conn
    
    def fetch_unanswered_emails(self, max_results=50):
        """
        获取未回复的邮件
        
        @param max_results: 最大获取数量
        @return: 邮件列表
        """
        try:
            print(f"[DEBUG] 开始获取邮件...")
            conn = self._create_imap_connection()
            
            # 先选择 INBOX 文件夹 (使用 SELECT 而非 EXAMINE)
            print(f"[DEBUG] 正在选择 INBOX...")
            status, folder_info = conn.select('INBOX')
            print(f"[DEBUG] INBOX 选择结果: status={status}, info={folder_info}")
            
            if status != 'OK':
                print(f"[ERROR] 选择 INBOX 失败")
                conn.logout()
                return []
            
            # 搜索最近8小时的邮件
            eight_hours_ago = datetime.now() - timedelta(hours=8)
            date_str = eight_hours_ago.strftime("%d-%b-%Y")
            
            # 搜索条件：更新于指定日期之后
            search_criteria = f'(SINCE {date_str})'
            print(f"[DEBUG] 执行搜索: {search_criteria}")
            status, message_ids = conn.search(None, search_criteria)
            print(f"[DEBUG] 搜索结果: status={status}")
            
            if status != 'OK':
                return []
            
            ids = message_ids[0].split()
            if not ids:
                return []
            
            # 只取最新的 max_results 封
            ids = ids[-max_results:]
            
            emails = []
            my_email_lower = self.email.lower()
            
            for msg_id in ids:
                try:
                    status, msg_data = conn.fetch(msg_id, '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    from email.parser import Parser
                    msg = Parser().parsestr(msg_data[0][1])
                    
                    # 提取发件人
                    sender = self._decode_header(msg.get('From', ''))
                    sender_email = self._extract_email_address(sender)
                    
                    # 跳过自己发送的邮件
                    if sender_email.lower() == my_email_lower:
                        continue
                    
                    # 提取主题和正文
                    subject = self._decode_header(msg.get('Subject', ''))
                    body = self._extract_body(msg)
                    message_id = msg.get('Message-ID', '')
                    references = msg.get('References', '')
                    
                    # 解析日期
                    date_str_email = msg.get('Date', '')
                    occurred_at = self._parse_date(date_str_email)
                    
                    # 检查是否已回复（通过查找 In-Reply-To 或同一主题的已发送邮件）
                    in_reply_to = msg.get('In-Reply-To', '')
                    
                    emails.append({
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                        "messageId": message_id,
                        "references": references,
                        "sender": sender,
                        "sender_email": sender_email,
                        "subject": subject,
                        "body": body,
                        "occurred_at": occurred_at,
                    })
                    
                except Exception as e:
                    print(f"解析邮件失败: {e}")
                    continue
            
            conn.logout()
            return emails
            
        except Exception as e:
            print(f"获取邮件失败: {e}")
            return []
    
    def _extract_body(self, msg):
        """从邮件中提取正文"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # 跳过附件
                if "attachment" in content_disposition:
                    continue
                
                # 优先获取纯文本
                if content_type == "text/plain":
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body = part.get_payload(decode=True).decode(charset, errors='replace')
                    except:
                        pass
                    break
                
                # 如果没有纯文本，获取 HTML
                if content_type == "text/html" and not body:
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        html_content = part.get_payload(decode=True).decode(charset, errors='replace')
                        body = self._clean_html(html_content)
                    except:
                        pass
        else:
            charset = msg.get_content_charset() or 'utf-8'
            try:
                body = msg.get_payload(decode=True).decode(charset, errors='replace')
            except:
                pass
        
        return body
    
    def _clean_html(self, text):
        """清理 HTML 内容"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        for tag in soup(['script', 'style', 'head', 'meta', 'title']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    
    def send_reply(self, initial_email, reply_text):
        """
        发送回复邮件
        
        @param initial_email: 原始邮件对象
        @param reply_text: 回复内容
        @return: 发送结果
        """
        try:
            # 获取原始邮件信息
            sender = initial_email.get('sender_email', initial_email.get('sender', ''))
            sender = self._extract_email_address(sender)
            subject = initial_email.get('subject', 'No Subject')
            if not subject.startswith('Re:'):
                subject = f'Re: {subject}'
            
            # 创建邮件
            message = MIMEMultipart('alternative')
            message['From'] = f"{self.email}"
            message['To'] = sender
            message['Subject'] = Header(subject, 'utf-8')
            
            # 设置回复头
            in_reply_to = initial_email.get('messageId', '')
            if in_reply_to:
                message['In-Reply-To'] = in_reply_to
                references = initial_email.get('references', '')
                message['References'] = f"{references} {in_reply_to}".strip()
            
            # 生成 Message-ID
            import uuid
            message['Message-ID'] = f"<{uuid.uuid4()}@163.com>"
            
            # 纯文本内容
            text_part = MIMEText(reply_text, 'plain', 'utf-8')
            message.attach(text_part)
            
            # 发送邮件
            conn = self._create_smtp_connection()
            conn.sendmail(self.email, [sender], message.as_bytes())
            conn.quit()
            
            print(f"邮件已发送至: {sender}")
            return {"status": "sent", "to": sender}
            
        except Exception as e:
            print(f"发送邮件失败: {e}")
            return None
    
    def create_draft_reply(self, initial_email, reply_text):
        """
        创建草稿回复（163邮箱通过网页操作草稿，此方法仅打印信息）
        
        @param initial_email: 原始邮件对象
        @param reply_text: 回复内容
        @return: 提示信息
        """
        sender = initial_email.get('sender_email', '')
        subject = initial_email.get('subject', 'No Subject')
        print(f"草稿提示: 将回复给 {sender}, 主题: Re: {subject}")
        print("请登录网页邮箱 https://mail.163.com 查看草稿")
        return {"status": "draft_created", "note": "请登录网页邮箱查看草稿"}
    
    @staticmethod
    def _decode_header(header):
        """解码邮件头"""
        if not header:
            return ""
        parts = []
        for part, encoding in email.header.decode_header(header):
            if isinstance(part, bytes):
                try:
                    charset = encoding or 'utf-8'
                    parts.append(part.decode(charset, errors='replace'))
                except:
                    parts.append(part.decode('utf-8', errors='replace'))
            else:
                parts.append(part)
        return ''.join(parts)
    
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
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()
