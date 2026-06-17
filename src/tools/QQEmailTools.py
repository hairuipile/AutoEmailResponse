import os
import re
import uuid
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import imaplib
import smtplib


class QQEmailToolsClass:
    def __init__(self):
        self.imap_server = os.getenv("QQ_IMAP_SERVER", "imap.qq.com")
        self.imap_port = int(os.getenv("QQ_IMAP_PORT", 993))
        self.smtp_server = os.getenv("QQ_SMTP_SERVER", "smtp.qq.com")
        self.smtp_port = int(os.getenv("QQ_SMTP_PORT", 465))
        self.email = os.getenv("MY_EMAIL")
        self.auth_code = os.getenv("QQ_EMAIL_AUTH_CODE")

    def fetch_unanswered_emails(self, max_results=50):
        """
        Fetches recent emails from QQ mailbox via IMAP.
        """
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mail.login(self.email, self.auth_code)
            mail.select("INBOX")

            now = datetime.now()
            delay = now - timedelta(hours=8)
            date_str = delay.strftime("%d-%b-%Y")
            status, message_data = mail.search(None, f"(SINCE {date_str})")
            if status != "OK" or not message_data[0]:
                mail.logout()
                return []

            email_ids = message_data[0].split()
            recent_ids = email_ids[-max_results:] if len(email_ids) > max_results else email_ids

            results = []
            for eid in recent_ids:
                status, msg_data = mail.fetch(eid, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                email_info = self._parse_email(msg, eid.decode())
                results.append(email_info)

            mail.logout()
            return results

        except Exception as error:
            print(f"An error occurred while fetching emails: {error}")
            return []

    def create_draft_reply(self, initial_email, reply_text):
        """
        QQ邮箱没有原生草稿接口，这里将回复保存到本地 drafts 目录作为 .eml 文件。
        """
        try:
            drafts_dir = os.path.join(os.getcwd(), "drafts")
            os.makedirs(drafts_dir, exist_ok=True)
            message = self._create_reply_message(initial_email, reply_text)
            subject = message["subject"]
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.eml"
            filepath = os.path.join(drafts_dir, filename)
            with open(filepath, "wb") as f:
                f.write(message.as_bytes())
            print(f"Draft saved to {filepath}")
            return {"draft_path": filepath}
        except Exception as error:
            print(f"An error occurred while creating draft: {error}")
            return None

    def send_reply(self, initial_email, reply_text):
        """
        Sends a reply via QQ SMTP.
        """
        try:
            message = self._create_reply_message(initial_email, reply_text, send=True)
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.email, self.auth_code)
                server.sendmail(self.email, initial_email.sender, message.as_bytes())
            print("Reply sent successfully")
            return {"status": "sent"}
        except Exception as error:
            print(f"An error occurred while sending reply: {error}")
            return None

    def _create_reply_message(self, email, reply_text, send=False):
        message = MIMEMultipart("alternative")
        message["to"] = email.sender
        message["subject"] = f"Re: {email.subject}" if not email.subject.startswith("Re: ") else email.subject

        text_part = MIMEText(reply_text, "plain", "utf-8")
        html_text = reply_text.replace("\n", "<br>").replace("\\n", "<br>")
        html_part = MIMEText(
            f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{html_text}</body></html>",
            "html",
            "utf-8",
        )
        message.attach(text_part)
        message.attach(html_part)

        if send:
            message["Message-ID"] = f"<{uuid.uuid4()}@qq.com>"
        return message

    def _parse_email(self, msg, eid):
        headers = {name.lower(): self._decode_header(value) for name, value in msg.items()}
        subject = headers.get("subject", "")
        sender = headers.get("from", "")
        message_id = headers.get("message-id", "")
        references = headers.get("references", "")
        date_str = headers.get("date", "")
        occurred_at = self._parse_date(date_str)

        body = self._get_email_body(msg)
        thread_id = self._extract_thread_id(references, message_id, eid)

        return {
            "id": eid,
            "threadId": thread_id,
            "messageId": message_id,
            "references": references,
            "sender": sender,
            "subject": subject,
            "body": body,
            "occurred_at": occurred_at,
        }

    @staticmethod
    def _decode_header(raw):
        if raw is None:
            return ""
        parts = decode_header(raw)
        decoded = []
        for part, charset in parts:
            if isinstance(part, bytes):
                decoded.append(part.decode(charset or "utf-8", errors="ignore"))
            else:
                decoded.append(part)
        return "".join(decoded)

    @staticmethod
    def _parse_date(date_str):
        if not date_str:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            dt = parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _extract_thread_id(references, message_id, fallback):
        if references:
            return references.split()[-1]
        return message_id or fallback

    @staticmethod
    def _get_email_body(msg):
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(part.get_content_charset() or "utf-8", errors="ignore").strip()
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/html" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(part.get_content_charset() or "utf-8", errors="ignore").strip()
            return ""
        payload = msg.get_payload(decode=True)
        if not payload:
            return ""
        return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore").strip()
