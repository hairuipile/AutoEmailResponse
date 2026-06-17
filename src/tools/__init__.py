"""
邮件工具模块

提供 QQ 邮箱、163 邮箱和 Gmail 的操作工具
"""

from .QQEmailTools import QQEmailToolsClass
from .QQMailTools import QQMailTools
from .GmailTools import GmailToolsClass
from .NeteaseEmailTools import NeteaseEmailTools

__all__ = ["QQEmailToolsClass", "QQMailTools", "GmailToolsClass", "NeteaseEmailTools"]
