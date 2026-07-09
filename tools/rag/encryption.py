"""RAG 数据加密模块 —— 使用 Fernet 对称加密保护本地文档原文

设计：
  - 透明加密：向量检索层无感知，在存入/读取 Chroma 时自动加解密
  - 密钥管理：首次运行自动生成密钥，存储在 .rag/ 目录（受文件权限保护）
  - 可选开关：通过 RAG_ENCRYPTION_ENABLED=true 启用

使用方式（在 vectorstore.py 中集成）：
  store → encrypt_text(txt) → Chroma.documents
  retrieve → Chroma.documents → decrypt_text(enc) → 明文
"""

import os
import json
import base64
import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

# 密钥文件路径
_KEY_FILE = config.RAG_DIR / ".encryption_key"


def _get_fernet():
    """获取 Fernet 实例（懒加载 + 自动生成密钥）"""
    from cryptography.fernet import Fernet

    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text().strip()
    else:
        # 首次运行：生成密钥并保存
        key = Fernet.generate_key().decode()
        config.RAG_DIR.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(key, encoding="utf-8")
        # Windows 上设置文件为只读（防止误删）
        try:
            _KEY_FILE.chmod(0o600)
        except Exception:
            pass
        logger.info(f"生成加密密钥: {_KEY_FILE}")

    return Fernet(key.encode())


def is_enabled() -> bool:
    """检查是否开启了加密"""
    return os.getenv("RAG_ENCRYPTION_ENABLED", "false").lower() in ("true", "1", "yes")


def encrypt_text(plain_text: str) -> str:
    """加密文本 → base64 编码字符串"""
    if not plain_text:
        return plain_text
    f = _get_fernet()
    encrypted = f.encrypt(plain_text.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_text(cipher_b64: str) -> str:
    """base64 编码的密文 → 解密文本"""
    if not cipher_b64:
        return cipher_b64
    try:
        f = _get_fernet()
        encrypted = base64.b64decode(cipher_b64.encode("utf-8"))
        return f.decrypt(encrypted).decode("utf-8")
    except Exception as e:
        logger.warning(f"解密失败（可能未加密）: {e}")
        return cipher_b64  # 兼容未加密的历史数据


def encrypt_documents(texts: list) -> list:
    """批量加密文档列表"""
    if not is_enabled():
        return texts
    return [encrypt_text(t) if t else t for t in texts]


def decrypt_documents(texts: list) -> list:
    """批量解密文档列表"""
    if not is_enabled():
        return texts
    return [decrypt_text(t) if t else t for t in texts]


def key_exists() -> bool:
    """检查密钥是否存在"""
    return _KEY_FILE.exists()


def export_key() -> str:
    """导出密钥（用于备份）"""
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()
    return ""


def import_key(key_b64: str) -> bool:
    """导入密钥（用于恢复）"""
    from cryptography.fernet import Fernet
    try:
        # 验证密钥格式
        Fernet(key_b64.encode())
        config.RAG_DIR.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(key_b64, encoding="utf-8")
        logger.info(f"密钥已导入: {_KEY_FILE}")
        return True
    except Exception as e:
        logger.error(f"密钥导入失败: {e}")
        return False