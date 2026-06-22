"""
Shared entity extraction — consolidated regex patterns for order IDs, phone numbers,
tracking numbers, dates, amounts, and email addresses.

Used by: intent_recognizer, context_manager.
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExtractedEntity:
    """A single entity extracted from user text."""
    name: str
    value: str
    entity_type: str          # order_id, phone_number, tracking_number, date, amount, email
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0
    normalized_value: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.entity_type,
            "confidence": round(self.confidence, 4),
            "normalized": self.normalized_value or self.value,
        }


# ---- Pattern definitions ----

def _match_order_ids(text: str) -> List[ExtractedEntity]:
    entities = []
    for pattern in [
        r"(?:订单|#|No\.)\s*(\d{6,20})",
        r"(?:order[_\-\s]?)(\d{6,20})",
        r"[A-Z]{2,4}\d{6,12}",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                name="订单号", value=m.group(0),
                entity_type="order_id", confidence=0.90,
                start_pos=m.start(), end_pos=m.end(),
            ))
    return entities


def _match_phone_numbers(text: str) -> List[ExtractedEntity]:
    entities = []
    for pattern in [r"1[3-9]\d{9}", r"\d{3,4}-\d{7,8}"]:
        for m in re.finditer(pattern, text):
            entities.append(ExtractedEntity(
                name="手机号", value=m.group(0),
                entity_type="phone_number", confidence=0.95,
                start_pos=m.start(), end_pos=m.end(),
            ))
    return entities


def _match_tracking_numbers(text: str) -> List[ExtractedEntity]:
    entities = []
    for pattern in [
        r"(?:SF|YT|YD|DB|ZTO|STO)\d{8,15}",
        r"[A-Z]{2}\d{8,12}",
        r"SF\d{10,15}",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                name="快递单号", value=m.group(0),
                entity_type="tracking_number", confidence=0.90,
                start_pos=m.start(), end_pos=m.end(),
            ))
    return entities


def _match_dates(text: str) -> List[ExtractedEntity]:
    entities = []
    for pattern in [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}月\d{1,2}[日号]",
        r"(?:今天|明天|昨天|前天|大前天)",
    ]:
        for m in re.finditer(pattern, text):
            entities.append(ExtractedEntity(
                name="日期", value=m.group(0),
                entity_type="date", confidence=0.90,
                start_pos=m.start(), end_pos=m.end(),
                normalized_value=m.group(0).replace("/", "-"),
            ))
    return entities


def _match_amounts(text: str) -> List[ExtractedEntity]:
    entities = []
    for m in re.finditer(r"(\d+\.?\d*)\s*(元|块|美元|USD|CNY|¥)", text, re.IGNORECASE):
        entities.append(ExtractedEntity(
            name="金额", value=m.group(0),
            entity_type="amount", confidence=0.85,
            start_pos=m.start(), end_pos=m.end(),
            normalized_value=m.group(1),
        ))
    return entities


def _match_emails(text: str) -> List[ExtractedEntity]:
    entities = []
    for m in re.finditer(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        entities.append(ExtractedEntity(
            name="邮箱", value=m.group(0),
            entity_type="email", confidence=0.98,
            start_pos=m.start(), end_pos=m.end(),
        ))
    return entities


def extract_entities(text: str) -> List[ExtractedEntity]:
    """Extract all known entity types from text in a single pass."""
    if not text:
        return []
    return (
        _match_order_ids(text)
        + _match_phone_numbers(text)
        + _match_tracking_numbers(text)
        + _match_dates(text)
        + _match_amounts(text)
        + _match_emails(text)
    )
