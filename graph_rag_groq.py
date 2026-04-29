#!/usr/bin/env python3
"""
Graph RAG System - Excel + PDF Integration with Relationships
Fully powered by Groq API (no Anthropic/Claude dependencies).
"""

import os
import logging
import pandas as pd
import numpy as np
import json
import time
import fitz  # PyMuPDF
import re
import hashlib
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from unicodedata import normalize as ucnorm
from groq import Groq

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config:
    CHUNK_SIZE = 1200
    OVERLAP = 250
    BM25_K1 = 1.5
    BM25_B = 0.75
    GROQ_TIMEOUT_SECONDS = 15
    GROQ_MAX_RETRIES = 3
    GROQ_MODEL       = "llama-3.1-8b-instant"
    GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"
    GROQ_RATE_LIMIT_DELAY = 2.0   # ← ADD THIS (seconds between calls)
    CACHE_DIR = "cache"
    CACHE_VERSION = "1.1"


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------
class CacheManager:
    """Manages caching for processed documents to avoid reprocessing."""

    def __init__(self, cache_dir: str = Config.CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        try:
            stat = os.stat(file_path)
            content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except OSError:
            return None

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("version") == Config.CACHE_VERSION:
                    logger.info(f"✅ Using cached data for {cache_key}")
                    return cached.get("data")
                else:
                    logger.info(f"🔄 Cache version mismatch for {cache_key}, rebuilding...")
                    os.remove(cache_path)
            except Exception as e:
                logger.warning(f"⚠️ Error loading cache for {cache_key}: {e}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        return None

    def set_cached_data(self, cache_key: str, data: Any) -> None:
        cache_path = self._get_cache_path(cache_key)
        try:
            payload = {
                "version": Config.CACHE_VERSION,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
            logger.info(f"💾 Cached data for {cache_key}")
        except Exception as e:
            logger.warning(f"⚠️ Error caching data for {cache_key}: {e}")

    def get_file_cache_key(self, file_path: str) -> str:
        file_hash = self._get_file_hash(file_path)
        return f"file_{file_hash}" if file_hash else f"file_{os.path.basename(file_path)}"

    def get_directory_cache_key(self, dir_path: str) -> str:
        file_hashes = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                fh = self._get_file_hash(os.path.join(root, file))
                if fh:
                    file_hashes.append(fh)
        if file_hashes:
            combined = hashlib.md5("".join(sorted(file_hashes)).encode()).hexdigest()
            return f"dir_{combined}"
        return f"dir_{os.path.basename(dir_path)}"

    def clear_cache(self) -> None:
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("🗑️ Cache cleared")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing cache: {e}")


# ---------------------------------------------------------------------------
# Generic File Processor  (Groq-only)
# ---------------------------------------------------------------------------
class GenericFileProcessor:
    """Processes any supported file type; uses Groq for AI-powered extraction."""

    def __init__(self, groq_api_key: Optional[str] = None, use_ai: bool = True):
        self.use_ai = use_ai and bool(groq_api_key)
        self.client: Optional[Groq] = Groq(api_key=groq_api_key) if self.use_ai else None

        self.supported_extensions: Dict[str, Any] = {
            ".pdf":  self._process_pdf,
            ".txt":  self._process_text,
            ".docx": self._process_docx,
            ".doc":  self._process_doc,
            ".xlsx": self._process_excel,
            ".xls":  self._process_excel,
            ".csv":  self._process_csv,
            ".json": self._process_json,
            ".xml":  self._process_xml,
            ".html": self._process_html,
            ".md":   self._process_markdown,
            ".rtf":  self._process_rtf,
            ".odt":  self._process_odt,
        }

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def process_file(self, file_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        if ext in self.supported_extensions:
            try:
                logger.info(f"  📄 Processing {filename} as {ext} ...")
                return self.supported_extensions[ext](file_path)
            except Exception as e:
                logger.warning(f"  ⚠️ Error processing {filename}: {e}")
                return self._process_generic(file_path)
        logger.info(f"  📄 Processing {filename} as generic file...")
        return self._process_generic(file_path)

    # ✅ NEW - always use fast regex for indexing, Groq only for queries
    def detect_document_type(self, text: str, filename: str, file_type: str = None) -> str:
        # Always use fast pattern matching during indexing - no API calls needed
        return self._detect_document_type_simple(text, filename, file_type)

    def extract_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        # Always use fast regex during indexing - no API calls needed
        return self._extract_basic_fields(text, doc_type)

    # ------------------------------------------------------------------ #
    # File-type processors
    # ------------------------------------------------------------------ #

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(file_path)
            parts = []
            for i, page in enumerate(doc):
                t = page.get_text()
                if t.strip():
                    parts.append(f"Page {i+1}:\n{t}")
            doc.close()
            text = "\n\n".join(parts)
            return {"content": text, "file_type": "pdf",
                    "metadata": {"pages": len(parts), "total_chars": len(text)}}
        except Exception as e:
            return {"content": "", "file_type": "pdf", "metadata": {"error": str(e)}}

    def _process_text(self, file_path: str) -> Dict[str, Any]:
        for enc in ("utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                return {"content": content, "file_type": "text",
                        "metadata": {"total_chars": len(content), "encoding": enc}}
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return {"content": "", "file_type": "text", "metadata": {"error": str(e)}}
        return {"content": "", "file_type": "text", "metadata": {"error": "encoding failure"}}

    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        return {"content": "", "file_type": "excel",
                "metadata": {"note": "Processed separately via load_excel_data"}}

    def _process_csv(self, file_path: str) -> Dict[str, Any]:
        return {"content": "", "file_type": "csv",
                "metadata": {"note": "Processed separately via load_excel_data"}}

    def _process_json(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = json.dumps(data, indent=2)
            return {"content": content, "file_type": "json",
                    "metadata": {"structure": data, "total_chars": len(content)}}
        except Exception as e:
            return {"content": "", "file_type": "json", "metadata": {"error": str(e)}}

    def _process_xml(self, file_path: str) -> Dict[str, Any]:
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            content = ET.tostring(root, encoding="unicode")
            return {"content": content, "file_type": "xml",
                    "metadata": {"root_tag": root.tag, "total_chars": len(content)}}
        except Exception as e:
            return {"content": "", "file_type": "xml", "metadata": {"error": str(e)}}

    def _process_html(self, file_path: str) -> Dict[str, Any]:
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()
            return {"content": text, "file_type": "html",
                    "metadata": {"title": soup.title.string if soup.title else None,
                                 "total_chars": len(text)}}
        except ImportError:
            return self._process_text(file_path)
        except Exception as e:
            return {"content": "", "file_type": "html", "metadata": {"error": str(e)}}

    def _process_markdown(self, file_path: str) -> Dict[str, Any]:
        return self._process_text(file_path)

    def _process_docx(self, file_path: str) -> Dict[str, Any]:
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs]
            content = "\n".join(paragraphs)
            return {"content": content, "file_type": "docx",
                    "metadata": {"paragraphs": len(paragraphs), "total_chars": len(content)}}
        except ImportError:
            return {"content": "", "file_type": "docx",
                    "metadata": {"error": "python-docx not installed"}}
        except Exception as e:
            return {"content": "", "file_type": "docx", "metadata": {"error": str(e)}}

    def _process_doc(self, file_path: str) -> Dict[str, Any]:
        try:
            import subprocess
            result = subprocess.run(["antiword", file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return {"content": result.stdout, "file_type": "doc",
                        "metadata": {"total_chars": len(result.stdout)}}
        except Exception:
            pass
        return {"content": "", "file_type": "doc",
                "metadata": {"error": "DOC processing requires antiword"}}

    def _process_rtf(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            clean = re.sub(r"\\[a-z]+\d*\s?", "", content)
            clean = re.sub(r"[{}]", "", clean)
            return {"content": clean, "file_type": "rtf",
                    "metadata": {"total_chars": len(clean)}}
        except Exception as e:
            return {"content": "", "file_type": "rtf", "metadata": {"error": str(e)}}

    def _process_odt(self, file_path: str) -> Dict[str, Any]:
        try:
            import zipfile, xml.etree.ElementTree as ET
            with zipfile.ZipFile(file_path, "r") as odt:
                xml_data = odt.read("content.xml")
            root = ET.fromstring(xml_data)
            parts = [elem.text for elem in root.iter() if elem.text]
            content = " ".join(parts)
            return {"content": content, "file_type": "odt",
                    "metadata": {"total_chars": len(content)}}
        except Exception as e:
            return {"content": "", "file_type": "odt", "metadata": {"error": str(e)}}

    def _process_generic(self, file_path: str) -> Dict[str, Any]:
        for enc in ("utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                return {"content": content, "file_type": "text",
                        "metadata": {"total_chars": len(content), "encoding": enc}}
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return {"content": f"[Binary file – {len(data)} bytes]", "file_type": "binary",
                    "metadata": {"size": len(data)}}
        except Exception as e:
            return {"content": "", "file_type": "unknown", "metadata": {"error": str(e)}}

    # ------------------------------------------------------------------ #
    # Groq-powered AI helpers
    # ------------------------------------------------------------------ #

    # ✅ FIXED - with rate limit delay
def _groq_complete(self, prompt: str, model: str = None, max_tokens: int = 512) -> str:
    model = model or Config.GROQ_MODEL
    for attempt in range(Config.GROQ_MAX_RETRIES):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    self.client.chat.completions.create,
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=max_tokens,
                )
                resp = future.result(timeout=Config.GROQ_TIMEOUT_SECONDS)
            time.sleep(Config.GROQ_RATE_LIMIT_DELAY)   # ← ADD THIS
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # Check specifically for 429 rate limit
            if "429" in str(e):
                wait = Config.GROQ_RATE_LIMIT_DELAY * (attempt + 2)
                logger.warning(f"  ⏳ Rate limited. Waiting {wait}s…")
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                logger.warning(f"  ⚠️ Groq attempt {attempt+1} failed: {e}. Retrying in {wait}s…")
                time.sleep(wait)
    raise RuntimeError("Groq API call failed after all retries")

    def _detect_type_groq(self, text: str, filename: str, file_type: str) -> str:
        prompt = (
            f"Filename: {filename}\nFile type: {file_type or 'unknown'}\n"
            f"Content (first 2000 chars):\n{text[:2000]}\n\n"
            "Classify this document into ONE specific category "
            "(e.g. invoice, contract, manual, report, receipt, letter, statement, "
            "quotation, purchase_order, delivery_note, certificate, email, log, config, data). "
            "Reply with ONLY the category name, nothing else."
        )
        try:
            raw = self._groq_complete(prompt)
            doc_type = re.sub(r"[^\w\s-]", "", raw).replace(" ", "_").lower()
            return doc_type or "document"
        except Exception as e:
            logger.error(f"Groq detect_type failed: {e}")
            return self._detect_document_type_simple(text, filename, file_type)

    def _extract_groq(self, text: str, doc_type: str) -> Dict[str, Any]:
        prompt = (
            f"Extract ALL structured data from this {doc_type} document.\n\n"
            f"Document:\n{text[:3000]}\n\n"
            "Return ONLY a valid JSON object with descriptive field names. "
            "Use null for missing fields. No markdown, no backticks, just raw JSON."
        )
        for attempt in range(Config.GROQ_MAX_RETRIES):
            try:
                raw = self._groq_complete(prompt, max_tokens=1000)
                # Strip any accidental fences
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                return json.loads(clean)
            except json.JSONDecodeError:
                logger.warning(f"  ⚠️ JSON parse failed on Groq extraction attempt {attempt+1}")
            except Exception as e:
                logger.warning(f"  ⚠️ Groq extraction attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        logger.info("  🔄 Falling back to basic field extraction…")
        return self._extract_basic_fields(text, doc_type)

    # ------------------------------------------------------------------ #
    # Fallback pattern-based helpers
    # ------------------------------------------------------------------ #

    def _detect_document_type_simple(self, text: str, filename: str,
                                      file_type: str = None) -> str:
        text_lower = text.lower()
        fname_lower = filename.lower()
        patterns = {
            "invoice":      ["invoice", "bill", "inv-", "receipt"],
            "contract":     ["contract", "agreement", "terms", "conditions"],
            "report":       ["report", "summary", "analysis", "findings"],
            "manual":       ["manual", "guide", "instructions", "tutorial"],
            "policy":       ["policy", "procedure", "guidelines"],
            "letter":       ["letter", "correspondence"],
            "statement":    ["statement", "account", "balance"],
            "receipt":      ["receipt", "payment", "confirmation"],
            "quotation":    ["quotation", "quote", "estimate"],
            "order":        ["order", "purchase", "po-"],
            "delivery":     ["delivery", "shipping", "dispatch"],
            "certificate":  ["certificate", "certification", "license"],
            "application":  ["application", "form", "request"],
            "proposal":     ["proposal", "bid", "tender"],
            "memo":         ["memo", "memorandum"],
            "email":        ["email", "message"],
            "log":          ["log", "record", "entry", "timestamp"],
            "config":       ["config", "configuration", "settings"],
            "data":         ["data", "dataset", "export"],
        }
        for dtype, kws in patterns.items():
            if any(k in fname_lower for k in kws):
                return dtype
        for dtype, kws in patterns.items():
            if any(k in text_lower for k in kws):
                return dtype
        if file_type in ("json", "xml"):
            return "data"
        if file_type in ("html", "markdown"):
            return "web_content"
        if file_type == "csv":
            return "spreadsheet"
        return "document"

    def _extract_basic_fields(self, text: str, doc_type: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        patterns_map = {
            "document_number": [
                r"(?:GRN|Invoice|PO|PI|Quote|Delivery)\s*(?:No|Number|Code)[:\s]*([A-Z0-9\-]+)",
                r"Reference[:\s]*([A-Z0-9\-]+)",
            ],
            "date": [
                r"(?:Date|Dated)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                r"(\d{1,2}\s+\w+\s+\d{4})",
            ],
            "amount": [
                r"(?:Total|Amount|Sum)[:\s]*(\d+\.?\d*)",
                r"(\$\d+\.?\d*)",
                r"(?:Rs\.?|INR)[:\s]*(\d+\.?\d*)",
            ],
            "supplier_name": [
                r"(?:Supplier|Vendor|From)[:\s]*([A-Za-z\s&.,]+)",
            ],
            "customer_name": [
                r"(?:Customer|Buyer|To)[:\s]*([A-Za-z\s&.,]+)",
                r"Bill\s+To[:\s]*([A-Za-z\s&.,]+)",
            ],
        }
        for field, pats in patterns_map.items():
            for pat in pats:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    data[field] = m.group(1).strip()
                    break
        return data


# ---------------------------------------------------------------------------
# Graph Relationship Builder
# ---------------------------------------------------------------------------
class GraphRelationshipBuilder:

    def __init__(self):
        self.relationships: List[Dict] = []
        self.entity_map: Dict[str, Dict] = {}
        self.document_connections: Dict = {}

    # -------------------------------- helpers ------------------------------- #

    @staticmethod
    def _normalize_name(value: Any) -> str:
        if value is None:
            return ""
        s = str(value)
        s = ucnorm("NFKC", s.replace("\xa0", " "))
        return re.sub(r"\s+", " ", s).strip().lower()

    # -------------------------------- entities ------------------------------ #

    def extract_entities_from_excel(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        entities: List[Dict] = []
        field_mapping = self._create_dynamic_field_mapping(list(df.columns))

        for idx, row in df.iterrows():
            for col_name, col_value in row.items():
                if pd.isna(col_value) or not str(col_value).strip():
                    continue
                entity_type = field_mapping.get(col_name,
                    col_name.lower().replace(" ", "_").replace(".", ""))
                eid = f"excel_{entity_type}_{idx}_{str(col_value).replace(' ', '_')}"
                entities.append({
                    "id": eid,
                    "name": str(col_value),
                    "type": entity_type,
                    "source": "excel",
                    "row_index": idx,
                    "attributes": {
                        "source_file": row.get("_source_file", "excel_data"),
                        "row_index": idx,
                        "column": col_name,
                    },
                })
                self.entity_map[eid] = {
                    "name": str(col_value),
                    "type": entity_type,
                    "source": "excel",
                    "row_index": idx,
                    "original_column": col_name,
                }
        return entities

    def _create_dynamic_field_mapping(self, columns: List[str]) -> Dict[str, str]:
        patterns = {
            "grn_code":        ["grn", "goods receipt"],
            "po_number":       ["purchase order", "po"],
            "invoice_number":  ["invoice", "inv", "sales inv"],
            "supplier_name":   ["supplier", "vendor"],
            "customer_name":   ["customer", "buyer", "client"],
            "book_title":      ["book title", "title", "product"],
            "author":          ["author", "writer"],
            "isbn":            ["isbn"],
            "store_location":  ["store location", "location", "branch"],
            "store_code":      ["store code", "branch code"],
            "amount":          ["amount", "price", "cost", "total"],
            "quantity":        ["quantity", "qty", "count"],
            "date":            ["date", "created", "updated"],
            "status":          ["status", "state"],
            "category":        ["category", "type", "class"],
        }
        mapping: Dict[str, str] = {}
        for col in columns:
            col_lower = col.lower().strip()
            best = next(
                (etype for etype, kws in patterns.items() if any(k in col_lower for k in kws)),
                col_lower.replace(" ", "_").replace(".", "").replace("-", "_"),
            )
            mapping[col] = best
        return mapping

    def extract_entities_from_pdf(self, pdf_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities: List[Dict] = []
        for chunk in pdf_chunks:
            structured = chunk.get("structured_data") or {}
            doc_type = chunk.get("document_type", "unknown")
            source = chunk["source"]
            for field_name, field_value in structured.items():
                if field_value is None or not str(field_value).strip():
                    continue
                eid = (
                    f"pdf_{field_name}_{str(field_value).replace(' ','_').replace('-','_')}"
                )
                etype = self._map_field_to_entity_type(field_name, doc_type)
                entities.append({
                    "id": eid,
                    "name": str(field_value),
                    "type": etype,
                    "source": "pdf",
                    "document_type": doc_type,
                    "source_file": source,
                    "field_name": field_name,
                    "attributes": structured,
                })
                self.entity_map[eid] = {
                    "name": str(field_value),
                    "type": etype,
                    "source": "pdf",
                    "document_type": doc_type,
                    "field_name": field_name,
                }
        return entities

    def _map_field_to_entity_type(self, field_name: str, doc_type: str) -> str:
        mapping = {
            "document_number": "document_number",
            "grn_code": "document_number",
            "invoice_number": "document_number",
            "po_number": "document_number",
            "quote_number": "document_number",
            "date": "date",
            "grn_date": "date",
            "invoice_date": "date",
            "po_date": "date",
            "amount": "amount",
            "total_amount": "amount",
            "supplier_name": "supplier_name",
            "vendor_name": "supplier_name",
            "customer_name": "customer_name",
            "buyer_name": "customer_name",
            "company_name": "company_name",
            "address": "address",
            "phone": "contact_info",
            "email": "contact_info",
            "items": "items",
            "quantity": "quantity",
            "unit_price": "price",
            "tax_amount": "tax",
            "due_date": "date",
            "payment_terms": "terms",
            "shipping_address": "address",
            "billing_address": "address",
            "notes": "notes",
        }
        return mapping.get(field_name, f"{doc_type}_{field_name}")

    # -------------------------------- relationships ------------------------- #

    def build_relationships(
        self,
        excel_entities: List[Dict],
        pdf_entities: List[Dict],
    ) -> List[Dict[str, Any]]:
        relationships: List[Dict] = []

        # Index PDF entities by (type, normalised_name)
        pdf_index: Dict[tuple, List[Dict]] = defaultdict(list)
        for e in pdf_entities:
            pdf_index[(e["type"], self._normalize_name(e["name"]))].append(e)

        # Collect all entity types for similarity lookup
        all_types: set = {e["type"] for e in excel_entities + pdf_entities}

        def similar_types(etype: str) -> List[str]:
            words = set(etype.split("_"))
            return [t for t in all_types if words & set(t.split("_"))]

        for ex_e in excel_entities:
            ex_norm = self._normalize_name(ex_e["name"])
            if not ex_norm:
                continue
            for pdf_type in similar_types(ex_e["type"]):
                for pdf_e in pdf_index.get((pdf_type, ex_norm), []):
                    rtype = self._get_relationship_type(ex_e["type"], pdf_e["type"])
                    strength = self._calc_strength(
                        ex_e["name"], pdf_e["name"], ex_e["type"], pdf_e["type"]
                    )
                    relationships.append({
                        "from": ex_e["id"],
                        "to": pdf_e["id"],
                        "type": rtype,
                        "strength": strength,
                        "description": (
                            f"Excel {ex_e['type']} '{ex_e['name']}' "
                            f"↔ PDF {pdf_e['type']} '{pdf_e['name']}'"
                        ),
                    })

        relationships += self._within_source_rels(excel_entities, "excel", "SAME_ROW")
        relationships += self._within_source_rels(pdf_entities, "pdf", "SAME_DOCUMENT")
        return relationships

    def _get_relationship_type(self, t1: str, t2: str) -> str:
        if t1 == t2:
            return f"SAME_{t1.upper()}"
        common = set(t1.split("_")) & set(t2.split("_"))
        if common:
            return f"SAME_{max(common, key=len).upper()}"
        semantic = {
            ("supplier", "vendor"): "SAME_SUPPLIER",
            ("customer", "buyer"):  "SAME_CUSTOMER",
            ("amount", "total"):    "SAME_AMOUNT",
            ("date", "time"):       "SAME_DATE",
        }
        for (w1, w2), rtype in semantic.items():
            if (w1 in t1 and w2 in t2) or (w2 in t1 and w1 in t2):
                return rtype
        return f"RELATED_{t1.upper()}_{t2.upper()}"

    def _calc_strength(self, n1: str, n2: str, t1: str, t2: str) -> float:
        if n1 == n2:
            return 1.0
        a, b = n1.strip().lower(), n2.strip().lower()
        if a == b:
            return 1.0
        if any(k in (t1 + t2) for k in ("name", "supplier", "customer", "company")):
            wa, wb = set(a.split()), set(b.split())
            if wa and wb:
                return len(wa & wb) / len(wa | wb)
        return 0.0

    def _within_source_rels(
        self, entities: List[Dict], source: str, rel_type: str
    ) -> List[Dict]:
        rels: List[Dict] = []
        groups: Dict[Any, List[Dict]] = defaultdict(list)
        for e in entities:
            key = e.get("row_index", 0) if source == "excel" else e.get("source_file", "?")
            groups[key].append(e)
        for key, group in groups.items():
            if len(group) < 2:
                continue
            hub = next((e for e in group if e.get("type") == "document_number"), group[0])
            for e in group:
                if e["id"] == hub["id"]:
                    continue
                rels.append({
                    "from": hub["id"],
                    "to": e["id"],
                    "type": rel_type,
                    "strength": 1.0,
                    "description": f"Co-located in {source} {key}",
                })
        return rels

    # -------------------------------- graph --------------------------------- #

    def create_network_graph(
        self, entities: List[Dict], relationships: List[Dict]
    ) -> nx.Graph:
        G = nx.Graph()
        for e in entities:
            G.add_node(e["id"], name=e["name"], type=e["type"],
                       source=e["source"], **e.get("attributes", {}))
        for r in relationships:
            G.add_edge(r["from"], r["to"], type=r["type"],
                       strength=r["strength"], description=r["description"])
        return G


# ---------------------------------------------------------------------------
# Graph RAG System  (Groq-only)
# ---------------------------------------------------------------------------
class GraphRAGSystem:
    """Main Graph RAG System — Excel + PDF Integration, fully powered by Groq."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        use_ai: bool = True,
        cache_dir: str = Config.CACHE_DIR,
    ):
        self.groq_api_key = groq_api_key if use_ai else None
        self.use_ai = use_ai and bool(groq_api_key)
        self.file_processor = GenericFileProcessor(self.groq_api_key, use_ai=self.use_ai)
        self.graph_builder = GraphRelationshipBuilder()
        self.cache_manager = CacheManager(cache_dir)
        self.groq_client: Optional[Groq] = (
            Groq(api_key=groq_api_key) if self.use_ai else None
        )

        self.faiss_index = None
        self.metadata_list: Optional[List[Dict]] = None
        self.model: Optional[SentenceTransformer] = None
        self.graph: Optional[nx.Graph] = None
        self.entities: List[Dict] = []
        self.relationships: List[Dict] = []
        self.df: Optional[pd.DataFrame] = None

        self.field_index: Dict[str, Any] = {
            "supplier_to_addresses": defaultdict(set),
            "name_to_addresses": defaultdict(set),
        }

        # BM25 internals
        self._bm25_docs: List[List[str]] = []
        self._bm25_df: Dict[str, int] = {}
        self._bm25_idf: Dict[str, float] = {}
        self._bm25_doc_len: List[int] = []
        self._bm25_avgdl: float = 0.0

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_excel_data(self, excel_path: str) -> pd.DataFrame:
        logger.info("📊 Loading Excel data…")
        cache_key = (
            self.cache_manager.get_directory_cache_key(excel_path)
            if os.path.isdir(excel_path)
            else self.cache_manager.get_file_cache_key(excel_path)
        )
        cached = self.cache_manager.get_cached_data(f"excel_data_{cache_key}")
        if cached is not None:
            logger.info(f"✅ Using cached Excel data ({len(cached)} rows)")
            self.df = cached
            return self.df

        def _norm(val: Any) -> Any:
            if pd.isna(val):
                return val
            s = ucnorm("NFKC", str(val).replace("\xa0", " "))
            return re.sub(r"\s+", " ", s).strip()

        frames: List[pd.DataFrame] = []
        if os.path.isdir(excel_path):
            for root, _, files in os.walk(excel_path):
                for fn in files:
                    fp = os.path.join(root, fn)
                    if not fn.lower().endswith((".xlsx", ".xls", ".csv")):
                        continue
                    try:
                        part = (
                            pd.read_csv(fp)
                            if fn.lower().endswith(".csv")
                            else pd.read_excel(fp)
                        )
                        part["_source_file"] = fn
                        frames.append(part)
                        logger.info(f"  ✅ Loaded '{fn}' ({len(part)} rows)")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Skipping '{fn}': {e}")
            if not frames:
                raise ValueError(f"No readable Excel/CSV files in: {excel_path}")
            self.df = pd.concat(frames, ignore_index=True)
        else:
            self.df = pd.read_excel(excel_path)
            self.df["_source_file"] = os.path.basename(excel_path)

        # Normalise text columns
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].apply(_norm)

        self.cache_manager.set_cached_data(f"excel_data_{cache_key}", self.df)
        logger.info(f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns")
        return self.df

    def load_documents(self, documents_folder: str) -> List[Dict[str, Any]]:
        logger.info("📄 Loading documents…")
        dir_key = self.cache_manager.get_directory_cache_key(documents_folder)
        cached = self.cache_manager.get_cached_data(f"doc_chunks_{dir_key}")
        if cached is not None:
            logger.info(f"✅ Using cached document chunks ({len(cached)} chunks)")
            self._rebuild_field_index(cached)
            return cached

        all_chunks: List[Dict] = []
        doc_files = [
            os.path.join(r, f)
            for r, _, files in os.walk(documents_folder)
            for f in files
            if not f.lower().endswith((".xlsx", ".xls", ".csv"))
        ]
        logger.info(f"📁 Found {len(doc_files)} document files")

        for i, doc_path in enumerate(doc_files, 1):
            filename = os.path.basename(doc_path)
            logger.info(f"  📄 Processing {filename} ({i}/{len(doc_files)})…")

            file_key = self.cache_manager.get_file_cache_key(doc_path)
            cached_file = self.cache_manager.get_cached_data(f"doc_file_{file_key}")
            if cached_file is not None:
                all_chunks.extend(cached_file)
                continue

            try:
                fdata = self.file_processor.process_file(doc_path)
                if not fdata.get("content"):
                    logger.warning(f"    ⚠️ No content from {filename}")
                    continue

                text = fdata["content"]
                ftype = fdata.get("file_type", "unknown")
                meta = fdata.get("metadata", {})

                doc_type = self.file_processor.detect_document_type(text, filename, ftype)
                logger.info(f"    📋 Type: {doc_type}")
                structured = self.file_processor.extract_structured_data(text, doc_type)
                logger.info(f"    ✅ Structured fields extracted")

                file_chunks: List[Dict] = []
                start, j = 0, 0
                total = len(text)
                while start < total:
                    end = min(start + Config.CHUNK_SIZE, total)
                    snippet = text[start:end].strip()
                    if snippet:
                        file_chunks.append({
                            "id": f"{filename}_{j}",
                            "source": filename,
                            "type": "document",
                            "file_type": ftype,
                            "document_type": doc_type,
                            "text": snippet,
                            "structured_data": structured,
                            "metadata": meta,
                            "chunk_index": j,
                        })
                        j += 1
                    if end == total:
                        break
                    start = end - Config.OVERLAP

                all_chunks.extend(file_chunks)
                self.cache_manager.set_cached_data(f"doc_file_{file_key}", file_chunks)

            except Exception as e:
                logger.warning(f"  ⚠️ Error processing {doc_path}: {e}")

        self._rebuild_field_index(all_chunks)
        if all_chunks:
            self.cache_manager.set_cached_data(f"doc_chunks_{dir_key}", all_chunks)
        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(doc_files)} files")
        return all_chunks

    def _rebuild_field_index(self, chunks: List[Dict]) -> None:
        for chunk in chunks:
            sd = chunk.get("structured_data") or {}
            for fname, fval in sd.items():
                if not fval:
                    continue
                if "name" in fname.lower() and "address" in sd:
                    n = self.graph_builder._normalize_name(fval)
                    a = str(sd["address"]).strip()
                    if n and a:
                        self.field_index["name_to_addresses"][n].add(a)
                if "supplier" in fname.lower() and "address" in sd:
                    n = self.graph_builder._normalize_name(fval)
                    a = str(sd["address"]).strip()
                    if n and a:
                        self.field_index["supplier_to_addresses"][n].add(a)

    # ------------------------------------------------------------------ #
    # Excel → chunk helper
    # ------------------------------------------------------------------ #

    def create_excel_chunks(self) -> List[Dict]:
        if self.df is None:
            raise ValueError("Excel data not loaded")
        chunks = []
        for idx, row in self.df.iterrows():
            header = " | ".join(
                f"{col}: {row[col]}"
                for col in self.df.columns
                if col != "_source_file" and pd.notna(row[col])
            )
            chunks.append({
                "id": f"excel_row_{idx}",
                "source": str(row.get("_source_file", "excel")),
                "type": "excel",
                "document_type": "inventory_register",
                "row_index": idx,
                "text": header + "\n" + str(row.to_dict()),
                "raw_data": row.to_dict(),
            })
        return chunks

    # ------------------------------------------------------------------ #
    # Build system
    # ------------------------------------------------------------------ #

    def build_system(self, excel_path: str, documents_folder: str):
        logger.info("🔧 Building Graph RAG System…")

        self.load_excel_data(excel_path)
        document_chunks = self.load_documents(documents_folder)

        logger.info("🔍 Extracting entities…")
        excel_entities = self.graph_builder.extract_entities_from_excel(self.df)
        doc_entities = self.graph_builder.extract_entities_from_pdf(document_chunks)
        logger.info(f"  Excel: {len(excel_entities)}  |  Docs: {len(doc_entities)}")

        logger.info("🔗 Building relationships…")
        self.relationships = self.graph_builder.build_relationships(
            excel_entities, doc_entities
        )
        logger.info(f"  {len(self.relationships)} relationships created")

        logger.info("🕸️ Creating network graph…")
        all_entities = excel_entities + doc_entities
        self.graph = self.graph_builder.create_network_graph(
            all_entities, self.relationships
        )
        self.entities = all_entities
        logger.info(
            f"  Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        logger.info("🔢 Building vector index…")
        excel_chunks = self.create_excel_chunks()
        all_chunks = excel_chunks + document_chunks

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(
            texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings /= norms

        dim = embeddings.shape[1]
        self.faiss_index = (
            faiss.IndexHNSWFlat(dim, 32)
            if len(embeddings) > 10_000
            else faiss.IndexFlatIP(dim)
        )
        self.faiss_index.add(embeddings)
        self.metadata_list = all_chunks
        logger.info(f"  Vector index: {len(all_chunks)} chunks")

        self._build_bm25([c["text"] for c in all_chunks])
        logger.info("✅ System ready")
        return all_entities, self.relationships

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2]

    def _build_bm25(self, docs: List[str]) -> None:
        tokenized = [self._tokenize(d) for d in docs]
        self._bm25_docs = tokenized
        N = len(tokenized)
        df: Dict[str, int] = {}
        lengths = []
        for toks in tokenized:
            lengths.append(len(toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._bm25_df = df
        self._bm25_doc_len = lengths
        self._bm25_avgdl = sum(lengths) / N if N else 0.0
        self._bm25_idf = {
            t: max(0.0, np.log((N - dfi + 0.5) / (dfi + 0.5) + 1))
            for t, dfi in df.items()
        }

    def _bm25_search(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        if not self._bm25_docs:
            return {}
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return {}
        k1, b = Config.BM25_K1, Config.BM25_B
        candidates = (
            candidate_indices if candidate_indices is not None
            else list(range(len(self._bm25_docs)))
        )
        scores: Dict[int, float] = {}
        for di in candidates:
            toks = self._bm25_docs[di]
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = self._bm25_doc_len[di] or 1
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                idf = self._bm25_idf.get(qt, 0.0)
                freq = tf[qt]
                denom = freq + k1 * (1 - b + b * dl / (self._bm25_avgdl or 1))
                score += idf * freq * (k1 + 1) / denom
            scores[di] = score / 10.0
        return scores

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self.faiss_index is None:
            raise ValueError("System not built. Call build_system() first.")

        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
        search_k = max(k * 3, 50)
        D, I = self.faiss_index.search(q_emb, search_k)

        q_tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) > 2]
        supplier_filter = None
        m = re.search(r"address\s+of\s+(.+)$", query, re.IGNORECASE)
        if m:
            supplier_filter = m.group(1).strip().lower()

        bm25_scores = self._bm25_search(query, candidate_indices=I[0].tolist())
        results: List[Dict] = []
        for i, idx in enumerate(I[0]):
            chunk = self.metadata_list[idx]
            sim = float(D[0][i])
            text_lower = chunk["text"].lower()
            boost = 0.1 if any(t in text_lower for t in q_tokens) else 0.0
            bm25 = bm25_scores.get(int(idx), 0.0)
            score = sim + boost + 0.4 * bm25
            if supplier_filter and supplier_filter not in text_lower:
                continue
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "type": chunk["type"],
                "document_type": chunk.get("document_type", "unknown"),
                "row_index": chunk.get("row_index"),
                "similarity": score,
                "distance": 1.0 - sim,
            })
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:k]

    def find_related_documents(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.graph:
            return []
        search_results = self.search(query, k=10)
        related: List[Dict] = []
        seen: set = set()

        for res in search_results:
            if res["type"] == "excel":
                candidates = [
                    e for e in self.entities if e.get("row_index") == res.get("row_index")
                ]
            else:
                candidates = [
                    e for e in self.entities if e.get("source_file") == res["source"]
                ]
            for entity in candidates:
                eid = entity["id"]
                if eid not in self.graph.nodes or eid in seen:
                    continue
                seen.add(eid)
                for nbr_id in self.graph.neighbors(eid):
                    if nbr_id in seen:
                        continue
                    nd = self.graph.nodes[nbr_id]
                    ed = self.graph.edges[eid, nbr_id]
                    related.append({
                        "entity_name": entity["name"],
                        "related_entity": nd.get("name", "Unknown"),
                        "relationship_type": ed.get("type", "unknown"),
                        "relationship_strength": ed.get("strength", 0),
                        "related_source": nd.get("source", "unknown"),
                        "related_type": nd.get("type", "unknown"),
                        "description": ed.get("description", ""),
                    })
                    seen.add(nbr_id)

        related.sort(key=lambda x: x["relationship_strength"], reverse=True)
        return related[:max_results]

    # ------------------------------------------------------------------ #
    # Groq-powered answer
    # ------------------------------------------------------------------ #

    def query_with_groq(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate an answer using Groq LLM given retrieved context chunks."""
        if not self.use_ai or not self.groq_client:
            return "Groq API not available — showing search results only."

        context_parts = []
        for chunk in context_chunks:
            prefix = "Excel Row" if chunk["type"] == "excel" else f"Document ({chunk['document_type']})"
            context_parts.append(f"[{prefix}] {chunk['text']}")
        context = "\n\n".join(context_parts)

        prompt = (
            "You are an expert data analyst for ABC Book Stores inventory and document management.\n\n"
            "Use the following retrieved data to answer the question accurately.\n\n"
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
            f"Question: {query}\n\n"
            "Provide a detailed answer that:\n"
            "1. Directly addresses the question\n"
            "2. Cites specific data points and sources\n"
            "3. Explains relationships between data sources where relevant\n"
            "4. Provides business insights where applicable\n\n"
            "Answer:"
        )

        for attempt in range(Config.GROQ_MAX_RETRIES):
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(
                        self.groq_client.chat.completions.create,
                        messages=[{"role": "user", "content": prompt}],
                        model=Config.GROQ_MODEL_LARGE,
                        max_tokens=1024,
                    )
                    resp = future.result(timeout=Config.GROQ_TIMEOUT_SECONDS)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"  ⚠️ Groq query attempt {attempt+1} failed: {e}. Retry in {wait}s…")
                time.sleep(wait)

        return "⚠️ Groq API unavailable after retries — please check your API key or network."

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def search_and_answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        logger.info(f"🔍 Query: '{query}'")

        # Aggregation intent → fetch more chunks
        is_agg = (
            any(re.search(p, query, re.IGNORECASE) for p in [r"\btotal\b", r"\bsum\b", r"\baggregate\b"])
            and any(re.search(p, query, re.IGNORECASE) for p in [r"supplier", r"customer"])
        )
        k_eff = max(k, 10)
        if is_agg and self.df is not None:
            k_eff = max(k_eff, len(self.df) + 20)

        addr_intent = bool(re.search(r"\baddress\b|\blocation\b", query, re.IGNORECASE))
        supplier_target = None
        m = re.search(r"address\s+of\s+(.+)$", query, re.IGNORECASE)
        if m:
            supplier_target = m.group(1).strip()

        if addr_intent:
            k_eff = (
                len(self.metadata_list)
                if isinstance(self.metadata_list, list)
                else max(k_eff, 100)
            )

        search_results = self.search(query, k_eff)

        # Fast path: address lookup via field index
        if addr_intent and supplier_target:
            sup_norm = self.graph_builder._normalize_name(supplier_target)
            addr_set = (
                self.field_index["supplier_to_addresses"].get(sup_norm)
                or self.field_index["name_to_addresses"].get(sup_norm)
            )
            if addr_set:
                answer = f"Address: {max(addr_set, key=len)}"
                related = self.find_related_documents(query, max_results=10)
                return {
                    "query": query, "answer": answer,
                    "search_results": search_results,
                    "related_documents": related,
                    "num_results": len(search_results),
                    "num_relationships": len(related),
                }

        # Address scan from retrieved PDF chunks
        if addr_intent:
            addr_regexes = [
                r"\b\d{1,4}[^\n,]{0,40},[^\n]{0,80}\b\d{6}\b",
                r"\b[A-Z0-9\-]{1,10}[^\n,]{0,40},[^\n]{0,120}\b",
            ]
            best_addr, best_len = None, 0
            q_sup = (supplier_target or "").lower()
            for res in search_results:
                if res.get("type") != "document":
                    continue
                text = res.get("text", "")
                if q_sup and q_sup not in text.lower():
                    continue
                for rgx in addr_regexes:
                    for ma in re.finditer(rgx, text, re.IGNORECASE):
                        addr = ma.group(0).strip()
                        if len(addr) > best_len:
                            best_len, best_addr = len(addr), addr
            if best_addr:
                related = self.find_related_documents(query, max_results=10)
                return {
                    "query": query, "answer": f"Address: {best_addr}",
                    "search_results": search_results,
                    "related_documents": related,
                    "num_results": len(search_results),
                    "num_relationships": len(related),
                }

        related = self.find_related_documents(query, max_results=10)
        answer = self.query_with_groq(query, search_results[:k])

        return {
            "query": query,
            "answer": answer,
            "search_results": search_results,
            "related_documents": related,
            "num_results": len(search_results),
            "num_relationships": len(related),
        }

    # ------------------------------------------------------------------ #
    # Statistics & export
    # ------------------------------------------------------------------ #

    def get_system_statistics(self) -> Dict[str, Any]:
        if not self.graph:
            return {"error": "System not built"}
        node_types: Dict[str, int] = {}
        source_types: Dict[str, int] = {}
        for nid in self.graph.nodes():
            nd = self.graph.nodes[nid]
            node_types[nd.get("type", "?")] = node_types.get(nd.get("type", "?"), 0) + 1
            source_types[nd.get("source", "?")] = source_types.get(nd.get("source", "?"), 0) + 1
        edge_types: Dict[str, int] = {}
        for _, _, ed in self.graph.edges(data=True):
            et = ed.get("type", "?")
            edge_types[et] = edge_types.get(et, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "node_types": node_types,
            "source_types": source_types,
            "edge_types": edge_types,
        }

    def export_graph_data(self, output_file: str = "graph_data.json") -> Dict[str, Any]:
        if not self.graph:
            return {"error": "Graph not built"}
        nodes = [
            {"id": nid, "label": nd.get("name", nid),
             "type": nd.get("type", "?"), "source": nd.get("source", "?"),
             "group": nd.get("source", "?")}
            for nid, nd in self.graph.nodes(data=True)
        ]
        edges = [
            {"from": e[0], "to": e[1],
             "type": e[2].get("type", "?"), "strength": e[2].get("strength", 0),
             "description": e[2].get("description", "")}
            for e in self.graph.edges(data=True)
        ]
        data = {"nodes": nodes, "edges": edges,
                "statistics": self.get_system_statistics()}
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ Graph exported to {output_file}")
        return data

    def clear_cache(self) -> Dict[str, Any]:
        try:
            self.cache_manager.clear_cache()
            return {"success": True, "message": "Cache cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def export_to_neo4j(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        clear_database: bool = True,
    ) -> Dict[str, Any]:
        """Export graph to Neo4j (requires neo4j Python driver)."""
        if not self.graph:
            return {"error": "Graph not built"}
        try:
            from neo4j import GraphDatabase as _GDB
            driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            with driver.session() as session:
                if clear_database:
                    session.run("MATCH (n) DETACH DELETE n")
                node_count = 0
                for nid, nd in self.graph.nodes(data=True):
                    labels = ":".join(
                        filter(None, [
                            nd.get("source", "Unknown").title(),
                            nd.get("type", "").title(),
                        ])
                    )
                    props = {
                        k: v for k, v in {
                            "id": nid,
                            "name": nd.get("name", "?"),
                            "type": nd.get("type", "?"),
                            "source": nd.get("source", "?"),
                            **nd.get("attributes", {}),
                        }.items() if v is not None
                    }
                    session.run(f"CREATE (n:{labels}) SET n = $p", p=props)
                    node_count += 1
                rel_count = 0
                for e in self.graph.edges(data=True):
                    rtype = e[2].get("type", "RELATED")
                    session.run(
                        f"MATCH (a {{id:$f}}),(b {{id:$t}}) CREATE (a)-[r:{rtype}]->(b) SET r=$p",
                        f=e[0], t=e[1],
                        p={"strength": e[2].get("strength", 0),
                           "description": e[2].get("description", "")},
                    )
                    rel_count += 1
            driver.close()
            logger.info(f"✅ Neo4j export: {node_count} nodes, {rel_count} rels")
            return {"success": True, "nodes_created": node_count,
                    "relationships_created": rel_count}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    logger.info("🚀 Graph RAG System — Groq Edition")
    logger.info("=" * 60)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning(
            "⚠️  GROQ_API_KEY not set. AI features disabled.\n"
            "    Export it with:  export GROQ_API_KEY='gsk_…'"
        )

    rag = GraphRAGSystem(groq_api_key=groq_api_key)

    excel_path = "data/excel"
    docs_folder = "data"

    if not os.path.exists(excel_path) or not os.path.exists(docs_folder):
        logger.error("❌ Required data directories not found (data/excel and data/)")
        return

    rag.build_system(excel_path, docs_folder)

    stats = rag.get_system_statistics()
    logger.info("\n📊 System Statistics:")
    logger.info(json.dumps(stats, indent=2))

    test_queries = [
        "What books are available in the inventory?",
        "Show me GRN documents from 2024",
        "Find purchase orders and their related inventory",
        "What invoices are available?",
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"❓ {query}")
        result = rag.search_and_answer(query)
        logger.info(f"📊 {result['num_results']} results  |  🔗 {result['num_relationships']} related docs")
        logger.info(f"\n🤖 Answer:\n{result['answer']}")

    rag.export_graph_data()
    logger.info("\n🎉 Done!")


if __name__ == "__main__":
    main()