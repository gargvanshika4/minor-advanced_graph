#!/usr/bin/env python3
"""
Graph RAG System Web Interface
Flask frontend powered entirely by Groq API.
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
# ✅ CORRECT
from graph_rag_groq import GraphRAGSystem

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load env
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
EXCEL_PATH = os.getenv("EXCEL_PATH", "data/excel")
DOCS_PATH  = os.getenv("DOCS_PATH",  "data")
PORT       = int(os.getenv("PORT", 5001))

if not GROQ_API_KEY:
    logger.warning(
        "⚠️  GROQ_API_KEY not set — AI features will be disabled.\n"
        "    Set it with:  export GROQ_API_KEY='gsk_…'  or add it to .env"
    )

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global RAG system instance
rag_system = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rag() -> GraphRAGSystem:
    """Create and fully initialise a fresh GraphRAGSystem instance."""
    use_ai = bool(GROQ_API_KEY)
    system = GraphRAGSystem(groq_api_key=GROQ_API_KEY, use_ai=use_ai)
    system.build_system(EXCEL_PATH, DOCS_PATH)
    return system


def get_system() -> GraphRAGSystem:
    """Return the global RAG system, initialising it on first call."""
    global rag_system
    if rag_system is None:
        logger.info("🔧 Initialising Graph RAG System…")
        rag_system = _build_rag()
        logger.info("✅ System ready.")
    return rag_system


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Lightweight health-check — does NOT trigger system initialisation."""
    return jsonify({
        "status":       "healthy",
        "groq_api_key": "configured" if GROQ_API_KEY else "missing",
        "system_ready": rag_system is not None,
    })


@app.route("/api/stats")
def stats():
    """Return graph and index statistics."""
    try:
        system = get_system()
        return jsonify(system.get_system_statistics())
    except Exception as e:
        logger.exception("Error in /api/stats")
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    """
    Ask a natural-language question against the indexed data.

    Request JSON:
        { "query": "Which invoices are linked to supplier XYZ?" }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "answer": "...",
            "num_results": 12,
            "num_relationships": 5,
            "related_documents": [...]
        }
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        query_text = (body.get("query") or "").strip()

        if not query_text:
            return jsonify({"error": "No query provided. Send JSON: {\"query\": \"…\"}"}), 400

        system = get_system()
        result = system.search_and_answer(query_text)

        return jsonify({
            "success":           True,
            "query":             query_text,
            "answer":            result.get("answer", ""),
            "num_results":       result.get("num_results", 0),
            "num_relationships": result.get("num_relationships", 0),
            "related_documents": result.get("related_documents", [])[:10],
        })

    except Exception as e:
        logger.exception("Error in /api/query")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reload", methods=["POST"])
def reload_system():
    """
    Rebuild the entire RAG system from scratch.
    Call this after adding new PDFs or Excel files to the data directories.
    Cache is preserved (speeds up re-ingestion of unchanged files).
    """
    global rag_system
    try:
        logger.info("🔄 Rebuilding Graph RAG System…")
        rag_system = _build_rag()
        logger.info("✅ Rebuild complete.")
        return jsonify({"success": True, "message": "System rebuilt successfully."})
    except Exception as e:
        logger.exception("Error in /api/reload")
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    """
    Wipe the on-disk cache.
    The next query (or /api/reload) will re-process all files from scratch.
    """
    global rag_system
    try:
        system = get_system()
        result = system.clear_cache()
        if result.get("success"):
            # Invalidate the in-memory system so everything is re-built fresh
            rag_system = None
            logger.info("🗑️ Cache cleared — system will reinitialise on next request.")
            return jsonify({"success": True, "message": "Cache cleared. System will reinitialise on next request."})
        return jsonify({"success": False, "error": result.get("error", "Unknown error")}), 500
    except Exception as e:
        logger.exception("Error in /api/clear-cache")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Graph RAG Web Interface (Groq Edition)…")
    logger.info(f"📡 Groq API key : {'✅ set' if GROQ_API_KEY else '❌ NOT SET'}")
    logger.info(f"📂 Excel path   : {EXCEL_PATH}")
    logger.info(f"📂 Docs path    : {DOCS_PATH}")
    logger.info(f"🌐 Listening on : http://0.0.0.0:{PORT}")

    # Pre-warm the system before accepting traffic
    try:
        get_system()
    except Exception as e:
        logger.error(f"❌ System initialisation failed: {e}")

    app.run(debug=False, host="0.0.0.0", port=PORT)