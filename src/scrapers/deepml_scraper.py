"""Deep-ML Problem Scraper & Knowledge Base Builder.

Scrapes all verified Machine Learning / Deep Learning problems from Open-Deep-ML
(https://www.deep-ml.com/) and builds a structured, offline technical knowledge base
for the AI Video Generator pipeline.
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepml_scraper")

GITHUB_API_URL = "https://api.github.com/repos/Open-Deep-ML/DML-OpenProblem/contents/questions"
RAW_BASE_URL = "https://raw.githubusercontent.com/Open-Deep-ML/DML-OpenProblem/main/questions"
OUTPUT_DIR = Path("assets/docs/deepml")
INDEX_FILE = OUTPUT_DIR / "problems_index.json"


def fetch_question_list() -> List[str]:
    """Fetch all question folder names from GitHub repository."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(GITHUB_API_URL, headers=headers, timeout=15)
    if resp.status_code != 200:
        logger.error(f"Failed to fetch question list: HTTP {resp.status_code}")
        return []
    
    items = resp.json()
    folders = [item["name"] for item in items if item.get("type") == "dir"]
    logger.info(f"Discovered {len(folders)} problem folders on Deep-ML repo.")
    return folders


def fetch_single_problem(folder: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """Download meta, description, learn, solution, and tests for one problem."""
    base_url = f"{RAW_BASE_URL}/{folder}"
    problem_data = {"folder": folder}

    # 1. Fetch meta.json
    try:
        r_meta = session.get(f"{base_url}/meta.json", timeout=10)
        if r_meta.status_code == 200:
            meta = r_meta.json()
            problem_data["id"] = meta.get("id", folder.split("_")[0])
            problem_data["title"] = meta.get("title", folder.replace("_", " ").title())
            problem_data["difficulty"] = meta.get("difficulty", "medium")
            problem_data["category"] = meta.get("category", "Machine Learning")
            problem_data["tags"] = meta.get("tags", [])
        else:
            problem_data["id"] = folder.split("_")[0]
            problem_data["title"] = folder.replace("_", " ").title()
            problem_data["difficulty"] = "medium"
            problem_data["category"] = "Machine Learning"
            problem_data["tags"] = []
    except Exception as e:
        logger.warning(f"Error parsing meta for {folder}: {e}")
        return None

    # 2. Fetch description.md
    try:
        r_desc = session.get(f"{base_url}/description.md", timeout=10)
        problem_data["description"] = r_desc.text if r_desc.status_code == 200 else ""
    except Exception:
        problem_data["description"] = ""

    # 3. Fetch learn.md (theory & math intuition)
    try:
        r_learn = session.get(f"{base_url}/learn.md", timeout=10)
        problem_data["learn"] = r_learn.text if r_learn.status_code == 200 else ""
    except Exception:
        problem_data["learn"] = ""

    # 4. Fetch solution.py (verified from-scratch NumPy code)
    try:
        r_sol = session.get(f"{base_url}/solution.py", timeout=10)
        problem_data["solution"] = r_sol.text if r_sol.status_code == 200 else ""
    except Exception:
        problem_data["solution"] = ""

    # 5. Fetch tests.json (concrete input/output data points)
    try:
        r_tests = session.get(f"{base_url}/tests.json", timeout=10)
        if r_tests.status_code == 200:
            problem_data["tests"] = r_tests.json()
        else:
            problem_data["tests"] = []
    except Exception:
        problem_data["tests"] = []

    return problem_data


def run_scrape(max_workers: int = 10) -> List[Dict[str, Any]]:
    """Scrape all Deep-ML problems and store in assets/docs/deepml/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = fetch_question_list()
    if not folders:
        logger.warning("No folders found to scrape.")
        return []

    results = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    logger.info(f"Starting parallel download with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_problem, f, session): f for f in folders}
        for future in as_completed(futures):
            p = future.result()
            if p:
                results.append(p)
                if len(results) % 25 == 0:
                    logger.info(f"Downloaded {len(results)}/{len(folders)} problems...")

    logger.info(f"Successfully scraped {len(results)} problems.")

    # Sort by numerical ID if possible
    def get_sort_key(item):
        try:
            return int(item.get("id", 0))
        except (ValueError, TypeError):
            return 999999

    results.sort(key=get_sort_key)

    # Save master JSON index
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved master problem index to {INDEX_FILE}")

    # Generate category-wise Markdown cheat-sheets
    generate_category_docs(results)
    return results


def generate_category_docs(problems: List[Dict[str, Any]]) -> None:
    """Group problems by category and write structured Markdown knowledge documents."""
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for p in problems:
        cat = p.get("category", "General ML")
        # Normalize category names
        norm_cat = cat.strip().title()
        if "Linear Algebra" in norm_cat:
            norm_cat = "Linear Algebra"
        elif "Deep Learning" in norm_cat:
            norm_cat = "Deep Learning"
        elif "Machine Learning" in norm_cat:
            norm_cat = "Machine Learning Fundamentals"
        elif "Reinforcement" in norm_cat:
            norm_cat = "Reinforcement Learning"
        elif "Nlp" in norm_cat or "Natural Language" in norm_cat:
            norm_cat = "Natural Language Processing"
        
        categories.setdefault(norm_cat, []).append(p)

    for cat_name, cat_probs in categories.items():
        safe_filename = cat_name.lower().replace(" ", "_").replace("/", "_") + ".md"
        doc_path = OUTPUT_DIR / safe_filename
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(f"# Deep-ML Knowledge Base: {cat_name}\n\n")
            f.write(f"Total verified problems: {len(cat_probs)}\n\n")
            f.write("Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)\n\n---\n\n")

            for p in cat_probs:
                f.write(f"## Problem {p.get('id')}: {p.get('title')}\n")
                f.write(f"- **Difficulty**: `{p.get('difficulty')}`\n")
                f.write(f"- **Category**: {p.get('category')}\n\n")
                
                if p.get("description"):
                    f.write(f"### Description & Spec\n{p['description'].strip()}\n\n")
                
                if p.get("learn"):
                    f.write(f"### Mathematical Derivation & Intuition\n{p['learn'].strip()}\n\n")
                
                if p.get("solution"):
                    f.write(f"### Verified Pure NumPy Solution (From Scratch)\n```python\n{p['solution'].strip()}\n```\n\n")
                
                if p.get("tests") and len(p["tests"]) > 0:
                    sample_test = p["tests"][0]
                    f.write(f"### Verified Sample Trace Data\n```json\n{json.dumps(sample_test, indent=2)}\n```\n\n")
                
                f.write("---\n\n")

        logger.info(f"Generated category document: {doc_path} ({len(cat_probs)} problems)")


if __name__ == "__main__":
    run_scrape()
