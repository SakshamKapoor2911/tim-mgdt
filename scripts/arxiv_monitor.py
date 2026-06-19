"""
Agent-Native ArXiv Monitor Script
---------------------------------
Designed specifically to be run by AI assistants via terminal tools.
Outputs structured Markdown to stdout for immediate context window ingestion,
and dumps a machine-readable JSON payload.

Usage:
    python scripts/arxiv_monitor.py
    python scripts/arxiv_monitor.py --keywords "mechanistic interpretability" "transformer pruning" --max-results 3
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
import argparse
import json
import os
import time

DEFAULT_KEYWORDS = [
    '"transformer pruning" OR "LLM pruning"',
    '("mechanistic interpretability" OR "circuit discovery") AND "LLM"',
    '"lottery ticket hypothesis" AND "language models"',
    '"activation patching" OR "causal tracing"'
]

def search_arxiv(query, max_results=3):
    """Query ArXiv API and return a list of parsed paper dictionaries."""
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'all:{urllib.parse.quote(query)}'
    
    url = f'{base_url}search_query={search_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    
    papers = []
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)

        for entry in entries:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text[:10]
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            link = entry.find('atom:id', ns).text
            
            papers.append({
                "title": title,
                "published_date": published,
                "authors": authors,
                "summary": summary,
                "url": link
            })
    except Exception as e:
        print(f"Error querying ArXiv for '{query}': {e}")
        
    return papers

def main():
    parser = argparse.ArgumentParser(description="Agent-Native ArXiv Monitor")
    parser.add_argument("--keywords", nargs='+', default=DEFAULT_KEYWORDS, help="List of exact-match keywords")
    parser.add_argument("--max-results", type=int, default=3, help="Max results per keyword")
    parser.add_argument("--out-json", type=str, default="scripts/arxiv_latest.json", help="Path to dump JSON results")
    args = parser.parse_args()

    # Agent-friendly Markdown output
    print("## [SCANNING] ArXiv Monitor Output")
    print(f"**Execution Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"**Target Keywords:** {', '.join(args.keywords)}\n")

    full_results = {}
    
    for kw in args.keywords:
        papers = search_arxiv(kw, args.max_results)
        full_results[kw] = papers
        time.sleep(3)  # Prevent HTTP 429 Too Many Requests
        
        print(f"### Keyword: `{kw}`")
        if not papers:
            print("*No recent papers found.*\n")
            continue
            
        for idx, p in enumerate(papers, 1):
            print(f"{idx}. **{p['title']}**")
            print(f"   - **Date:** {p['published_date']} | **Authors:** {', '.join(p['authors'][:3])}{' et al.' if len(p['authors']) > 3 else ''}")
            print(f"   - **Link:** {p['url']}")
            # Truncate summary for agent context window limits
            print(f"   - **Abstract:** {p['summary'][:300]}...\n")

    # Dump machine-readable state for deep agent inspection
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"timestamp": datetime.datetime.now().isoformat()},
            "results": full_results
        }, f, indent=2)
        
    print(f"*[SYSTEM] Full machine-readable payload dumped to: {args.out_json}*")

if __name__ == "__main__":
    main()
