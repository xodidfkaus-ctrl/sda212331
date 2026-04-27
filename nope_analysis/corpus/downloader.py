"""
Corpus downloader and sampler for NoPE analysis experiments.

Supported corpora:
  'en'       — WikiText-103 (Wikipedia articles, ~700K passages, avg 185 tokens)
  'en_edgar' — EDGAR 10-K filings (SEC annual reports, ~7K docs, avg 17K tokens per section)
  'ko'       — KLUE-MRC (Korean reading comprehension, ~17K passages, avg 500 tokens)
  'ko_dart'  — DART Korean financial disclosures (requires DART_API_KEY env var)

Cache: nope_analysis/corpus/cache/{lang}.json  (ephemeral — not committed to git)

Usage:
    from nope_analysis.corpus.downloader import get_text_sample, download_all
    text = get_text_sample('en_edgar', min_tokens=4096, tokenizer=tokenizer)
    download_all()                   # pre-cache all available corpora
    download_all(['en', 'en_edgar']) # selective
"""
import json
import os
import random
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

_CORPUS: dict = {}  # lang -> list[str]  (loaded lazily)

# EDGAR: years to pull. 2018-2020 gives ~21K filings with complete section text.
_EDGAR_YEARS = ['2018', '2019', '2020']
# EDGAR: sections with substantive prose (skip boilerplate stubs)
_EDGAR_SECTIONS = ['section_1', 'section_1A', 'section_7', 'section_7A', 'section_9A']
_EDGAR_MIN_SECTION_CHARS = 2000  # skip sections shorter than this


# ---------------------------------------------------------------------------
# English — WikiText-103
# ---------------------------------------------------------------------------

def _load_wikitext103():
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        texts = [row['text'] for row in ds if len(row['text'].strip()) > 200]
        print(f"[corpus] WikiText-103 loaded: {len(texts):,} passages")
        return texts
    except Exception as e:
        print(f"[corpus] WikiText-103 load failed: {e}")
        return []


# ---------------------------------------------------------------------------
# English — EDGAR 10-K filings (SEC annual reports)
# Each document is one 10-K filing split into sections.
# section_1 / section_7 alone are often 10,000–70,000 chars (no concatenation needed).
# ---------------------------------------------------------------------------

def _load_edgar():
    """
    Load EDGAR 10-K filings from HuggingFace (eloukas/edgar-corpus).
    Returns list of long-form text strings, one per filing section.
    Uses hf_hub_download per year (dataset script is deprecated).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[corpus] EDGAR load failed: huggingface_hub not installed")
        return []

    texts = []
    for year in _EDGAR_YEARS:
        try:
            path = hf_hub_download(
                repo_id='eloukas/edgar-corpus',
                filename=f'{year}/train.jsonl',
                repo_type='dataset',
            )
            with open(path, encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    for sec in _EDGAR_SECTIONS:
                        content = doc.get(sec, '')
                        if len(content) >= _EDGAR_MIN_SECTION_CHARS:
                            texts.append(content)
            print(f"[corpus] EDGAR {year}: loaded, running total {len(texts):,} sections")
        except Exception as e:
            print(f"[corpus] EDGAR {year} load failed: {e}")

    print(f"[corpus] EDGAR total: {len(texts):,} sections (from {_EDGAR_YEARS})")
    return texts


# ---------------------------------------------------------------------------
# Korean — KLUE-MRC
# ---------------------------------------------------------------------------

def _load_klue_mrc():
    try:
        from datasets import load_dataset
        ds = load_dataset('klue', 'mrc', split='train')
        texts = [row['context'] for row in ds if len(row['context'].strip()) > 200]
        print(f"[corpus] KLUE-MRC loaded: {len(texts):,} passages")
        return texts
    except Exception as e:
        print(f"[corpus] KLUE-MRC load failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Korean — DART (FSS electronic disclosure system)
# Requires DART_API_KEY environment variable.
# Free public API: https://opendart.fss.or.kr/
# ---------------------------------------------------------------------------

def _load_dart(api_key: str | None = None):
    """
    Load Korean financial disclosure reports from DART OpenAPI.
    Returns list of report body strings.

    Requires DART_API_KEY env var (or pass api_key argument directly).
    Free API, no authentication tier needed for public disclosures.

    Rate limit: 1,000 calls/day. One call per report. We fetch up to 200 reports.
    """
    key = api_key or os.environ.get('DART_API_KEY', '')
    if not key:
        print("[corpus] DART load skipped: DART_API_KEY not set. "
              "Set with: export DART_API_KEY=your_key_here")
        return []

    try:
        import urllib.request
        import urllib.parse
    except ImportError:
        print("[corpus] DART load failed: urllib not available")
        return []

    texts = []

    # Step 1: fetch recent annual report list (사업보고서, pblntf_ty=A)
    list_url = (
        "https://opendart.fss.or.kr/api/list.json?"
        + urllib.parse.urlencode({
            'crtfc_key': key,
            'pblntf_ty': 'A',   # 사업보고서 (annual report)
            'page_count': 40,
            'page_no': 1,
        })
    )
    try:
        with urllib.request.urlopen(list_url, timeout=15) as resp:
            data = json.loads(resp.read())
        if data.get('status') != '000':
            print(f"[corpus] DART list API error: {data.get('message', data.get('status'))}")
            return []
        reports = data.get('list', [])
        print(f"[corpus] DART: fetched {len(reports)} report entries")
    except Exception as e:
        print(f"[corpus] DART list fetch failed: {e}")
        return []

    # Step 2: for each report, fetch full document text
    for report in reports[:200]:  # cap at 200 to stay within rate limit
        rcept_no = report.get('rcept_no', '')
        if not rcept_no:
            continue
        doc_url = (
            "https://opendart.fss.or.kr/api/document.json?"
            + urllib.parse.urlencode({'crtfc_key': key, 'rcept_no': rcept_no})
        )
        try:
            with urllib.request.urlopen(doc_url, timeout=15) as resp:
                doc_data = json.loads(resp.read())
            body = doc_data.get('body', '')
            if len(body) >= 1000:
                texts.append(body)
        except Exception:
            pass  # skip individual failures silently

    print(f"[corpus] DART loaded: {len(texts):,} reports")
    return texts


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _load_corpus(lang: str, dart_api_key: str | None = None):
    global _CORPUS
    if lang in _CORPUS:
        return _CORPUS[lang]

    cache_file = CACHE_DIR / f'{lang}.json'
    if cache_file.exists():
        with open(cache_file, encoding='utf-8') as f:
            _CORPUS[lang] = json.load(f)
        print(f"[corpus] Loaded '{lang}' from local cache ({len(_CORPUS[lang]):,} passages)")
        return _CORPUS[lang]

    if lang == 'en':
        texts = _load_wikitext103()
    elif lang == 'en_edgar':
        texts = _load_edgar()
    elif lang == 'ko':
        texts = _load_klue_mrc()
    elif lang == 'ko_dart':
        texts = _load_dart(api_key=dart_api_key)
    else:
        raise ValueError(
            f"Unknown lang '{lang}'. Supported: 'en', 'en_edgar', 'ko', 'ko_dart'"
        )

    if texts:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False)
        print(f"[corpus] Cached '{lang}' to {cache_file}")

    _CORPUS[lang] = texts
    return texts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_text_sample(
    lang: str,
    min_tokens: int,
    tokenizer,
    seed: int = 42,
    max_retries: int = 20,
    dart_api_key: str | None = None,
) -> str:
    """
    Return a single text sample of at least min_tokens tokens.
    For en_edgar / ko_dart, a single document is usually sufficient.
    For en / ko, passages are concatenated until the budget is met.
    Falls back to synthetic repeated text if corpus unavailable.
    """
    corpus = _load_corpus(lang, dart_api_key=dart_api_key)

    if not corpus:
        print(f"[corpus] WARNING: no corpus for lang='{lang}', using fallback synthetic text")
        base = (
            "The development of large language models has fundamentally changed natural language understanding. "
            "Sliding window attention limits each token to attending only within a local window. "
            "Global attention layers allow every token to attend to every other token in the sequence. "
        )
        return base * (min_tokens // 30 + 10)

    rng = random.Random(seed)
    passages = rng.sample(corpus, k=min(max_retries, len(corpus)))
    combined = ' '.join(passages)

    ids = tokenizer(combined, return_tensors='pt', add_special_tokens=False)['input_ids']
    if ids.shape[1] >= min_tokens:
        return combined

    extra = rng.choices(corpus, k=max_retries * 2)
    combined = combined + ' ' + ' '.join(extra)
    return combined


def build_input_from_corpus(lang: str, target_len: int, tokenizer, dart_api_key: str | None = None):
    """
    Build a token tensor of exactly target_len tokens.
    Returns input_ids tensor (1, target_len).
    """
    import torch
    text = get_text_sample(lang, min_tokens=target_len + 256, tokenizer=tokenizer,
                           dart_api_key=dart_api_key)
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids']
    if ids.shape[1] < target_len:
        raise ValueError(
            f"Corpus text too short after tokenization: got {ids.shape[1]} tokens, "
            f"need {target_len}. Try a different lang or increase max_retries."
        )
    return ids[:, :target_len]


def download_all(languages=('en', 'en_edgar', 'ko'), dart_api_key: str | None = None):
    """
    Pre-download and cache corpora. Run once at session start.
    ko_dart is excluded by default — pass dart_api_key to include it.

    Example:
        download_all()                              # en + en_edgar + ko
        download_all(['ko_dart'], dart_api_key='xxx')  # DART only
    """
    print("Pre-downloading corpora...")
    for lang in languages:
        texts = _load_corpus(lang, dart_api_key=dart_api_key)
        print(f"  {lang}: {len(texts):,} passages cached")
    if dart_api_key and 'ko_dart' not in languages:
        texts = _load_corpus('ko_dart', dart_api_key=dart_api_key)
        print(f"  ko_dart: {len(texts):,} reports cached")
    print("Done.")


if __name__ == '__main__':
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else None
    download_all(dart_api_key=key)
