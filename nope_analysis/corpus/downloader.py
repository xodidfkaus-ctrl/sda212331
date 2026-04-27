"""
Corpus downloader and sampler for NoPE analysis experiments.

Downloads WikiText-103 (English) and KLUE-MRC (Korean) via HuggingFace datasets.
Provides get_text_sample(lang, min_tokens, tokenizer) for experiments.

Cache: ~/.cache/huggingface/datasets/ (HF default)
Local copy: /home/elicer/sda212331/nope_analysis/corpus/cache/

Usage:
    from nope_analysis.corpus.downloader import get_text_sample
    text = get_text_sample('en', min_tokens=8192, tokenizer=tokenizer)
"""
import json
import random
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

_CORPUS: dict = {}  # lang -> list[str]  (loaded lazily)


def _load_wikitext103():
    """Load WikiText-103 English articles. Returns list of non-empty paragraphs."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        texts = [row['text'] for row in ds if len(row['text'].strip()) > 200]
        print(f"[corpus] WikiText-103 loaded: {len(texts):,} passages")
        return texts
    except Exception as e:
        print(f"[corpus] WikiText-103 load failed: {e}")
        return []


def _load_klue_mrc():
    """Load KLUE Machine Reading Comprehension — Korean passages."""
    try:
        from datasets import load_dataset
        ds = load_dataset('klue', 'mrc', split='train')
        texts = [row['context'] for row in ds if len(row['context'].strip()) > 200]
        print(f"[corpus] KLUE-MRC loaded: {len(texts):,} passages")
        return texts
    except Exception as e:
        print(f"[corpus] KLUE-MRC load failed: {e}")
        return []


def _load_corpus(lang: str):
    global _CORPUS
    if lang in _CORPUS:
        return _CORPUS[lang]

    cache_file = CACHE_DIR / f'{lang}.json'
    if cache_file.exists():
        with open(cache_file) as f:
            _CORPUS[lang] = json.load(f)
        print(f"[corpus] Loaded {lang} from local cache ({len(_CORPUS[lang]):,} passages)")
        return _CORPUS[lang]

    if lang == 'en':
        texts = _load_wikitext103()
    elif lang == 'ko':
        texts = _load_klue_mrc()
    else:
        raise ValueError(f"Unknown lang '{lang}'. Supported: 'en', 'ko'")

    if texts:
        with open(cache_file, 'w') as f:
            json.dump(texts, f, ensure_ascii=False)
        print(f"[corpus] Cached {lang} corpus to {cache_file}")

    _CORPUS[lang] = texts
    return texts


def get_text_sample(lang: str, min_tokens: int, tokenizer, seed: int = 42, max_retries: int = 20) -> str:
    """
    Return a single concatenated text sample of at least min_tokens tokens.

    Concatenates randomly sampled passages until the token budget is met.
    Falls back gracefully to repeated BASE_TEXT if corpus unavailable.
    """
    corpus = _load_corpus(lang)

    if not corpus:
        # Fallback: repeat a base sentence
        print(f"[corpus] WARNING: no corpus for lang='{lang}', using fallback text")
        base = (
            "The development of large language models has fundamentally changed natural language understanding. "
            "Sliding window attention limits each token to attending only within a local window. "
            "Global attention layers allow every token to attend to every other token in the sequence. "
        )
        return base * (min_tokens // 30 + 10)

    rng = random.Random(seed)
    passages = rng.sample(corpus, k=min(max_retries, len(corpus)))

    combined = ' '.join(passages)
    # Check token count
    ids = tokenizer(combined, return_tensors='pt', add_special_tokens=False)['input_ids']
    if ids.shape[1] >= min_tokens:
        return combined

    # If not enough, keep adding passages
    extra = rng.choices(corpus, k=max_retries * 2)
    combined = combined + ' ' + ' '.join(extra)
    return combined


def build_input_from_corpus(lang: str, target_len: int, tokenizer):
    """
    Build a token tensor of exactly target_len tokens using real corpus text.
    Returns input_ids tensor (1, target_len).
    """
    import torch
    text = get_text_sample(lang, min_tokens=target_len + 256, tokenizer=tokenizer)
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids']
    if ids.shape[1] < target_len:
        raise ValueError(
            f"Corpus text too short after tokenization: got {ids.shape[1]} tokens, need {target_len}. "
            "Try increasing max_retries or using a larger corpus."
        )
    return ids[:, :target_len]


def download_all(languages=('en', 'ko')):
    """Pre-download and cache all corpora. Run once before experiments."""
    print("Pre-downloading corpora...")
    for lang in languages:
        texts = _load_corpus(lang)
        print(f"  {lang}: {len(texts):,} passages cached")
    print("Done.")


if __name__ == '__main__':
    download_all()
