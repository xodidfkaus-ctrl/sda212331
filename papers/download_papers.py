"""
LG EXAONE 논문 일괄 다운로드 + 텍스트/이미지 추출 스크립트

실행:
    cd /home/elicer/sda212331
    python3 papers/download_papers.py

결과 구조:
    papers/
      {arxiv_id}/
        paper.pdf          ← 원본 PDF
        metadata.json      ← 제목/저자/요약/URL
        fulltext.txt       ← 전체 텍스트
        images/
          page{N}_img{M}.png  ← PDF에서 추출된 이미지
"""
import json
import time
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF

PAPERS_DIR = Path(__file__).parent

PAPERS = [
    {
        'arxiv_id': '2604.08644',
        'title': 'EXAONE 4.5 Technical Report',
        'year': 2026,
        'note': '우리가 분석 중인 모델. VLM (33B LM + Vision 1.29B), NoPE Global attention, SWA 4096, 262K context',
    },
    {
        'arxiv_id': '2507.11407',
        'title': 'EXAONE 4.0: Unified LLMs Integrating Non-reasoning and Reasoning Modes',
        'year': 2025,
        'note': 'EXAONE 4.5의 베이스 LM. Reasoning mode가 어떻게 설계됐는지 확인 (주제 C 연구 배경)',
    },
    {
        'arxiv_id': '2601.01739',
        'title': 'K-EXAONE Technical Report',
        'year': 2026,
        'note': 'MoE 구조 (236B total, 23B active). 한국어 특화 모델 — 주제 D(한국어 vs 영어 레이어 전문화) 참고',
    },
    {
        'arxiv_id': '2503.12524',
        'title': 'EXAONE Deep: Reasoning Enhanced Language Models',
        'year': 2025,
        'note': 'EXAONE의 추론 강화 버전. 주제 C(Reasoning 모드 attention 변화) 연구의 직접 배경',
    },
    {
        'arxiv_id': '2412.04862',
        'title': 'EXAONE 3.5: Series of Large Language Models for Real-world Use Cases',
        'year': 2024,
        'note': '32B/7.8B/2.4B. long-context comprehension 강조 — Exp 3/4 장거리 의존성 연구 비교 기준',
    },
    {
        'arxiv_id': '2408.03541',
        'title': 'EXAONE 3.0 7.8B Instruction Tuned Language Model',
        'year': 2024,
        'note': 'EXAONE 오픈소스 1호 모델. 초기 아키텍처 설계 철학 이해용',
    },
]


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    url = f'https://arxiv.org/pdf/{arxiv_id}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research)'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        print(f'  Downloaded: {dest} ({dest.stat().st_size // 1024} KB)')
        return True
    except Exception as e:
        print(f'  ERROR downloading {arxiv_id}: {e}')
        return False


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return '\n'.join(pages)


def extract_images(pdf_path: Path, img_dir: Path, min_size: int = 10000):
    """Extract images larger than min_size bytes from PDF."""
    img_dir.mkdir(exist_ok=True)
    doc = fitz.open(str(pdf_path))
    saved = 0
    for page_num, page in enumerate(doc):
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
                img_bytes = base_img['image']
                if len(img_bytes) < min_size:
                    continue  # 너무 작은 아이콘/장식 이미지 제외
                ext = base_img['ext']
                fname = img_dir / f'page{page_num+1:03d}_img{img_idx+1:02d}.{ext}'
                fname.write_bytes(img_bytes)
                saved += 1
            except Exception:
                pass
    doc.close()
    return saved


def process_paper(paper: dict):
    arxiv_id = paper['arxiv_id']
    paper_dir = PAPERS_DIR / arxiv_id
    paper_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{arxiv_id}] {paper['title']}")

    pdf_path = paper_dir / 'paper.pdf'
    if pdf_path.exists() and pdf_path.stat().st_size > 50000:
        print(f'  PDF already exists, skipping download.')
    else:
        if not download_pdf(arxiv_id, pdf_path):
            return
        time.sleep(2)  # arXiv rate limit 존중

    # 텍스트 추출
    text_path = paper_dir / 'fulltext.txt'
    if not text_path.exists():
        try:
            text = extract_text(pdf_path)
            text_path.write_text(text, encoding='utf-8')
            print(f'  Text: {len(text):,} chars → {text_path.name}')
        except Exception as e:
            print(f'  Text extraction failed: {e}')

    # 이미지 추출
    img_dir = paper_dir / 'images'
    if not img_dir.exists() or not any(img_dir.iterdir() if img_dir.exists() else []):
        try:
            n = extract_images(pdf_path, img_dir)
            print(f'  Images: {n} saved → images/')
        except Exception as e:
            print(f'  Image extraction failed: {e}')
    else:
        existing = list(img_dir.iterdir())
        print(f'  Images: {len(existing)} already extracted')

    # 메타데이터 저장
    meta_path = paper_dir / 'metadata.json'
    meta = {
        'arxiv_id': arxiv_id,
        'title': paper['title'],
        'year': paper['year'],
        'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}',
        'abs_url': f'https://arxiv.org/abs/{arxiv_id}',
        'research_relevance': paper['note'],
        'pdf_size_kb': pdf_path.stat().st_size // 1024 if pdf_path.exists() else 0,
        'text_chars': len(text_path.read_text(encoding='utf-8')) if text_path.exists() else 0,
        'images_count': len(list(img_dir.iterdir())) if img_dir.exists() else 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  Metadata saved.')


def main():
    print(f"LG EXAONE 논문 다운로드 ({len(PAPERS)}편)")
    print(f"저장 위치: {PAPERS_DIR}\n")

    success = 0
    for paper in PAPERS:
        try:
            process_paper(paper)
            success += 1
        except Exception as e:
            print(f'  FAILED [{paper["arxiv_id"]}]: {e}')

    # 전체 인덱스 저장
    index = []
    for paper in PAPERS:
        paper_dir = PAPERS_DIR / paper['arxiv_id']
        meta_file = paper_dir / 'metadata.json'
        if meta_file.exists():
            index.append(json.loads(meta_file.read_text()))

    index_path = PAPERS_DIR / 'index.json'
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"완료: {success}/{len(PAPERS)}편 처리")
    print(f"인덱스: {index_path}")
    print(f"\n논문 목록:")
    for m in index:
        print(f"  [{m['arxiv_id']}] {m['title']}")
        print(f"    텍스트 {m['text_chars']:,}자 / 이미지 {m['images_count']}장")


if __name__ == '__main__':
    main()
