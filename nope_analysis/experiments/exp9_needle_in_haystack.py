"""
e009: Needle-in-Haystack Retrieval — Behavioral RQ3 Validation

Does Global attention enable retrieval of facts beyond the SWA window (4096 tokens)?

Method:
- Insert a unique code at token position 4200-5000 in a long EDGAR document
- Ask model to retrieve the code (greedy generation)
- Compare: baseline vs Global-ablated (all 16 Global layers zeroed)
- Statistical test: McNemar's test on 50 paired (haystack, needle) trials
"""
import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json, torch, random, re
import numpy as np
from pathlib import Path

from nope_analysis.loader import load_model_and_tokenizer, load_config, get_global_layer_indices
from nope_analysis.seeds import set_all_seeds, EXPERIMENT_SEEDS
from nope_analysis.corpus.downloader import get_text_sample
from nope_analysis.analysis.auto_validate import validate_experiment

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUT_DIR = Path('/home/elicer/sda212331/outputs/e009_needle_in_haystack')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PAIRS      = 50
NEEDLE_POS   = (4200, 5000)   # target needle insertion range (tokens)
TARGET_LEN   = 6500           # total input length
MAX_NEW_TOKENS = 30
BASE_SEED    = EXPERIMENT_SEEDS.get("e009", 20090428)


def make_needle(code: str) -> str:
    return f" The verification code for this document is EXAONE{code}. "


def extract_code(text: str) -> str | None:
    """Extract 6-digit code from generated text."""
    m = re.search(r'EXAONE(\d{6})', text)
    return m.group(1) if m else None


def build_haystack_with_needle(tokenizer, needle_code: str, seed: int):
    """Build (input_ids, needle_token_pos) with needle inserted at target position."""
    needle_str = make_needle(needle_code)

    for lang in ['en_edgar', 'en']:
        try:
            text = get_text_sample(lang, min_tokens=TARGET_LEN + 512,
                                   tokenizer=tokenizer, seed=seed)
            base_ids = tokenizer(text, return_tensors='pt',
                                 add_special_tokens=True)['input_ids'][0]
            if base_ids.shape[0] < TARGET_LEN:
                continue

            # Find insertion point (token-level)
            insert_at = random.randint(*NEEDLE_POS)
            insert_at = min(insert_at, base_ids.shape[0] - 200)

            # Build text with needle injected at character level
            prefix_ids = base_ids[:insert_at]
            suffix_ids = base_ids[insert_at:]
            needle_ids = tokenizer(needle_str, return_tensors='pt',
                                   add_special_tokens=False)['input_ids'][0]

            combined = torch.cat([prefix_ids, needle_ids, suffix_ids])
            combined = combined[:TARGET_LEN].unsqueeze(0)  # (1, TARGET_LEN)

            return combined, insert_at, lang
        except Exception as e:
            print(f"  [corpus] {lang}: {e}")

    raise RuntimeError("Could not build haystack — corpus unavailable")


def make_retrieval_prompt(tokenizer, haystack_ids: torch.Tensor) -> torch.Tensor:
    """Append retrieval question to haystack."""
    question = "\n\nQuestion: What is the verification code mentioned in the document above? Answer with only the code in the format EXAONE followed by 6 digits.\nAnswer: EXAONE"
    q_ids = tokenizer(question, return_tensors='pt',
                      add_special_tokens=False)['input_ids'][0]
    combined = torch.cat([haystack_ids[0], q_ids]).unsqueeze(0)
    return combined


def run_generation(model, tokenizer, input_ids, ablate_global: bool,
                   global_indices: list) -> str:
    """Run greedy generation with optional Global layer ablation."""
    hooks = []
    if ablate_global:
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # output[0] is attention output tensor
                    zeroed = torch.zeros_like(output[0])
                    return (zeroed,) + output[1:]
                return torch.zeros_like(output)
            return hook

        for li in global_indices:
            layer = model.language_model.model.layers[li].self_attn
            h = layer.register_forward_hook(make_hook(li))
            hooks.append(h)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids.to('cuda:0'),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0, input_ids.shape[1]:],
                                     skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()

    return generated


def mcnemar_test(b: int, c: int):
    """McNemar's test: b = baseline_hit & ablated_miss, c = baseline_miss & ablated_hit."""
    from scipy.stats import chi2
    if b + c == 0:
        return float('nan'), float('nan')
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    p_val = 1 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_val)


def main():
    print("e009 — Needle-in-Haystack Retrieval")
    print(f"N_PAIRS={N_PAIRS}, TARGET_LEN={TARGET_LEN}, NEEDLE_POS={NEEDLE_POS}\n")

    set_all_seeds(BASE_SEED)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    global_indices = get_global_layer_indices()
    print(f"Global layers: {global_indices}")
    print(f"Model loaded.\n")

    results = []
    baseline_hits = 0
    ablated_hits = 0

    for pair_idx in range(N_PAIRS):
        code = f"{random.randint(0, 999999):06d}"
        seed = BASE_SEED + pair_idx

        try:
            haystack_ids, needle_pos, lang = build_haystack_with_needle(
                tokenizer, code, seed=seed)
            prompt_ids = make_retrieval_prompt(tokenizer, haystack_ids)

            # Baseline
            gen_base = run_generation(model, tokenizer, prompt_ids,
                                      ablate_global=False,
                                      global_indices=global_indices)
            hit_base = int(code in gen_base)

            # Ablated
            gen_ablated = run_generation(model, tokenizer, prompt_ids,
                                         ablate_global=True,
                                         global_indices=global_indices)
            hit_ablated = int(code in gen_ablated)

            baseline_hits += hit_base
            ablated_hits += hit_ablated

            r = {
                'pair_idx': pair_idx,
                'code': code,
                'needle_token_pos': needle_pos,
                'lang': lang,
                'hit_baseline': hit_base,
                'hit_ablated': hit_ablated,
                'gen_baseline': gen_base[:80],
                'gen_ablated': gen_ablated[:80],
            }
            results.append(r)

            with open(OUT_DIR / 'results.jsonl', 'a') as f:
                f.write(json.dumps(r) + '\n')

            print(f"  pair {pair_idx:02d} | code={code} | "
                  f"base={'HIT' if hit_base else 'miss'} | "
                  f"ablated={'HIT' if hit_ablated else 'miss'} | "
                  f"lang={lang}")

        except Exception as e:
            print(f"  pair {pair_idx:02d} | ERROR: {e}")

    # McNemar's test
    b = sum(1 for r in results if r['hit_baseline'] and not r['hit_ablated'])
    c = sum(1 for r in results if not r['hit_baseline'] and r['hit_ablated'])
    chi2_stat, p_val = mcnemar_test(b, c)

    n_complete = len(results)
    hit_rate_base = baseline_hits / n_complete if n_complete else 0
    hit_rate_ablated = ablated_hits / n_complete if n_complete else 0

    summary = {
        'experiment_id': 'e009',
        'n_pairs': n_complete,
        'hit_rate_baseline': hit_rate_base,
        'hit_rate_ablated': hit_rate_ablated,
        'delta_hit_rate': hit_rate_base - hit_rate_ablated,
        'mcnemar_b': b,
        'mcnemar_c': c,
        'mcnemar_chi2': chi2_stat,
        'p_value': p_val,
        'verdict': (
            'VALIDATED' if (p_val <= 0.05 and hit_rate_base >= 0.3
                            and hit_rate_base > hit_rate_ablated)
            else ('INCONCLUSIVE' if hit_rate_base < 0.3
                  else 'FAILED')
        ),
    }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Baseline hit rate:  {hit_rate_base:.3f} ({baseline_hits}/{n_complete})")
    print(f"  Ablated hit rate:   {hit_rate_ablated:.3f} ({ablated_hits}/{n_complete})")
    print(f"  Delta:              {hit_rate_base - hit_rate_ablated:+.3f}")
    print(f"  McNemar: chi2={chi2_stat:.3f}, p={p_val:.4f} (b={b}, c={c})")
    print(f"  Verdict: {summary['verdict']}")

    validate_experiment(
        experiment_id='e009',
        results_path=OUT_DIR / 'results.jsonl',
        repo_root=Path('/home/elicer/sda212331'),
    )

    print(f"\nOutputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
