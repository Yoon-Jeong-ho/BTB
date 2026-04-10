from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
FIGURE_PATH = OUT_DIR / 'corpus_quality_overview.svg'

RAW_DOCUMENTS = [
    {
        'id': 'ko-news-001',
        'language': 'ko',
        'domain': 'news',
        'quality': 0.92,
        'text': '반도체 시장 보고서: 메모리 수요는 클라우드 학습 워크로드와 함께 증가한다.',
    },
    {
        'id': 'ko-news-001-copy',
        'language': 'ko',
        'domain': 'news',
        'quality': 0.92,
        'text': '반도체 시장 보고서: 메모리 수요는 클라우드 학습 워크로드와 함께 증가한다.',
    },
    {
        'id': 'en-forum-001',
        'language': 'en',
        'domain': 'forum',
        'quality': 0.74,
        'text': 'Tokenizer tradeoffs matter for multilingual models when Korean and English share a vocabulary.',
    },
    {
        'id': 'en-forum-002',
        'language': 'en',
        'domain': 'forum',
        'quality': 0.72,
        'text': 'Tokenizer tradeoffs matter for multilingual model design when Korean and English share vocabulary.',
    },
    {
        'id': 'ko-docs-001',
        'language': 'ko',
        'domain': 'docs',
        'quality': 0.88,
        'text': '사용자 안내서: 데이터 중복 제거와 오염 검사를 분리하고 출처 메타데이터를 보존하세요.',
    },
    {
        'id': 'en-academic-001',
        'language': 'en',
        'domain': 'academic',
        'quality': 0.90,
        'text': 'Domain adaptation benefits when the corpus matches downstream terminology and evaluation tasks.',
    },
    {
        'id': 'ja-news-001',
        'language': 'ja',
        'domain': 'news',
        'quality': 0.81,
        'text': '多言語コーパスでは言語ごとのトークン効率と品質を一緒に確認する。',
    },
    {
        'id': 'ko-dialogue-001',
        'language': 'ko',
        'domain': 'dialogue',
        'quality': 0.79,
        'text': '사용자: 모델이 왜 같은 문서를 반복해서 외우나요? 튜터: 중복 데이터가 gradient를 과하게 반복시킵니다.',
    },
    {
        'id': 'en-eval-001',
        'language': 'en',
        'domain': 'eval_overlap',
        'quality': 0.60,
        'text': 'Evaluation snippet: What is the capital of France? The capital of France is Paris.',
    },
    {
        'id': 'ko-eval-001',
        'language': 'ko',
        'domain': 'eval_overlap',
        'quality': 0.60,
        'text': '평가 문항 예시: 프랑스의 수도는 파리입니다. 이 문장은 contamination 점검에 사용된다.',
    },
    {
        'id': 'code-docs-001',
        'language': 'en',
        'domain': 'code_docs',
        'quality': 0.84,
        'text': 'def tokenize(text): return text.lower().split()  # code docs show API examples',
    },
]

TOKEN_RE = re.compile(r'[A-Za-z0-9_]+|[가-힣]+|[ぁ-んァ-ヶ一-龯]+|[^\s]')
WORD_RE = re.compile(r'[A-Za-z0-9_가-힣ぁ-んァ-ヶ一-龯]+')
CONTAMINATION_PATTERNS = ('capital of france', '프랑스의 수도')


def _normalize(text: str) -> str:
    return ' '.join(token.lower() for token in WORD_RE.findall(text))


def _words(text: str) -> set[str]:
    return set(_normalize(text).split())


def _similarity(left: str, right: str) -> float:
    left_words = _words(left)
    right_words = _words(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def _is_contaminated(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in CONTAMINATION_PATTERNS)


def _dedup_and_filter(documents: list[dict[str, object]]) -> dict[str, object]:
    exact_seen: set[str] = set()
    exact_clean: list[dict[str, object]] = []
    exact_removed: list[str] = []
    for doc in documents:
        signature = _normalize(str(doc['text']))
        if signature in exact_seen:
            exact_removed.append(str(doc['id']))
            continue
        exact_seen.add(signature)
        exact_clean.append(doc)

    near_clean: list[dict[str, object]] = []
    near_removed: list[dict[str, object]] = []
    for doc in exact_clean:
        matched_doc = None
        matched_score = 0.0
        for kept in near_clean:
            score = _similarity(str(doc['text']), str(kept['text']))
            if score >= 0.66:
                matched_doc = kept
                matched_score = score
                break
        if matched_doc is None:
            near_clean.append(doc)
        else:
            near_removed.append(
                {
                    'dropped': str(doc['id']),
                    'kept': str(matched_doc['id']),
                    'jaccard': round(matched_score, 3),
                }
            )

    contamination_hits = [str(doc['id']) for doc in near_clean if _is_contaminated(str(doc['text']))]
    clean = [doc for doc in near_clean if str(doc['id']) not in contamination_hits]
    return {
        'clean_documents': clean,
        'exact_removed': exact_removed,
        'near_removed': near_removed,
        'contamination_hits': contamination_hits,
    }


def toy_whitespace(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def toy_unigram_like(text: str) -> list[str]:
    # A coarse toy tokenizer: whitespace pieces after punctuation trimming.
    pieces = []
    for raw in text.lower().split():
        token = raw.strip('.,:;?!()#`"\'')
        if token:
            pieces.append(token)
    return pieces


def toy_aggressive_subword(text: str) -> list[str]:
    pieces: list[str] = []
    for token in toy_whitespace(text):
        if re.fullmatch(r'[A-Za-z0-9_]+', token):
            step = 4
        elif re.fullmatch(r'[가-힣]+', token):
            step = 2
        elif re.fullmatch(r'[ぁ-んァ-ヶ一-龯]+', token):
            step = 1
        else:
            step = 1
        pieces.extend(token[i : i + step] for i in range(0, len(token), step))
    return pieces


TOKENIZERS = {
    'toy_whitespace': toy_whitespace,
    'toy_unigram_like': toy_unigram_like,
    'toy_aggressive_subword': toy_aggressive_subword,
}


def _tokenizer_stats(documents: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for name, tokenizer in TOKENIZERS.items():
        token_lengths = [len(tokenizer(str(doc['text']))) for doc in documents]
        char_lengths = [len(str(doc['text']).replace(' ', '')) for doc in documents]
        language_fragmentation: dict[str, list[float]] = {}
        for doc in documents:
            lang = str(doc['language'])
            tokens = tokenizer(str(doc['text']))
            chars = max(1, len(str(doc['text']).replace(' ', '')))
            language_fragmentation.setdefault(lang, []).append(len(tokens) / chars)
        stats[name] = {
            'avg_tokens_per_doc': round(mean(token_lengths), 3),
            'chars_per_token': round(sum(char_lengths) / max(1, sum(token_lengths)), 3),
            'vocab_size': len({piece for doc in documents for piece in tokenizer(str(doc['text']))}),
            'fragmentation_by_language': {
                lang: round(mean(values), 3) for lang, values in sorted(language_fragmentation.items())
            },
        }
    return stats


def _normalized_share(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    keys = sorted(counts)
    shares = {key: round(counts[key] / total, 6) for key in keys}
    residual = round(1.0 - sum(shares.values()), 6)
    if keys:
        shares[keys[-1]] = round(shares[keys[-1]] + residual, 6)
    return shares


def _token_share(documents: list[dict[str, object]], field: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for doc in documents:
        key = str(doc[field])
        counts[key] = counts.get(key, 0) + len(toy_unigram_like(str(doc['text'])))
    return _normalized_share(counts)


def _write_svg(raw_count: int, after_dedup: int, clean_count: int, contamination_count: int) -> None:
    values = [
        ('raw docs', raw_count, '#64748b'),
        ('after dedup', after_dedup, '#2563eb'),
        ('trainable', clean_count, '#16a34a'),
        ('blocked eval', contamination_count, '#dc2626'),
    ]
    max_value = max(value for _, value, _ in values)
    rows = []
    for idx, (label, value, color) in enumerate(values):
        y = 58 + idx * 42
        width = int(260 * value / max_value)
        rows.append(
            f'<text x="24" y="{y + 16}" font-size="13">{label}</text>'
            f'<rect x="130" y="{y}" width="{width}" height="24" fill="{color}" rx="4" />'
            f'<text x="{140 + width}" y="{y + 16}" font-size="13">{value}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="460" height="250" viewBox="0 0 460 250">
  <title>Corpus quality and mixture overview</title>
  <rect width="460" height="250" fill="#f8fafc" />
  <text x="24" y="32" font-size="18" font-weight="700">Corpus quality and mixture overview</text>
  {''.join(rows)}
  <text x="24" y="228" font-size="12" fill="#475569">Toy CPU-safe counts: dedup and contamination reduce usable token budget.</text>
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtering = _dedup_and_filter(RAW_DOCUMENTS)
    clean_documents = filtering['clean_documents']
    exact_removed = filtering['exact_removed']
    near_removed = filtering['near_removed']
    contamination_hits = filtering['contamination_hits']
    after_dedup_count = len(RAW_DOCUMENTS) - len(exact_removed) - len(near_removed)

    tokenizer_stats = _tokenizer_stats(clean_documents)
    whitespace_avg = tokenizer_stats['toy_whitespace']['avg_tokens_per_doc']
    aggressive_avg = tokenizer_stats['toy_aggressive_subword']['avg_tokens_per_doc']
    context_window = 64

    metrics = {
        'seed': 0,
        'raw_document_count': len(RAW_DOCUMENTS),
        'document_count_after_dedup': after_dedup_count,
        'trainable_document_count': len(clean_documents),
        'dedup_removed_documents': len(exact_removed) + len(near_removed),
        'exact_duplicate_ids': exact_removed,
        'near_duplicate_pairs': near_removed,
        'contamination_blocked_documents': len(contamination_hits),
        'contamination_hit_ids': contamination_hits,
        'language_token_share': _token_share(clean_documents, 'language'),
        'mixture_token_share': _token_share(clean_documents, 'domain'),
        'tokenizers': tokenizer_stats,
        'token_budget_demo': {
            'context_window': context_window,
            'toy_whitespace_docs_per_context': int(context_window // whitespace_avg),
            'toy_aggressive_subword_docs_per_context': int(context_window // aggressive_avg),
            'aggressive_token_inflation_vs_whitespace': round(aggressive_avg / whitespace_avg, 3),
        },
        'figure_path': 'artifacts/scratch-manual/corpus_quality_overview.svg',
    }

    _write_svg(len(RAW_DOCUMENTS), after_dedup_count, len(clean_documents), len(contamination_hits))
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
