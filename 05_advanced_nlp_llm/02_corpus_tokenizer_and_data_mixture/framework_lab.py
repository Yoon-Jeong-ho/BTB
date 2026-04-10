from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'

RAW_DOCUMENTS = [
    ('ko-news-001', 'ko', 'news', '반도체 시장 보고서: 메모리 수요는 클라우드 학습 워크로드와 함께 증가한다.'),
    ('ko-news-001-copy', 'ko', 'news', '반도체 시장 보고서: 메모리 수요는 클라우드 학습 워크로드와 함께 증가한다.'),
    ('en-forum-001', 'en', 'forum', 'Tokenizer tradeoffs matter for multilingual models when Korean and English share a vocabulary.'),
    ('en-forum-002', 'en', 'forum', 'Tokenizer tradeoffs matter for multilingual model design when Korean and English share vocabulary.'),
    ('ko-docs-001', 'ko', 'docs', '사용자 안내서: 데이터 중복 제거와 오염 검사를 분리하고 출처 메타데이터를 보존하세요.'),
    ('en-academic-001', 'en', 'academic', 'Domain adaptation benefits when the corpus matches downstream terminology and evaluation tasks.'),
    ('ja-news-001', 'ja', 'news', '多言語コーパスでは言語ごとのトークン効率と品質を一緒に確認する。'),
    ('ko-dialogue-001', 'ko', 'dialogue', '사용자: 모델이 왜 같은 문서를 반복해서 외우나요? 튜터: 중복 데이터가 gradient를 과하게 반복시킵니다.'),
    ('en-eval-001', 'en', 'eval_overlap', 'Evaluation snippet: What is the capital of France? The capital of France is Paris.'),
    ('ko-eval-001', 'ko', 'eval_overlap', '평가 문항 예시: 프랑스의 수도는 파리입니다. 이 문장은 contamination 점검에 사용된다.'),
    ('code-docs-001', 'en', 'code_docs', 'def tokenize(text): return text.lower().split()  # code docs show API examples'),
]

TOKEN_RE = re.compile(r'[A-Za-z0-9_]+|[가-힣]+|[ぁ-んァ-ヶ一-龯]+|[^\s]')
WORD_RE = re.compile(r'[A-Za-z0-9_가-힣ぁ-んァ-ヶ一-龯]+')
CONTAMINATION_PATTERNS = ('capital of france', '프랑스의 수도')


class ToyTokenizer:
    def __init__(self, name: str, mode: str) -> None:
        self.name = name
        self.mode = mode

    def tokenize(self, text: str) -> list[str]:
        if self.mode == 'unigram':
            return [piece for piece in (raw.lower().strip('.,:;?!()#`"\'') for raw in text.split()) if piece]
        if self.mode == 'aggressive':
            pieces: list[str] = []
            for token in [token.lower() for token in TOKEN_RE.findall(text)]:
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
        return [token.lower() for token in TOKEN_RE.findall(text)]


def _normalize(text: str) -> str:
    return ' '.join(token.lower() for token in WORD_RE.findall(text))


def _similarity(left: str, right: str) -> float:
    left_words = set(_normalize(left).split())
    right_words = set(_normalize(right).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def _is_contaminated(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in CONTAMINATION_PATTERNS)


def _curate_documents() -> dict[str, object]:
    docs = [
        {'id': doc_id, 'language': language, 'domain': domain, 'text': text}
        for doc_id, language, domain, text in RAW_DOCUMENTS
    ]

    exact_seen: set[str] = set()
    exact_clean: list[dict[str, str]] = []
    removed_exact: list[str] = []
    for doc in docs:
        signature = _normalize(doc['text'])
        if signature in exact_seen:
            removed_exact.append(doc['id'])
            continue
        exact_seen.add(signature)
        exact_clean.append(doc)

    near_clean: list[dict[str, str]] = []
    removed_near: list[dict[str, object]] = []
    for doc in exact_clean:
        duplicate_of = None
        duplicate_score = 0.0
        for kept in near_clean:
            score = _similarity(doc['text'], kept['text'])
            if score >= 0.66:
                duplicate_of = kept
                duplicate_score = score
                break
        if duplicate_of is None:
            near_clean.append(doc)
        else:
            removed_near.append({'dropped': doc['id'], 'kept': duplicate_of['id'], 'jaccard': round(duplicate_score, 3)})

    contamination_hits = [doc['id'] for doc in near_clean if _is_contaminated(doc['text'])]
    clean = [doc for doc in near_clean if doc['id'] not in contamination_hits]
    return {
        'clean': clean,
        'removed_exact': removed_exact,
        'removed_near': removed_near,
        'contamination_hits': contamination_hits,
    }


def _normalized_share(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    keys = sorted(counts)
    shares = {key: round(counts[key] / total, 6) for key in keys}
    if keys:
        shares[keys[-1]] = round(shares[keys[-1]] + round(1.0 - sum(shares.values()), 6), 6)
    return shares


def _token_share(documents: list[dict[str, str]], tokenizer: ToyTokenizer, field: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc[field]] = counts.get(doc[field], 0) + len(tokenizer.tokenize(doc['text']))
    return _normalized_share(counts)


def _stats_for_tokenizer(documents: list[dict[str, str]], tokenizer: ToyTokenizer) -> dict[str, object]:
    tokenized = [tokenizer.tokenize(doc['text']) for doc in documents]
    avg_tokens = mean(len(tokens) for tokens in tokenized)
    by_language: dict[str, list[float]] = {}
    for doc, tokens in zip(documents, tokenized):
        chars = max(1, len(doc['text'].replace(' ', '')))
        by_language.setdefault(doc['language'], []).append(len(tokens) / chars)
    return {
        'avg_tokens_per_doc': round(avg_tokens, 3),
        'vocab_size': len({piece for tokens in tokenized for piece in tokens}),
        'fragmentation_by_language': {lang: round(mean(values), 3) for lang, values in sorted(by_language.items())},
    }


def _make_batch_preview(documents: list[dict[str, str]], tokenizer: ToyTokenizer) -> list[dict[str, object]]:
    preferred_domains = ['news', 'docs', 'academic', 'code_docs']
    preview = []
    for domain in preferred_domains:
        doc = next(item for item in documents if item['domain'] == domain)
        tokens = tokenizer.tokenize(doc['text'])
        preview.append(
            {
                'doc_id': doc['id'],
                'language': doc['language'],
                'domain': doc['domain'],
                'token_count': len(tokens),
                'first_tokens': tokens[:6],
            }
        )
    return preview


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curation = _curate_documents()
    clean_docs = curation['clean']
    tokenizers = {
        'toy_whitespace': ToyTokenizer('toy_whitespace', 'whitespace'),
        'toy_unigram_like': ToyTokenizer('toy_unigram_like', 'unigram'),
        'toy_aggressive_subword': ToyTokenizer('toy_aggressive_subword', 'aggressive'),
    }
    selected = tokenizers['toy_unigram_like']
    avg_tokens = _stats_for_tokenizer(clean_docs, selected)['avg_tokens_per_doc']
    context_window = 64

    metrics = {
        'device': 'cpu',
        'tokenizer_name': selected.name,
        'raw_document_count': len(RAW_DOCUMENTS),
        'accepted_document_count': len(clean_docs),
        'removed_exact_duplicates': len(curation['removed_exact']),
        'removed_exact_ids': curation['removed_exact'],
        'removed_near_duplicates': len(curation['removed_near']),
        'removed_near_pairs': curation['removed_near'],
        'contamination_blocked': len(curation['contamination_hits']),
        'contamination_hit_ids': curation['contamination_hits'],
        'language_token_share': _token_share(clean_docs, selected, 'language'),
        'domain_token_share': _token_share(clean_docs, selected, 'domain'),
        'tokenizer_stats': {name: _stats_for_tokenizer(clean_docs, tokenizer) for name, tokenizer in tokenizers.items()},
        'batch_preview': _make_batch_preview(clean_docs, selected),
        'token_budget': {
            'context_window': context_window,
            'avg_tokens_per_doc': avg_tokens,
            'docs_per_context_estimate': int(context_window // float(avg_tokens)),
            'steps_for_1024_tokens_at_batch4': math.ceil(1024 / (context_window * 4)),
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
