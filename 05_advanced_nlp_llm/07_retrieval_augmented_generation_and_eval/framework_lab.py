from __future__ import annotations

import json
import math
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
TOP_K = 3
EMBEDDING_DIM = 8

DOCS = [
    ('travel_2026_limits', [0.92, 0.88, 0.76, 0.12, 0.10, 0.08, 0.16, 0.34]),
    ('travel_2024_old', [0.82, 0.78, 0.66, 0.10, 0.09, 0.08, 0.14, 0.22]),
    ('refund_window', [0.10, 0.18, 0.08, 0.91, 0.86, 0.72, 0.15, 0.26]),
    ('security_retention', [0.08, 0.16, 0.18, 0.12, 0.17, 0.20, 0.94, 0.88]),
    ('laptop_procurement', [0.20, 0.24, 0.18, 0.16, 0.10, 0.82, 0.18, 0.32]),
    ('rag_eval_note', [0.14, 0.12, 0.18, 0.20, 0.24, 0.16, 0.84, 0.92]),
]

QUERIES = [
    ('travel_limit_latest', [0.95, 0.82, 0.72, 0.08, 0.11, 0.06, 0.18, 0.28], ['travel_2026_limits']),
    ('refund_window', [0.12, 0.16, 0.08, 0.94, 0.90, 0.70, 0.14, 0.24], ['refund_window']),
    ('log_retention', [0.08, 0.14, 0.16, 0.12, 0.20, 0.18, 0.96, 0.82], ['security_retention']),
    ('meal_policy_missing', [0.46, 0.34, 0.30, 0.28, 0.18, 0.22, 0.20, 0.26], []),
]


def _round(value: float) -> float:
    return round(value, 6)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    return _dot(left, right) / (_norm(left) * _norm(right))


def _rank_all(query_vector: list[float]) -> list[tuple[str, float]]:
    scores = [(doc_id, _round(_cosine(query_vector, doc_vector))) for doc_id, doc_vector in DOCS]
    return sorted(scores, key=lambda row: (-row[1], row[0]))


def _dcg(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    return sum((1.0 if doc_id in relevant else 0.0) / math.log2(index + 2) for index, doc_id in enumerate(ranked_ids))


def _retrieval_metrics(rankings: list[dict[str, object]]) -> dict[str, float]:
    answerable = [row for row in rankings if row['relevant_ids']]
    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for row in answerable:
        relevant = list(row['relevant_ids'])
        ranked_ids = list(row['ranked_ids'])
        recall_at_1 += int(any(doc_id in relevant for doc_id in ranked_ids[:1]))
        recall_at_3 += int(any(doc_id in relevant for doc_id in ranked_ids[:TOP_K]))
        rank = next((index + 1 for index, doc_id in enumerate(ranked_ids) if doc_id in relevant), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        ideal = _dcg(relevant[:TOP_K], relevant)
        ndcgs.append(0.0 if ideal == 0.0 else _dcg(ranked_ids[:TOP_K], relevant) / ideal)
    denominator = len(answerable)
    return {
        'recall_at_1': _round(recall_at_1 / denominator),
        'recall_at_3': _round(recall_at_3 / denominator),
        'mrr': _round(sum(reciprocal_ranks) / denominator),
        'ndcg_at_3': _round(sum(ndcgs) / denominator),
    }


def run() -> None:
    rankings: list[dict[str, object]] = []
    for query_id, query_vector, relevant_ids in QUERIES:
        ranked = _rank_all(query_vector)
        rankings.append(
            {
                'query_id': query_id,
                'relevant_ids': relevant_ids,
                'ranked_ids': [doc_id for doc_id, _score in ranked],
                'topk_indices': [next(index for index, (doc_id, _vector) in enumerate(DOCS) if doc_id == ranked_id) for ranked_id, _score in ranked[:TOP_K]],
                'topk_scores': [score for _doc_id, score in ranked[:TOP_K]],
            }
        )

    retrieval = _retrieval_metrics(rankings)
    claim_count = 8
    unsupported_claims = 1
    groundedness = (claim_count - unsupported_claims) / claim_count
    citation_precision = 3 / 3
    citation_coverage = 3 / 3
    correction_rate = 0.12 + unsupported_claims / claim_count * 0.2

    metrics = {
        'device': 'cpu',
        'simulation': 'deterministic_lightweight_rag',
        'seed': 0,
        'embedding_dim': EMBEDDING_DIM,
        'doc_count': len(DOCS),
        'query_count': len(QUERIES),
        'top_k': TOP_K,
        'batch_shapes': {
            'query_embeddings': [len(QUERIES), EMBEDDING_DIM],
            'doc_embeddings': [len(DOCS), EMBEDDING_DIM],
            'topk_indices': [len(QUERIES), TOP_K],
            'prompt_tokens': [len(QUERIES), 192],
        },
        'rankings': rankings,
        'retrieval_metrics': retrieval,
        'answer_metrics': {
            'claim_count': claim_count,
            'unsupported_claims': unsupported_claims,
            'unsupported_claim_rate': _round(unsupported_claims / claim_count),
            'groundedness': _round(groundedness),
            'citation_precision': _round(citation_precision),
            'citation_coverage': _round(citation_coverage),
            'answer_correctness': 0.75,
            'abstention_accuracy': 1.0,
        },
        'context_injection': {
            'metadata_included': True,
            'citation_tags_required': True,
            'stale_source_penalty_enabled': True,
            'prompt_template': 'system grounding rule + source-tagged context + user query',
            'max_evidence_chunks': TOP_K,
            'evidence_tokens_mean': 84,
        },
        'failure_mode_probes': {
            'highest_risk': 'unsupported_claim',
            'missing_evidence_queries': 1,
            'stale_source_queries': 1,
            'irrelevant_context_flag': True,
            'citation_without_support_flag': True,
            'abstention_rate': 0.25,
        },
        'eval_harness': {
            'offline': {
                'retriever_recall_at_3': retrieval['recall_at_3'],
                'mrr': retrieval['mrr'],
                'groundedness': _round(groundedness),
                'unsupported_claim_rate': _round(unsupported_claims / claim_count),
            },
            'online': {
                'acceptance_proxy': 0.68,
                'citation_click_proxy': 0.34,
                'correction_rate_proxy': _round(correction_rate),
                'latency_ms_p50': 145,
            },
            'metric_split_note': 'offline retrieval quality, answer grounding, and online user behavior are complementary guardrails.',
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'device': metrics['device'],
        'simulation': metrics['simulation'],
        'retrieval_metrics': metrics['retrieval_metrics'],
        'answer_metrics': metrics['answer_metrics'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
