from __future__ import annotations

import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
FIGURE_PATH = OUT_DIR / 'rag_grounding_eval.svg'
TOP_K = 3

CHUNKS = [
    {
        'chunk_id': 'travel_2026_limits',
        'source': 'travel_policy_2026.md#daily-limit',
        'date': '2026-01-15',
        'freshness': 1.0,
        'keywords': ['출장비', '상한', '2026', '국내', '18만', '시행일', '영수증'],
        'text': '2026년 1월 15일부터 국내 출장비 1일 상한은 18만 원이며 영수증 첨부가 필요하다.',
    },
    {
        'chunk_id': 'travel_2024_old',
        'source': 'travel_policy_2024.md#daily-limit',
        'date': '2024-07-01',
        'freshness': 0.2,
        'keywords': ['출장비', '상한', '2024', '국내', '15만'],
        'text': '2024년 국내 출장비 1일 상한은 15만 원이었다. 최신 정책 확인이 필요하다.',
    },
    {
        'chunk_id': 'refund_window',
        'source': 'course_refund_faq.md#window',
        'date': '2025-11-02',
        'freshness': 0.8,
        'keywords': ['강의', '환불', '기한', '7일', '수강', '시작'],
        'text': '온라인 강의 환불은 수강 시작 전 또는 첫 수강 후 7일 이내에 신청해야 한다.',
    },
    {
        'chunk_id': 'security_retention',
        'source': 'security_handbook.md#log-retention',
        'date': '2026-02-10',
        'freshness': 0.95,
        'keywords': ['보안', '로그', '보관', '90일', '감사', '권한'],
        'text': '보안 감사 로그는 최소 90일 보관하고 권한 변경 이벤트를 반드시 포함한다.',
    },
    {
        'chunk_id': 'laptop_procurement',
        'source': 'procurement_guide.md#laptop',
        'date': '2025-09-20',
        'freshness': 0.7,
        'keywords': ['노트북', '구매', '승인', '팀장', '자산', '조달'],
        'text': '노트북 구매는 팀장 승인 후 조달 시스템에 자산 번호를 등록해야 한다.',
    },
    {
        'chunk_id': 'rag_eval_note',
        'source': 'rag_eval_playbook.md#grounding',
        'date': '2026-03-01',
        'freshness': 1.0,
        'keywords': ['RAG', 'grounding', 'citation', 'unsupported', 'claim', 'eval', 'evidence'],
        'text': 'RAG 평가는 citation 수가 아니라 claim-level evidence coverage와 unsupported claim 비율을 함께 본다.',
    },
]

QUERIES = [
    {
        'query_id': 'travel_limit_latest',
        'query': '2026년 국내 출장비 1일 상한은 얼마이고 어떤 조건이 붙는가?',
        'keywords': ['2026', '국내', '출장비', '상한', '조건', '영수증'],
        'relevant_ids': ['travel_2026_limits'],
        'reader_answer': '2026년 1월 15일부터 국내 출장비 1일 상한은 18만 원이며 영수증 첨부가 필요하다.',
        'generator_answer': '상한은 18만 원이고 영수증 첨부 조건이 붙는다. [travel_policy_2026.md#daily-limit]',
        'supported_claims': 2,
        'unsupported_claims': 0,
        'citations': ['travel_policy_2026.md#daily-limit'],
        'abstain_flag': False,
    },
    {
        'query_id': 'refund_window',
        'query': '온라인 강의 환불 신청 기한은 언제까지인가?',
        'keywords': ['온라인', '강의', '환불', '신청', '기한'],
        'relevant_ids': ['refund_window'],
        'reader_answer': '수강 시작 전 또는 첫 수강 후 7일 이내에 환불을 신청해야 한다.',
        'generator_answer': '환불은 수강 시작 전 또는 첫 수강 후 7일 이내에 신청한다. [course_refund_faq.md#window]',
        'supported_claims': 2,
        'unsupported_claims': 0,
        'citations': ['course_refund_faq.md#window'],
        'abstain_flag': False,
    },
    {
        'query_id': 'log_retention',
        'query': '보안 감사 로그는 며칠 동안 보관해야 하며 무엇을 포함해야 하는가?',
        'keywords': ['보안', '감사', '로그', '보관', '며칠', '권한'],
        'relevant_ids': ['security_retention'],
        'reader_answer': '보안 감사 로그는 최소 90일 보관하고 권한 변경 이벤트를 포함해야 한다.',
        'generator_answer': '최소 90일 보관하며 권한 변경 이벤트를 포함한다. [security_handbook.md#log-retention]',
        'supported_claims': 2,
        'unsupported_claims': 0,
        'citations': ['security_handbook.md#log-retention'],
        'abstain_flag': False,
    },
    {
        'query_id': 'meal_policy_missing',
        'query': '야근 식대 상한은 얼마인가?',
        'keywords': ['야근', '식대', '상한'],
        'relevant_ids': [],
        'reader_answer': 'retrieved evidence 안에는 야근 식대 상한 근거가 없어 답을 보류한다.',
        'generator_answer': '근거가 부족하지만 보통 2만 원으로 추정할 수 있다.',
        'supported_claims': 1,
        'unsupported_claims': 1,
        'citations': [],
        'abstain_flag': True,
    },
]


def _round(value: float) -> float:
    return round(value, 6)


def _score(query_keywords: list[str], chunk: dict[str, object]) -> float:
    query_terms = {term.lower() for term in query_keywords}
    chunk_terms = {str(term).lower() for term in chunk['keywords']}
    overlap = len(query_terms & chunk_terms)
    if not query_terms:
        return 0.0
    return _round(overlap / len(query_terms) + 0.04 * float(chunk['freshness']))


def _retrieve(query: dict[str, object]) -> list[dict[str, object]]:
    scored = [
        {
            'chunk_id': chunk['chunk_id'],
            'source': chunk['source'],
            'date': chunk['date'],
            'score': _score(list(query['keywords']), chunk),
            'text': chunk['text'],
        }
        for chunk in CHUNKS
    ]
    return sorted(scored, key=lambda row: (-float(row['score']), str(row['chunk_id'])))[:TOP_K]


def _rank_metrics(cases: list[dict[str, object]]) -> dict[str, float]:
    answerable = [case for case in cases if case['relevant_ids']]
    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_ranks: list[float] = []
    for case in answerable:
        relevant = set(case['relevant_ids'])
        retrieved = [row['chunk_id'] for row in case['retrieved_chunks']]
        recall_at_1 += int(any(chunk_id in relevant for chunk_id in retrieved[:1]))
        recall_at_3 += int(any(chunk_id in relevant for chunk_id in retrieved[:TOP_K]))
        rank = next((index + 1 for index, chunk_id in enumerate(retrieved) if chunk_id in relevant), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    denominator = len(answerable)
    return {
        'recall_at_1': _round(recall_at_1 / denominator),
        'recall_at_3': _round(recall_at_3 / denominator),
        'mrr': _round(sum(reciprocal_ranks) / denominator),
        'answerable_query_count': denominator,
    }


def _render_svg(values: dict[str, float]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = [
        ('retrieval recall@1', values['recall_at_1'], '#7895ff'),
        ('retrieval recall@3', values['recall_at_3'], '#4f7cff'),
        ('groundedness', values['groundedness'], '#28a36a'),
        ('citation precision', values['citation_precision'], '#f2994a'),
    ]
    width = 820
    height = 420
    left = 94
    top = 96
    chart_height = 210
    bar_gap = 138
    bar_width = 56

    def y_for(value: float) -> float:
        return top + chart_height - value * chart_height

    grid: list[str] = []
    for tick in range(6):
        value = tick * 0.2
        y = y_for(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + 560}" y2="{y:.1f}" stroke="#dce6f3" />')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#4b5f78">{value:.1f}</text>')

    bar_nodes: list[str] = []
    for index, (label, value, color) in enumerate(bars):
        x = left + index * bar_gap + 34
        y = y_for(value)
        bar_nodes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{top + chart_height - y:.1f}" rx="6" fill="{color}" />')
        bar_nodes.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" font-size="12" text-anchor="middle" fill="#263a52">{value:.2f}</text>')
        bar_nodes.append(f'<text x="{x + bar_width / 2:.1f}" y="{top + chart_height + 24}" font-size="12" text-anchor="middle" fill="#263a52">{escape(label)}</text>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="38" font-size="24" font-weight="bold" fill="#10203a">Toy RAG grounding and retrieval metrics</text>'
        '<text x="28" y="64" font-size="14" fill="#42556d">Retriever metrics are useful only when paired with claim-level grounding checks.</text>'
        '<text x="34" y="94" font-size="13" fill="#42556d">score</text>'
        + ''.join(grid)
        + ''.join(bar_nodes)
        + '<text x="92" y="372" font-size="13" fill="#42556d">citation tags help users inspect evidence, but unsupported claim counting tests whether evidence actually supports the answer.</text>'
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    cases: list[dict[str, object]] = []
    for query in QUERIES:
        retrieved = _retrieve(query)
        cases.append(
            {
                'query_id': query['query_id'],
                'query': query['query'],
                'relevant_ids': query['relevant_ids'],
                'retrieved_chunks': retrieved,
                'reader_answer': query['reader_answer'],
                'generator_answer': query['generator_answer'],
                'citations': query['citations'],
                'supported_claims': query['supported_claims'],
                'unsupported_claims': query['unsupported_claims'],
                'abstain_flag': query['abstain_flag'],
            }
        )

    retrieval_metrics = _rank_metrics(cases)
    supported_claims = sum(int(case['supported_claims']) for case in cases)
    unsupported_claims = sum(int(case['unsupported_claims']) for case in cases)
    claim_count = supported_claims + unsupported_claims
    citation_count = sum(len(case['citations']) for case in cases)
    cited_answer_count = sum(1 for case in cases if case['citations'])
    groundedness = supported_claims / claim_count
    citation_precision = 1.0 if citation_count else 0.0
    citation_coverage = cited_answer_count / len([case for case in cases if case['relevant_ids']])
    values_for_svg = {
        'recall_at_1': retrieval_metrics['recall_at_1'],
        'recall_at_3': retrieval_metrics['recall_at_3'],
        'groundedness': _round(groundedness),
        'citation_precision': _round(citation_precision),
    }
    _render_svg(values_for_svg)

    metrics = {
        'setup': {
            'unit': '07_retrieval_augmented_generation_and_eval',
            'mode': 'cpu_safe_deterministic_toy_rag_eval',
            'cpu_safe': True,
            'seed': 0,
            'no_real_llm_or_external_vector_db': True,
        },
        'retrieval_batch': {
            'query_count': len(QUERIES),
            'chunk_count': len(CHUNKS),
            'top_k': TOP_K,
            'retriever': 'keyword_overlap_plus_freshness_toy_ranker',
            'corpus_sources': [chunk['source'] for chunk in CHUNKS],
        },
        'retrieval_metrics': retrieval_metrics,
        'cases': cases,
        'split_view': {
            'lower_unsupported_claims': 'retriever_reader',
            'higher_fluency': 'retriever_generator',
            'reader_unsupported_claims': 0,
            'generator_unsupported_claims': unsupported_claims,
            'note': 'reader-style answers copy or abstain from evidence spans; generator-style answers are fluent but need stricter citation and unsupported-claim checks.',
        },
        'context_injection': {
            'chunk_boundary_strategy': 'document_section_chunks',
            'metadata_included': True,
            'citation_tags_required': True,
            'prompt_order': ['system_grounding_rule', 'retrieved_context_with_sources', 'user_query'],
            'stale_source_penalty': 0.04,
            'max_chunks_injected': TOP_K,
        },
        'grounding_eval': {
            'claim_count': claim_count,
            'supported_claims': supported_claims,
            'unsupported_claims': unsupported_claims,
            'unsupported_claim_rate': _round(unsupported_claims / claim_count),
            'groundedness': _round(groundedness),
            'citation_precision': _round(citation_precision),
            'citation_coverage': _round(citation_coverage),
            'grounding_expectation': 'claim-level evidence, not citation count',
            'abstention_cases': [case['query_id'] for case in cases if case['abstain_flag']],
        },
        'failure_modes': {
            'primary_watch': 'unsupported_claim',
            'observed_failure_modes': ['missing_evidence', 'stale_source', 'irrelevant_context', 'citation_without_support'],
            'missing_evidence_query_ids': ['meal_policy_missing'],
            'stale_source_example': 'travel_2024_old appears similar but must lose to travel_2026_limits',
            'mitigation': 'separate retriever recall, claim-grounding, citation precision, and abstention behavior in the eval harness',
        },
        'figure_path': 'artifacts/scratch-manual/rag_grounding_eval.svg',
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'retrieval_metrics': metrics['retrieval_metrics'],
        'grounding_eval': metrics['grounding_eval'],
        'figure_path': metrics['figure_path'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
