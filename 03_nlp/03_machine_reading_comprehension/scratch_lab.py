from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'answerability_breakdown.svg'
TOKEN_PATTERN = re.compile(r'[가-힣A-Za-z0-9]+')
QUESTION_SUFFIXES = (
    '에서는',
    '으로는',
    '에게는',
    '한테는',
    '이었다',
    '였다',
    '했나',
    '했는가',
    '됐나',
    '되나',
    '인가',
    '인가요',
    '시작하나',
    '진행하나',
    '열리나',
    '있나',
    '인가요',
    '이다',
    '한다',
    '했다',
    '되다',
    '이다',
    '에서',
    '으로',
    '에게',
    '한테',
    '까지',
    '부터',
    '처럼',
    '은',
    '는',
    '이',
    '가',
    '을',
    '를',
    '에',
    '도',
    '만',
    '의',
)
QUESTION_STOPWORDS = {
    '무엇',
    '무엇을',
    '무엇이',
    '누가',
    '누구',
    '누구에게',
    '언제',
    '어디',
    '어디에',
    '어디에서',
    '몇',
    '시',
    '인가',
    '인가요',
    '했나',
    '했는가',
    '열리나',
    '시작하나',
    '시작했나',
    '진행됐나',
    '진행했나',
}
LOCATION_HINTS = {'회의실', '랩실', '부스', '로비', '홀', '정원', '위키', '채널', '데스크', '본관', '별관'}
TIME_HINTS = {'오전', '오후', '아침', '점심', '저녁', '새벽', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'}
ROLE_HINTS = {'연구원', '매니저', '코디네이터', '인턴', '교수', '기자', '담당자'}
TRAIN_ROWS = [
    {
        'context': '도서관 안내 데스크는 2층 로비 옆에 있다. 오늘 세미나는 3강의실에서 열린다.',
        'question': '오늘 세미나는 어디에서 열리나?',
        'answers': ['3강의실'],
    },
    {
        'context': '발표는 오전 열 시에 시작되고 질의응답은 열한 시 반에 끝난다.',
        'question': '발표는 언제 시작하나?',
        'answers': ['오전 열 시'],
    },
    {
        'context': '프로젝트 책임자는 박지훈 연구원이며 기록 담당은 이서연 매니저다.',
        'question': '프로젝트 책임자는 누구인가?',
        'answers': ['박지훈 연구원'],
    },
    {
        'context': '신규 서버 점검은 금요일 오후에 진행되고 데이터 백업은 토요일 새벽에 완료된다.',
        'question': '데이터 백업은 언제 완료되나?',
        'answers': ['토요일 새벽'],
    },
    {
        'context': '행사 접수처는 중앙 홀 왼쪽 부스에 있고 식사는 야외 정원에서 제공된다.',
        'question': '행사 접수처는 어디에 있나?',
        'answers': ['중앙 홀 왼쪽 부스'],
    },
    {
        'context': '가이드 문서는 보안팀 위키에 정리돼 있으며 최신 버전은 민지에게 문의하면 된다.',
        'question': '최신 버전은 누구에게 문의하면 되나?',
        'answers': ['민지'],
    },
    {
        'context': '회의 자료는 오전에 공유됐고 참석자들은 메신저로 알림을 받았다.',
        'question': '회의는 몇 시에 끝났나?',
        'answers': [],
    },
    {
        'context': '앱 설치 파일은 다운로드 페이지에 있고 사용 설명서는 도움말 메뉴에서 확인할 수 있다.',
        'question': '앱 개발자는 누구인가?',
        'answers': [],
    },
]
EVAL_ROWS = [
    {
        'context': '아침 브리핑은 본관 4층 회의실에서 진행됐고 후속 워크숍은 별관 랩실에서 이어졌다.',
        'question': '아침 브리핑은 어디에서 진행됐나?',
        'answers': ['본관 4층 회의실'],
    },
    {
        'context': '최종 점검은 오후 세 시에 시작했고 배포는 오후 다섯 시에 마무리됐다.',
        'question': '최종 점검은 언제 시작했나?',
        'answers': ['오후 세 시'],
    },
    {
        'context': '현장 기록은 서유진 코디네이터가 맡았고 발표 자료 정리는 한지민 인턴이 도왔다.',
        'question': '현장 기록은 누가 맡았나?',
        'answers': ['서유진 코디네이터'],
    },
    {
        'context': '공지문은 메일과 게시판에 동시에 올라갔고 문의는 운영팀 채널로 받는다.',
        'question': '공지문은 누가 승인했나?',
        'answers': [],
    },
]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_token(token: str) -> str:
    normalized = token.lower()
    for suffix in QUESTION_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def normalize_text(text: str) -> str:
    return ' '.join(normalize_token(token) for token in tokenize(text)).strip()


def detect_question_type(question: str) -> str:
    if '어디' in question:
        return 'where'
    if '언제' in question or '몇 시' in question:
        return 'when'
    if '누가' in question or '누구' in question:
        return 'who'
    return 'what'


def build_answer_profiles(rows: list[dict[str, object]]) -> tuple[dict[str, Counter[str]], dict[str, float]]:
    lexicon: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        answers = row['answers']
        if not isinstance(answers, list) or not answers:
            continue
        qtype = detect_question_type(str(row['question']))
        answer_tokens = tokenize(str(answers[0]))
        lengths[qtype].append(len(answer_tokens))
        for token in answer_tokens:
            lexicon[qtype][normalize_token(token)] += 1
    average_lengths = {
        qtype: sum(values) / len(values)
        for qtype, values in lengths.items()
        if values
    }
    return lexicon, average_lengths


def question_keywords(question: str) -> list[str]:
    keywords: list[str] = []
    for token in tokenize(question):
        normalized = normalize_token(token)
        if not normalized or normalized in QUESTION_STOPWORDS:
            continue
        keywords.append(normalized)
    return keywords


def is_person_like(token: str) -> bool:
    return bool(re.fullmatch(r'[가-힣]{2,4}', token))


def type_bonus(qtype: str, candidate_tokens: list[str]) -> float:
    normalized = [normalize_token(token) for token in candidate_tokens]
    if qtype == 'where':
        if any(token in LOCATION_HINTS or '층' in token or token.endswith(('실', '관', '장', '부스')) for token in normalized):
            return 1.4
        return 0.0
    if qtype == 'when':
        if any(token in TIME_HINTS or token.endswith('시') or token.isdigit() for token in normalized):
            return 1.4
        return 0.0
    if qtype == 'who':
        if any(token in ROLE_HINTS for token in normalized):
            return 1.2
        if any(is_person_like(token) for token in normalized):
            return 0.8
        return 0.0
    return 0.2


def type_token_hits(qtype: str, candidate_tokens: list[str]) -> int:
    normalized = [normalize_token(token) for token in candidate_tokens]
    if qtype == 'where':
        return sum(
            1
            for token in normalized
            if token in LOCATION_HINTS or '층' in token or token.endswith(('실', '관', '장', '부스'))
        )
    if qtype == 'when':
        return sum(1 for token in normalized if token in TIME_HINTS or token.endswith('시') or token.isdigit())
    if qtype == 'who':
        return sum(1 for token in normalized if token in ROLE_HINTS or is_person_like(token))
    return 0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rounded(value: float) -> float:
    return round(float(value), 6)


def token_f1(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 1.0 if not normalize_text(prediction) else 0.0
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_text(answer).split()
        overlap = Counter(pred_tokens) & Counter(gold_tokens)
        shared = sum(overlap.values())
        if shared == 0:
            continue
        precision = shared / len(pred_tokens)
        recall = shared / len(gold_tokens)
        best = max(best, (2 * precision * recall) / (precision + recall))
    return best


def exact_match(prediction: str, answers: list[str]) -> float:
    pred_normalized = normalize_text(prediction)
    if not answers:
        return 1.0 if not pred_normalized else 0.0
    return 1.0 if any(pred_normalized == normalize_text(answer) for answer in answers) else 0.0


def score_candidate(
    context_tokens: list[str],
    question: str,
    candidate_tokens: list[str],
    start_index: int,
    end_index: int,
    lexicon: dict[str, Counter[str]],
    length_profile: dict[str, float],
) -> tuple[float, dict[str, float]]:
    qtype = detect_question_type(question)
    keywords = question_keywords(question)
    window = context_tokens[max(0, start_index - 3): min(len(context_tokens), end_index + 4)]
    window_norm = {normalize_token(token) for token in window}
    candidate_norm = [normalize_token(token) for token in candidate_tokens]

    overlap = sum(1 for token in keywords if token in window_norm)
    lexicon_hits = sum(1 for token in candidate_norm if lexicon[qtype][token] > 0)
    candidate_question_overlap = sum(1 for token in candidate_norm if token in keywords)
    hint_hits = type_token_hits(qtype, candidate_tokens)
    average_length = length_profile.get(qtype, 2.0)
    length_bonus = max(0.0, 1.0 - abs(len(candidate_tokens) - average_length) * 0.35)
    bonus = type_bonus(qtype, candidate_tokens)
    score = (
        overlap * 1.2
        + lexicon_hits * 0.6
        + hint_hits * 0.55
        + bonus
        + length_bonus * 0.3
        - candidate_question_overlap * 0.9
    )
    if not candidate_tokens:
        score = -math.inf
    return score, {
        'overlap': rounded(overlap),
        'lexicon_hits': rounded(lexicon_hits),
        'hint_hits': rounded(hint_hits),
        'type_bonus': rounded(bonus),
        'length_bonus': rounded(length_bonus),
        'candidate_question_overlap': rounded(candidate_question_overlap),
    }


def best_candidate(
    row: dict[str, object],
    lexicon: dict[str, Counter[str]],
    length_profile: dict[str, float],
) -> dict[str, object]:
    context = str(row['context'])
    question = str(row['question'])
    context_tokens = tokenize(context)
    best: dict[str, object] = {
        'answer': '',
        'score': -math.inf,
        'start': -1,
        'end': -1,
        'components': {},
    }
    candidate_traces: list[dict[str, object]] = []

    for start in range(len(context_tokens)):
        for length in range(1, min(4, len(context_tokens) - start) + 1):
            end = start + length - 1
            candidate_tokens = context_tokens[start:end + 1]
            score, components = score_candidate(
                context_tokens=context_tokens,
                question=question,
                candidate_tokens=candidate_tokens,
                start_index=start,
                end_index=end,
                lexicon=lexicon,
                length_profile=length_profile,
            )
            trace = {
                'candidate': ' '.join(candidate_tokens),
                'score': rounded(score),
                'start': start,
                'end': end,
                'components': components,
            }
            candidate_traces.append(trace)
            best_score = float(best['score'])
            best_length = (int(best['end']) - int(best['start']) + 1) if int(best['start']) >= 0 else 99
            current_length = end - start + 1
            if score > best_score or (math.isclose(score, best_score) and current_length < best_length):
                best = {
                    'answer': ' '.join(candidate_tokens),
                    'score': score,
                    'start': start,
                    'end': end,
                    'components': components,
                }

    candidate_traces.sort(key=lambda item: (float(item['score']), -int(item['start'])), reverse=True)
    best['top_candidates'] = candidate_traces[:3]
    best['question_type'] = detect_question_type(question)
    return best


def choose_threshold(rows: list[dict[str, object]], lexicon: dict[str, Counter[str]], length_profile: dict[str, float]) -> float:
    scores = [float(best_candidate(row, lexicon, length_profile)['score']) for row in rows]
    if not scores:
        return 0.0
    sorted_scores = sorted(set(scores))
    thresholds = [sorted_scores[0] - 0.5]
    thresholds.extend((left + right) / 2 for left, right in zip(sorted_scores, sorted_scores[1:]))
    thresholds.append(sorted_scores[-1] + 0.5)

    best_threshold = thresholds[0]
    best_accuracy = -1.0
    for threshold in thresholds:
        predictions = [score >= threshold for score in scores]
        gold = [bool(row['answers']) for row in rows]
        accuracy = safe_div(sum(1 for pred, answerable in zip(predictions, gold) if pred == answerable), len(rows))
        if accuracy > best_accuracy or (math.isclose(accuracy, best_accuracy) and threshold > best_threshold):
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def save_svg(answerable_em: float, unanswerable_em: float) -> None:
    width, height = 640, 360
    chart_height = 220
    chart_bottom = 290
    left = 120
    bar_width = 120
    gap = 120
    labels = [
        ('answerable EM', answerable_em, '#1c7ed6'),
        ('unanswerable EM', unanswerable_em, '#d94841'),
    ]

    bars: list[str] = []
    for index, (label, value, color) in enumerate(labels):
        x = left + index * (bar_width + gap)
        bar_height = chart_height * value
        y = chart_bottom - bar_height
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{color}" opacity="0.88" />')
        bars.append(f'<text x="{x + bar_width / 2:.2f}" y="{chart_bottom + 26}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{label}</text>')
        bars.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{value:.2f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="32" y="34" font-size="22" font-family="Arial, sans-serif">Answerability breakdown (scratch MRC)</text>
  <line x1="90" y1="{chart_bottom}" x2="560" y2="{chart_bottom}" stroke="#495057" stroke-width="2" />
  <line x1="90" y1="60" x2="90" y2="{chart_bottom}" stroke="#495057" stroke-width="2" />
  <text x="54" y="76" font-size="12" font-family="Arial, sans-serif" fill="#495057">EM</text>
  {''.join(bars)}
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    lexicon, length_profile = build_answer_profiles(TRAIN_ROWS)
    threshold = choose_threshold(TRAIN_ROWS, lexicon, length_profile)

    prediction_rows: list[dict[str, object]] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    answerable_flags: list[bool] = []
    predicted_flags: list[bool] = []

    for row in EVAL_ROWS:
        best = best_candidate(row, lexicon, length_profile)
        predicted_answerable = float(best['score']) >= threshold
        predicted_answer = str(best['answer']) if predicted_answerable else ''
        answers = row['answers']
        exact = exact_match(predicted_answer, answers if isinstance(answers, list) else [])
        f1 = token_f1(predicted_answer, answers if isinstance(answers, list) else [])
        gold_answerable = bool(answers)
        em_scores.append(exact)
        f1_scores.append(f1)
        answerable_flags.append(gold_answerable)
        predicted_flags.append(predicted_answerable)
        prediction_rows.append(
            {
                'question': row['question'],
                'context': row['context'],
                'question_type': best['question_type'],
                'gold_answers': answers,
                'predicted_answer': predicted_answer,
                'best_span_score': rounded(best['score']),
                'no_answer_threshold': rounded(threshold),
                'gold_answerable': gold_answerable,
                'predicted_answerable': predicted_answerable,
                'exact_match': rounded(exact),
                'token_f1': rounded(f1),
                'score_components': best['components'],
                'top_candidates': best['top_candidates'],
            }
        )

    answerable_exact = safe_div(
        sum(score for score, flag in zip(em_scores, answerable_flags) if flag),
        sum(1 for flag in answerable_flags if flag),
    )
    unanswerable_exact = safe_div(
        sum(score for score, flag in zip(em_scores, answerable_flags) if not flag),
        sum(1 for flag in answerable_flags if not flag),
    )

    metrics = {
        'train_size': len(TRAIN_ROWS),
        'eval_size': len(EVAL_ROWS),
        'answerable_eval_size': sum(1 for row in EVAL_ROWS if row['answers']),
        'unanswerable_eval_size': sum(1 for row in EVAL_ROWS if not row['answers']),
        'question_type_lengths': {key: rounded(value) for key, value in length_profile.items()},
        'answer_lexicon_top': {
            key: [token for token, _ in counts.most_common(4)]
            for key, counts in lexicon.items()
        },
        'eval_exact_match': rounded(sum(em_scores) / len(em_scores)),
        'eval_token_f1': rounded(sum(f1_scores) / len(f1_scores)),
        'answerable_accuracy': rounded(
            safe_div(sum(1 for gold, pred in zip(answerable_flags, predicted_flags) if gold == pred), len(answerable_flags))
        ),
        'answerable_exact_match': rounded(answerable_exact),
        'unanswerable_exact_match': rounded(unanswerable_exact),
        'no_answer_threshold': rounded(threshold),
        'prediction_rows': prediction_rows,
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(answerable_exact, unanswerable_exact)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
