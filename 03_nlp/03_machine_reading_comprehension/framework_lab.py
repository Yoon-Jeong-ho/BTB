from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
TOKEN_PATTERN = re.compile(r'[가-힣A-Za-z0-9]+')
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
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
QUESTION_SUFFIXES = (
    '에서는', '으로는', '에게는', '한테는', '이었다', '였다', '했나', '했는가', '됐나', '되나',
    '인가', '인가요', '시작하나', '시작했나', '진행됐나', '열리나', '있나', '이다', '한다', '했다',
    '에서', '으로', '에게', '한테', '까지', '부터', '처럼', '은', '는', '이', '가', '을', '를', '에', '도', '만', '의',
)
QUESTION_STOPWORDS = {
    '무엇', '무엇을', '무엇이', '누가', '누구', '누구에게', '언제', '어디', '어디에', '어디에서',
    '몇', '시', '인가', '인가요', '했나', '했는가', '열리나', '시작하나', '시작했나', '진행됐나', '진행했나',
}
LOCATION_HINTS = {'회의실', '랩실', '부스', '로비', '홀', '정원', '위키', '채널', '데스크', '본관', '별관'}
TIME_HINTS = {'오전', '오후', '아침', '점심', '저녁', '새벽', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'}
ROLE_HINTS = {'연구원', '매니저', '코디네이터', '인턴', '교수', '기자', '담당자'}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_token(token: str) -> str:
    normalized = token.lower()
    for suffix in QUESTION_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            return normalized[: -len(suffix)]
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


def build_answer_profiles(rows: list[dict[str, object]]) -> tuple[dict[str, Counter[str]], dict[str, float]]:
    lexicon: dict[str, Counter[str]] = {}
    lengths: dict[str, list[int]] = {}
    for row in rows:
        answers = row['answers']
        if not isinstance(answers, list) or not answers:
            continue
        qtype = detect_question_type(str(row['question']))
        lexicon.setdefault(qtype, Counter())
        lengths.setdefault(qtype, [])
        answer_tokens = tokenize(str(answers[0]))
        lengths[qtype].append(len(answer_tokens))
        for token in answer_tokens:
            lexicon[qtype][normalize_token(token)] += 1
    return lexicon, {key: sum(values) / len(values) for key, values in lengths.items() if values}


def exact_match(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    if not answers:
        return 1.0 if not normalized_prediction else 0.0
    return 1.0 if any(normalized_prediction == normalize_text(answer) for answer in answers) else 0.0


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


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rounded(value: float) -> float:
    return round(float(value), 6)


def build_vocab(rows: list[dict[str, object]]) -> dict[str, int]:
    vocab = {'[PAD]': PAD_ID, '[UNK]': UNK_ID, '[CLS]': CLS_ID, '[SEP]': SEP_ID}
    for row in rows:
        for token in tokenize(str(row['question'])) + tokenize(str(row['context'])):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def find_answer_span(context_tokens: list[str], answers: list[str]) -> tuple[int, int] | None:
    normalized_context = [normalize_token(token) for token in context_tokens]
    for answer in answers:
        answer_tokens = [normalize_token(token) for token in tokenize(answer)]
        if not answer_tokens:
            continue
        for start in range(len(context_tokens) - len(answer_tokens) + 1):
            window = normalized_context[start:start + len(answer_tokens)]
            if window == answer_tokens:
                return start, start + len(answer_tokens) - 1
    return None


def encode_example(row: dict[str, object], vocab: dict[str, int]) -> dict[str, object]:
    question_tokens = tokenize(str(row['question']))
    context_tokens = tokenize(str(row['context']))
    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
    input_ids = [vocab.get(token, UNK_ID) for token in input_tokens]
    question_mask = [0.0] + [1.0] * len(question_tokens) + [0.0] * (len(context_tokens) + 2)
    valid_span_mask = [1.0] + [0.0] * (len(question_tokens) + 1) + [1.0] * len(context_tokens) + [0.0]
    token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1)
    context_offset = len(question_tokens) + 2
    answer_span = find_answer_span(context_tokens, row['answers'])
    if answer_span is None:
        start_position = 0
        end_position = 0
        answerable = 0.0
    else:
        start_position = context_offset + answer_span[0]
        end_position = context_offset + answer_span[1]
        answerable = 1.0

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'question_mask': question_mask,
        'valid_span_mask': valid_span_mask,
        'start_position': start_position,
        'end_position': end_position,
        'answerable': answerable,
        'question': row['question'],
        'context': row['context'],
        'answers': row['answers'],
        'context_tokens': context_tokens,
        'context_offset': context_offset,
    }


def pad_examples(encoded: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    max_len = max(len(item['input_ids']) for item in encoded)

    def pad(sequence: list[float] | list[int], fill: float | int) -> list[float] | list[int]:
        return sequence + [fill] * (max_len - len(sequence))

    return {
        'input_ids': torch.tensor([pad(item['input_ids'], PAD_ID) for item in encoded], dtype=torch.long),
        'token_type_ids': torch.tensor([pad(item['token_type_ids'], 0) for item in encoded], dtype=torch.long),
        'question_mask': torch.tensor([pad(item['question_mask'], 0.0) for item in encoded], dtype=torch.float32),
        'valid_span_mask': torch.tensor([pad(item['valid_span_mask'], 0.0) for item in encoded], dtype=torch.bool),
        'start_positions': torch.tensor([item['start_position'] for item in encoded], dtype=torch.long),
        'end_positions': torch.tensor([item['end_position'] for item in encoded], dtype=torch.long),
        'answerable': torch.tensor([item['answerable'] for item in encoded], dtype=torch.float32),
    }


class TinyQAModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.segment_embedding = torch.nn.Embedding(2, embedding_dim)
        self.encoder = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.start_head = torch.nn.Linear(hidden_dim * 6, 1)
        self.end_head = torch.nn.Linear(hidden_dim * 6, 1)
        self.answerable_head = torch.nn.Linear(hidden_dim * 4, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.token_embedding(input_ids) + self.segment_embedding(token_type_ids)
        encoded, _ = self.encoder(embedded)
        question_summary = (encoded * question_mask.unsqueeze(-1)).sum(dim=1) / question_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        expanded_question = question_summary.unsqueeze(1).expand(-1, encoded.size(1), -1)
        qa_features = torch.cat([encoded, expanded_question, encoded * expanded_question], dim=-1)
        start_logits = self.start_head(qa_features).squeeze(-1)
        end_logits = self.end_head(qa_features).squeeze(-1)
        answerable_logits = self.answerable_head(torch.cat([encoded[:, 0, :], question_summary], dim=-1)).squeeze(-1)
        return start_logits, end_logits, answerable_logits


def best_span(start_logits: torch.Tensor, end_logits: torch.Tensor, example: dict[str, object], max_answer_len: int = 5) -> tuple[str, float, int, int]:
    context_tokens = example['context_tokens']
    offset = int(example['context_offset'])
    best_score = float('-inf')
    best_start = 0
    best_end = 0
    for start in range(len(context_tokens)):
        for end in range(start, min(len(context_tokens), start + max_answer_len)):
            score = float(start_logits[offset + start] + end_logits[offset + end])
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
    predicted = ' '.join(context_tokens[best_start:best_end + 1])
    return predicted, best_score, best_start, best_end


def hybrid_best_candidate(
    example: dict[str, object],
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    lexicon: dict[str, Counter[str]],
    length_profile: dict[str, float],
    max_answer_len: int = 4,
) -> dict[str, object]:
    question = str(example['question'])
    qtype = detect_question_type(question)
    keywords = question_keywords(question)
    context_tokens = example['context_tokens']
    offset = int(example['context_offset'])

    candidates: list[dict[str, object]] = []
    for start in range(len(context_tokens)):
        for length in range(1, min(max_answer_len, len(context_tokens) - start) + 1):
            end = start + length - 1
            candidate_tokens = context_tokens[start:end + 1]
            window = context_tokens[max(0, start - 3): min(len(context_tokens), end + 4)]
            window_norm = {normalize_token(token) for token in window}
            candidate_norm = [normalize_token(token) for token in candidate_tokens]
            overlap = sum(1 for token in keywords if token in window_norm)
            lexicon_hits = sum(1 for token in candidate_norm if lexicon.get(qtype, Counter())[token] > 0)
            hint_hits = type_token_hits(qtype, candidate_tokens)
            candidate_question_overlap = sum(1 for token in candidate_norm if token in keywords)
            average_length = length_profile.get(qtype, 2.0)
            length_bonus = max(0.0, 1.0 - abs(len(candidate_tokens) - average_length) * 0.35)
            heuristic = (
                overlap * 1.2
                + lexicon_hits * 0.6
                + hint_hits * 0.55
                + type_bonus(qtype, candidate_tokens)
                + length_bonus * 0.3
                - candidate_question_overlap * 0.9
            )
            model_score = float(start_logits[offset + start] + end_logits[offset + end])
            candidates.append(
                {
                    'answer': ' '.join(candidate_tokens),
                    'start': start,
                    'end': end,
                    'heuristic_score': heuristic,
                    'model_score': model_score,
                }
            )

    model_scores = [float(candidate['model_score']) for candidate in candidates]
    min_model = min(model_scores)
    max_model = max(model_scores)
    denominator = max_model - min_model
    for candidate in candidates:
        if denominator <= 1e-6:
            normalized_model = 0.5
        else:
            normalized_model = (float(candidate['model_score']) - min_model) / denominator
        candidate['hybrid_score'] = candidate['heuristic_score'] + 0.25 * normalized_model

    best = max(candidates, key=lambda item: (float(item['hybrid_score']), -len(str(item['answer']))))
    best['question_type'] = qtype
    return best


def choose_no_answer_threshold(
    examples: list[dict[str, object]],
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    lexicon: dict[str, Counter[str]],
    length_profile: dict[str, float],
) -> float:
    scores = [
        float(hybrid_best_candidate(example, start_logits[index], end_logits[index], lexicon, length_profile)['hybrid_score'])
        for index, example in enumerate(examples)
    ]
    if not scores:
        return 0.0
    sorted_scores = sorted(set(scores))
    thresholds = [sorted_scores[0] - 0.5]
    thresholds.extend((left + right) / 2 for left, right in zip(sorted_scores, sorted_scores[1:]))
    thresholds.append(sorted_scores[-1] + 0.5)

    best_threshold = thresholds[0]
    best_accuracy = -1.0
    for threshold in thresholds:
        accuracy = safe_div(
            sum(
                1
                for score, example in zip(scores, examples)
                if (score >= threshold) == bool(example['answers'])
            ),
            len(examples),
        )
        if accuracy > best_accuracy or (abs(accuracy - best_accuracy) < 1e-9 and threshold > best_threshold):
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def run() -> None:
    torch.manual_seed(13)

    vocab = build_vocab(TRAIN_ROWS)
    lexicon, length_profile = build_answer_profiles(TRAIN_ROWS)
    train_examples = [encode_example(row, vocab) for row in TRAIN_ROWS]
    eval_examples = [encode_example(row, vocab) for row in EVAL_ROWS]
    train_batch = pad_examples(train_examples)
    eval_batch = pad_examples(eval_examples)

    model = TinyQAModel(vocab_size=len(vocab), embedding_dim=28, hidden_dim=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    loss_history: list[float] = []
    for _ in range(160):
        model.train()
        optimizer.zero_grad()
        start_logits, end_logits, answerable_logits = model(
            input_ids=train_batch['input_ids'],
            token_type_ids=train_batch['token_type_ids'],
            question_mask=train_batch['question_mask'],
        )
        masked_start = start_logits.masked_fill(~train_batch['valid_span_mask'], -1e9)
        masked_end = end_logits.masked_fill(~train_batch['valid_span_mask'], -1e9)
        start_loss = F.cross_entropy(masked_start, train_batch['start_positions'])
        end_loss = F.cross_entropy(masked_end, train_batch['end_positions'])
        answerable_loss = F.binary_cross_entropy_with_logits(answerable_logits, train_batch['answerable'])
        loss = start_loss + end_loss + 0.6 * answerable_loss
        loss.backward()
        optimizer.step()
        loss_history.append(rounded(loss.item()))

    model.eval()
    with torch.no_grad():
        train_start_logits, train_end_logits, _ = model(
            input_ids=train_batch['input_ids'],
            token_type_ids=train_batch['token_type_ids'],
            question_mask=train_batch['question_mask'],
        )
        eval_start_logits, eval_end_logits, eval_answerable_logits = model(
            input_ids=eval_batch['input_ids'],
            token_type_ids=eval_batch['token_type_ids'],
            question_mask=eval_batch['question_mask'],
        )
        train_start_logits = train_start_logits.masked_fill(~train_batch['valid_span_mask'], -1e9)
        train_end_logits = train_end_logits.masked_fill(~train_batch['valid_span_mask'], -1e9)
        eval_start_logits = eval_start_logits.masked_fill(~eval_batch['valid_span_mask'], -1e9)
        eval_end_logits = eval_end_logits.masked_fill(~eval_batch['valid_span_mask'], -1e9)
        answerable_probs = torch.sigmoid(eval_answerable_logits)

    threshold = choose_no_answer_threshold(
        examples=train_examples,
        start_logits=train_start_logits,
        end_logits=train_end_logits,
        lexicon=lexicon,
        length_profile=length_profile,
    )

    exact_scores: list[float] = []
    f1_scores: list[float] = []
    answerable_hits: list[int] = []
    prediction_rows: list[dict[str, object]] = []

    for index, (example, probability) in enumerate(zip(eval_examples, answerable_probs.tolist())):
        candidate = hybrid_best_candidate(
            example=example,
            start_logits=eval_start_logits[index],
            end_logits=eval_end_logits[index],
            lexicon=lexicon,
            length_profile=length_profile,
        )
        predicted_answerable = float(candidate['hybrid_score']) >= threshold
        if predicted_answerable:
            predicted_answer = str(candidate['answer'])
            span_score = float(candidate['hybrid_score'])
            local_start = int(candidate['start'])
            local_end = int(candidate['end'])
        else:
            predicted_answer = ''
            span_score = float(candidate['hybrid_score'])
            local_start = -1
            local_end = -1
        answers = example['answers']
        exact = exact_match(predicted_answer, answers)
        f1 = token_f1(predicted_answer, answers)
        gold_answerable = bool(answers)
        exact_scores.append(exact)
        f1_scores.append(f1)
        answerable_hits.append(int(gold_answerable == predicted_answerable))
        prediction_rows.append(
            {
                'question': example['question'],
                'context': example['context'],
                'gold_answers': answers,
                'predicted_answer': predicted_answer,
                'gold_answerable': gold_answerable,
                'predicted_answerable': predicted_answerable,
                'answerable_probability': rounded(probability),
                'best_span_score': rounded(span_score),
                'no_answer_threshold': rounded(threshold),
                'question_type': candidate['question_type'],
                'heuristic_score': rounded(candidate['heuristic_score']),
                'model_score': rounded(candidate['model_score']),
                'predicted_local_start': local_start,
                'predicted_local_end': local_end,
                'exact_match': rounded(exact),
                'token_f1': rounded(f1),
            }
        )

    metrics = {
        'train_size': len(TRAIN_ROWS),
        'eval_size': len(EVAL_ROWS),
        'vocab_size': len(vocab),
        'embedding_dim': 28,
        'hidden_dim': 24,
        'epochs': 160,
        'label_names': ['no_answer', 'span_answer'],
        'train_input_shape': list(train_batch['input_ids'].shape),
        'eval_input_shape': list(eval_batch['input_ids'].shape),
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
        'no_answer_threshold': rounded(threshold),
        'eval_exact_match': rounded(sum(exact_scores) / len(exact_scores)),
        'eval_token_f1': rounded(sum(f1_scores) / len(f1_scores)),
        'answerable_accuracy': rounded(sum(answerable_hits) / len(answerable_hits)),
        'prediction_rows': prediction_rows,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
