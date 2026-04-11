# 02 Accelerate Workflows 분석

## 해석 프레임
- Accelerate는 학습 루프를 대체하지 않고 실행 환경 적응을 단순화한다.
- `prepare()`가 감추는 wrapper와 여전히 남는 backend complexity를 분리해 읽는다.
- distributed_type과 num_processes는 편의 설정이 아니라 실행 계약이다.
