# Scripts

이 폴더는 학습/평가/검증 스크립트를 모으는 공간이다.

```text
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework
python scripts/check_curriculum_links.py
python scripts/train.py --track 03_nlp --stage 01_text_classification --config path/to/config.yaml
python scripts/eval.py --run-dir runs/03_nlp/01_text_classification/<run_id>
python scripts/eval.py --run-dir runs/05_multimodal/01_image_text_retrieval/<run_id>
```
