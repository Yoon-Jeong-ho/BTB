#!/usr/bin/env node
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = process.env.BTB_QA_OUT || path.join(os.tmpdir(), 'btb-playwright-site-qa');

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function waitForServer(url, timeoutMs = 8000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError?.message || 'unknown error'}`);
}

async function withStaticServer(fn) {
  const port = await freePort();
  const python = process.env.PYTHON || 'python';
  const server = spawn(python, ['scripts/study_server.py', '--port', String(port), '--bind', '127.0.0.1'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stderr = '';
  server.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    await waitForServer(`${baseUrl}/web/`);
    return await fn(baseUrl);
  } finally {
    server.kill();
    if (stderr && process.env.BTB_QA_DEBUG) process.stderr.write(stderr);
  }
}

async function withPlainHttpServer(fn) {
  const port = await freePort();
  const python = process.env.PYTHON || 'python';
  const server = spawn(python, ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stderr = '';
  server.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    await waitForServer(`${baseUrl}/web/`);
    return await fn(baseUrl);
  } finally {
    server.kill();
    if (stderr && process.env.BTB_QA_DEBUG) process.stderr.write(stderr);
  }
}

async function assertNoHorizontalOverflow(page, label) {
  const metrics = await page.evaluate(() => {
    const box = (selector) => {
      const element = document.querySelector(selector);
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      return {
        clientWidth: element.clientWidth,
        scrollWidth: element.scrollWidth,
        width: rect.width,
        top: rect.top,
        bottom: rect.bottom,
      };
    };
    return {
      innerWidth: window.innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      bodyScrollWidth: document.body.scrollWidth,
      reader: box('.reader-panel'),
      content: box('.lesson-content'),
      sidebar: box('.study-sidebar'),
      track: box('.track-panel'),
      unit: box('.unit-panel'),
      lessonReader: box('.lesson-reader'),
      lessonWorkspace: box('.lesson-workspace'),
      tabs: box('.document-tabs'),
      codeBlock: box('.code-block'),
    };
  });
  if (metrics.scrollWidth > metrics.innerWidth + 1 || metrics.bodyScrollWidth > metrics.innerWidth + 1) {
    throw new Error(`${label}: horizontal overflow document=${metrics.scrollWidth}, body=${metrics.bodyScrollWidth}, viewport=${metrics.innerWidth}`);
  }
  for (const [name, box] of Object.entries({
    content: metrics.content,
    lessonReader: metrics.lessonReader,
    lessonWorkspace: metrics.lessonWorkspace,
    tabs: metrics.tabs,
    codeBlock: metrics.codeBlock,
  })) {
    if (box && box.scrollWidth > box.clientWidth + 1) {
      throw new Error(`${label}: ${name} overflows horizontally ${box.scrollWidth} > ${box.clientWidth}`);
    }
  }
  if (metrics.innerWidth >= 1200 && metrics.reader.width < 850) {
    throw new Error(`${label}: reader is too narrow (${metrics.reader.width}px)`);
  }
  if (metrics.innerWidth >= 1200 && metrics.sidebar.width > 340) {
    throw new Error(`${label}: left sidebar is too wide (${metrics.sidebar.width}px)`);
  }
  if (metrics.innerWidth >= 900 && Math.abs(metrics.unit.top - metrics.track.bottom) > 28) {
    throw new Error(`${label}: unit list is detached from tracks (gap ${metrics.unit.top - metrics.track.bottom}px)`);
  }
  return metrics;
}

async function assertReaderBeforeGuideOnNarrow(page, label) {
  const order = await page.evaluate(() => {
    const reader = document.querySelector('.lesson-reader')?.getBoundingClientRect();
    const guide = document.querySelector('.lesson-guide')?.getBoundingClientRect();
    return {
      readerTop: reader?.top ?? 0,
      guideTop: guide?.top ?? 0,
    };
  });
  if (order.readerTop > order.guideTop) {
    throw new Error(`${label}: lesson reader should appear before guide on narrow screens (${order.readerTop} > ${order.guideTop})`);
  }
}

function isAllowedRequest(urlText, baseUrl) {
  const url = new URL(urlText);
  if (['about:', 'blob:', 'data:'].includes(url.protocol)) return true;
  return url.origin === new URL(baseUrl).origin;
}

async function newLocalOnlyContext(browser, baseUrl, options) {
  const context = await browser.newContext(options);
  const externalRequests = new Set();
  await context.route('**/*', async (route) => {
    const url = route.request().url();
    if (!isAllowedRequest(url, baseUrl)) {
      externalRequests.add(url);
      await route.abort();
      return;
    }
    await route.continue();
  });
  return { context, externalRequests };
}

function assertLocalOnlyRequests(externalRequests, label) {
  if (externalRequests.size) {
    throw new Error(`${label}: external requests are not allowed:\n${Array.from(externalRequests).join('\n')}`);
  }
}

async function assertInjectedPythonComments(page, tabName) {
  await page.getByRole('tab', { name: tabName }).click();
  const codeText = await page.locator('.code-block code').innerText();
  for (const token of ['# 코드 읽기 힌트:', '# 학습 포인트:']) {
    if (!codeText.includes(token)) {
      throw new Error(`${tabName}: missing inline Korean code hint ${token}`);
    }
  }
  for (const removed of ['# 학습자용 한글 주석', '# 이 파일은 무엇인가:', '# 어떻게 읽으면 좋은가:', '# 실행하면 남는 결과:', '# 아래부터 원본 Python 코드입니다.', '# 핵심 함수', '파일을 실행했을 때 전체 흐름이 모이는 진입점입니다.']) {
    if (codeText.includes(removed)) {
      throw new Error(`${tabName}: removed top-level guidance comment is still present: ${removed}`);
    }
  }
  if (codeText.trimStart().startsWith('#')) {
    throw new Error(`${tabName}: Korean hints should not be prepended as a repeated header block`);
  }
  if (await page.locator('.learner-comment').count()) {
    throw new Error(`${tabName}: guidance must be injected into code, not rendered as a separate learner-comment block`);
  }
}

async function assertTrackMarkdownRendered(page) {
  const foundations = page.locator('.track-card', { hasText: '00 Foundations' }).first();
  await foundations.waitFor({ state: 'visible' });
  const text = await foundations.innerText();
  if (text.includes('**')) {
    throw new Error(`track summary exposes raw markdown syntax: ${text}`);
  }
  await foundations.locator('strong', { hasText: '딥러닝 공통 기초 트랙' }).waitFor({ state: 'visible' });
}

async function assertRunButton(page, tabName, expectedText) {
  await page.getByRole('tab', { name: tabName }).click();
  await page.locator('.cell-probe', { hasText: '선택 함수 미리보기' }).waitFor({ state: 'visible' });
  await page.locator('[data-run-code]', { hasText: `${tabName} 실행` }).click();
  const output = page.locator('[data-run-output]');
  await output.waitFor({ state: 'visible' });
  await output.locator(`text=${expectedText}`).waitFor({ state: 'visible', timeout: 30000 });
  const text = await output.innerText();
  if (!text.includes('종료 코드: 0')) {
    throw new Error(`${tabName}: run output did not finish with exit code 0:\n${text}`);
  }
  if (!text.includes('실행 환경:')) {
    throw new Error(`${tabName}: run output should show the selected Python/conda and CPU/GPU runner:\n${text}`);
  }
  await page.locator('.run-primer', { hasText: '실행 전에 볼 것' }).waitFor({ state: 'visible' });
  await page.locator('.run-primer', { hasText: '예상 산출물' }).waitFor({ state: 'visible' });
  await page.locator('.run-primer', { hasText: '좋은 결과 기준' }).waitFor({ state: 'visible' });
  const insights = page.locator('[data-run-insights]');
  await insights.locator('text=실행 관찰 카드').waitFor({ state: 'visible' });
  await insights.locator('text=예상 산출물').waitFor({ state: 'visible' });
  await insights.locator('text=봐야 할 숫자').waitFor({ state: 'visible' });
  await insights.locator(`text=${expectedText}`).waitFor({ state: 'visible' });
  await insights.locator('strong', { hasText: '산출물' }).waitFor({ state: 'visible' });
  await insights.locator('text=좋은 결과 기준').waitFor({ state: 'visible' });
  await insights.locator('text=다음 질문').waitFor({ state: 'visible' });
  const artifactViewer = page.locator('.artifact-viewer');
  await artifactViewer.locator('text=실행 산출물 바로 보기').waitFor({ state: 'visible' });
  await artifactViewer.locator('text=지표 요약').first().waitFor({ state: 'visible' });
  await artifactViewer.locator('code', { hasText: 'metrics.json' }).first().waitFor({ state: 'visible' });

  await page.locator('[data-run-cell]').click();
  const cellOutput = page.locator('[data-cell-output]');
  await cellOutput.locator('text=선택 함수 미리보기').waitFor({ state: 'visible' });
  await cellOutput.locator('text=줄 범위').waitFor({ state: 'visible' });
  await cellOutput.locator('text=ARTIFACT_DIR').first().waitFor({ state: 'visible' });
}

async function assertQuizAndWrongNotes(page) {
  await page.locator('.quiz-panel', { hasText: '단원 점검 퀴즈' }).waitFor({ state: 'visible' });
  const evidenceQuestion = page.locator('.quiz-question', { hasText: 'matmul shape mismatch를 찾을 때 가장 먼저 맞춰야 하는 축은 무엇인가요?' }).first();
  await evidenceQuestion.getByLabel(/batch 차원만 같으면/).check();
  await evidenceQuestion.getByRole('button', { name: '정답 확인' }).click();
  await evidenceQuestion.locator('.quiz-feedback strong', { hasText: '다시 확인' }).waitFor({ state: 'visible' });
  await evidenceQuestion.locator('[data-wrong-note-memo]').fill('matmul 내적 축 대신 batch 축만 봄');
  await evidenceQuestion.locator('[data-wrong-note-memo]').dispatchEvent('change');
  await page.locator('.wrong-note-panel', { hasText: 'matmul 내적 축 대신 batch 축만 봄' }).waitFor({ state: 'visible' });

  const conceptQuestion = page.locator('.quiz-question', { hasText: 'shape mismatch를 입력 축 기준으로' }).first();
  await conceptQuestion.locator('textarea[data-quiz-id="concept"]').fill('shape가 계산 결과와 연결되는 방식');
  await conceptQuestion.getByRole('button', { name: '예시와 비교 저장' }).click();
  await page.locator('.quiz-feedback.review', { hasText: '자동 채점 대신 예시와 비교하세요' }).waitFor({ state: 'visible' });

  await page.locator('#review-mistakes').click();
  await page.locator('#mistake-dialog', { hasText: 'matmul 내적 축 대신 batch 축만 봄' }).waitFor({ state: 'visible' });
  await page.locator('#mistake-dialog button[value="cancel"]').click();

  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('.quiz-panel', { hasText: '단원 점검 퀴즈' }).waitFor({ state: 'visible' });
  await page.locator('#review-mistakes').click();
  await page.locator('#mistake-dialog', { hasText: 'matmul 내적 축 대신 batch 축만 봄' }).waitFor({ state: 'visible' });
  await page.locator('#mistake-dialog button[value="cancel"]').click();
  await page.getByRole('tab', { name: '기초 실습 코드' }).click();
  await page.getByText('코드 읽기 안내').waitFor({ state: 'visible' });
}

async function selectStudyUnit(page, trackText, unitText) {
  await page.locator('.track-card', { hasText: trackText }).click();
  await page.locator('.unit-card', { hasText: unitText }).click();
  await page.locator('#detail-title', { hasText: unitText }).waitFor({ state: 'visible' });
}

async function assertResourceDocument(page, label, expectedText) {
  await page.locator('.resource-button', { hasText: label }).click();
  await page.locator('.document-title', { hasText: label }).waitFor({ state: 'visible' });
  await page.locator('#lesson-content', { hasText: expectedText }).waitFor({ state: 'visible' });
  return assertNoHorizontalOverflow(page, `resource-${label}`);
}

async function assertBridgeResources(page) {
  await selectStudyUnit(page, '05 Advanced NLP + LLM', '01 Language Modeling and Pretraining Objectives');
  await assertResourceDocument(page, 'Decoder 생성 연결', 'KV-cache intuition');

  await selectStudyUnit(page, '09 Multimodal', '02 Image Captioning');
  await assertResourceDocument(page, '멀티모달 생성 연결', 'Grounding failure vs retrieval failure');

  await selectStudyUnit(page, '10 VLA', '01 VLA Vision-Language-Action Grounding');
  await assertResourceDocument(page, 'RL→VLA 연결', 'Behavior cloning vs RL vs offline RL');
}

async function assertMlRunnerResources(page) {
  await selectStudyUnit(page, '01 ML', '01 Tabular Classification');
  await assertResourceDocument(page, 'ML→DL 연결 문서', 'DataLoader');
  await page.getByRole('tab', { name: '데이터 준비 코드' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: '모델 코드' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: '실험 흐름 코드' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: '실험 실행 코드' }).waitFor({ state: 'visible' });
  if (await page.getByRole('tab', { name: '기초 실습 코드' }).count()) {
    throw new Error('01_ml should not show missing scratch_lab.py tab');
  }
  await assertMlHelperRunsStage(page, '데이터 준비 코드', '데이터 준비 코드: 실험에 넣을 표 만들기', 'dataset.py', { executeStage: true });
  await assertMlHelperRunsStage(page, '모델 코드', '모델 코드: 후보 모델을 비교 가능하게 만들기', 'models.py');
  await assertMlHelperRunsStage(page, '실험 흐름 코드', '실험 흐름 코드: 준비·학습·평가 연결하기', 'experiment.py');
  for (const hiddenLabel of ['리포트 코드', '결과 해석 코드']) {
    if (await page.getByRole('tab', { name: hiddenLabel }).count()) {
      throw new Error(`01_ml should not expose shallow helper tab: ${hiddenLabel}`);
    }
  }
  await page.getByRole('tab', { name: '실험 실행 코드' }).click();
  await page.locator('.code-block code', { hasText: '# 코드 읽기 힌트:' }).waitFor({ state: 'visible' });
  await page.locator('.code-block code', { hasText: 'run_stage(device)' }).waitFor({ state: 'visible' });
  const runStageButton = page.locator('[data-run-code]', { hasText: '실험 실행 코드 실행' });
  await runStageButton.waitFor({ state: 'visible' });
  const runStagePath = await runStageButton.getAttribute('data-run-path');
  if (runStagePath !== '01_ml/01_tabular_classification/run_stage.py') {
    throw new Error(`run_stage tab should keep its own runnable path, got ${runStagePath}`);
  }

  await selectStudyUnit(page, '01 ML', '02 표형 회귀');
  for (const hiddenLabel of ['데이터 준비 코드', '모델 코드', '리포트 코드', '결과 해석 코드']) {
    if (await page.getByRole('tab', { name: hiddenLabel }).count()) {
      throw new Error(`02_ml should hide shallow helper tab: ${hiddenLabel}`);
    }
  }
  await page.getByRole('tab', { name: '실험 흐름 코드' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: '실험 실행 코드' }).waitFor({ state: 'visible' });
}

async function assertMlHelperRunsStage(page, tabName, explanationText, sourceFile, options = {}) {
  await page.getByRole('tab', { name: tabName }).click();
  await page.locator('.code-explanation', { hasText: explanationText }).waitFor({ state: 'visible' });
  await page.locator('.run-target-note', { hasText: '실험 실행 코드' }).waitFor({ state: 'visible' });
  await page.locator('[data-run-cell]', { hasText: '함수 구조 보기' }).click();
  await page.locator('[data-cell-output]', { hasText: '선택 함수 미리보기' }).waitFor({ state: 'visible' });
  const probeText = await page.locator('[data-cell-output]').innerText();
  if (probeText.includes('실행 서버 필요') || probeText.includes('Unsupported method') || probeText.includes('403')) {
    throw new Error(`${tabName}: helper cell probe should work without execution errors:\n${probeText}`);
  }
  const button = page.locator('[data-run-code]', { hasText: '전체 ML 실험 실행' });
  await button.waitFor({ state: 'visible' });
  const data = await button.evaluate((element) => ({
    runPath: element.getAttribute('data-run-path'),
    sourcePath: element.getAttribute('data-run-source-path'),
  }));
  if (data.runPath !== '01_ml/01_tabular_classification/run_stage.py') {
    throw new Error(`${tabName}: helper tab should execute run_stage.py, got ${data.runPath}`);
  }
  if (data.sourcePath !== `01_ml/01_tabular_classification/${sourceFile}`) {
    throw new Error(`${tabName}: helper tab should preserve source path ${sourceFile}, got ${data.sourcePath}`);
  }
  if (options.executeStage) {
    await button.click();
    const output = page.locator('[data-run-output]');
    await output.locator('text=종료 코드:').waitFor({ state: 'visible', timeout: 150000 });
    const text = await output.innerText();
    if (!text.includes('종료 코드: 0')) {
      throw new Error(`${tabName}: helper run did not finish successfully:\n${text}`);
    }
    if (!text.includes('실행한 코드: 실험 실행 코드') || !text.includes('실행 환경:')) {
      throw new Error(`${tabName}: helper run should show learner-friendly target and runner summary:\n${text}`);
    }
    await page.locator('.artifact-viewer', { hasText: '실행 산출물 바로 보기' }).waitFor({ state: 'visible' });
  }
}

async function assertGuideAndQuizPersonalization(page) {
  await selectStudyUnit(page, '00 Foundations', '01 Tensor Shapes');
  const introGuide = await page.locator('.lesson-guide').innerText();
  if (!introGuide.includes('처음 학습 순서') || !introGuide.includes('학습 루프')) {
    throw new Error(`intro unit should teach the learner workflow:\n${introGuide}`);
  }
  if (introGuide.includes('이번 단원 브리핑')) {
    throw new Error(`intro unit should not use later-unit briefing copy:\n${introGuide}`);
  }
  const introQuiz = await page.locator('.quiz-panel').innerText();
  for (const token of ['matmul shape mismatch', '왼쪽 마지막 차원', 'batch 차원만 같으면']) {
    if (!introQuiz.includes(token)) throw new Error(`intro quiz should be tensor-specific (${token}):\n${introQuiz}`);
  }

  await selectStudyUnit(page, '01 ML', '01 Tabular Classification');
  const mlGuide = await page.locator('.lesson-guide').innerText();
  for (const token of ['이번 단원 브리핑', 'baseline model', 'primary metric', '자주 틀리는 지점']) {
    if (!mlGuide.includes(token)) throw new Error(`ML guide should include ${token}:\n${mlGuide}`);
  }
  if (mlGuide.includes('처음 학습 순서')) {
    throw new Error(`later unit should not repeat the onboarding order:\n${mlGuide}`);
  }
  const mlQuiz = await page.locator('.quiz-panel').innerText();
  for (const token of ['majority baseline', 'primary metric', 'confusion/error slice']) {
    if (!mlQuiz.includes(token)) throw new Error(`ML quiz should include ${token}:\n${mlQuiz}`);
  }
  if (mlQuiz === introQuiz) {
    throw new Error('ML quiz should not be identical to the intro quiz');
  }

  await selectStudyUnit(page, '09 Multimodal', '03 Visual Question Answering');
  const vqaGuide = await page.locator('.lesson-guide').innerText();
  const vqaQuiz = await page.locator('.quiz-panel').innerText();
  for (const token of ['answer type', 'shortcut bias', 'grounded reasoning']) {
    if (!vqaGuide.includes(token) && !vqaQuiz.includes(token)) {
      throw new Error(`VQA guide/quiz should include ${token}:\nGUIDE:\n${vqaGuide}\nQUIZ:\n${vqaQuiz}`);
    }
  }
  if (vqaQuiz === mlQuiz) {
    throw new Error('VQA quiz should not be identical to ML quiz');
  }
}

async function assertCoreCodeSummaries(page) {
  await selectStudyUnit(page, '00 Foundations', '03 Gradients and Backpropagation');
  await page.getByRole('tab', { name: '기초 실습 코드' }).click();
  await page.locator('.core-code-summary', { hasText: 'Gradient 실습은 이 네 덩어리만 먼저 읽으면 됩니다' }).waitFor({ state: 'visible' });
  await page.locator('.core-code-summary', { hasText: 'finite difference로 미분값 검산하기' }).waitFor({ state: 'visible' });
  await page.locator('.mini-code', { hasText: 'updated_weight = WEIGHT - (LEARNING_RATE * grad_w)' }).waitFor({ state: 'visible' });

  await selectStudyUnit(page, '00 Foundations', '04 Regularization and Normalization');
  await page.getByRole('tab', { name: '기초 실습 코드' }).click();
  await page.locator('.core-code-summary', { hasText: 'Normalization/Regularization 실습은 이 네 부분이 핵심입니다' }).waitFor({ state: 'visible' });
  await page.locator('.core-code-summary', { hasText: 'z-score normalization으로 입력 스케일 맞추기' }).waitFor({ state: 'visible' });
  await page.locator('.mini-code', { hasText: 'centered = values - values.mean()' }).waitFor({ state: 'visible' });
  await assertCoreCodeStepLabelsAreNotManuallyNumbered(page);
  await page.getByRole('tab', { name: '프레임워크 실습 코드' }).click();
  await page.locator('.core-code-summary', { hasText: 'LayerNorm·Dropout·Weight Decay' }).waitFor({ state: 'visible' });
  await page.locator('.mini-code', { hasText: 'optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=weight_decay)' }).waitFor({ state: 'visible' });
  await assertCoreCodeStepLabelsAreNotManuallyNumbered(page);
  await page.getByRole('tab', { name: '결과 해석 코드' }).click();
  await page.locator('.core-code-summary', { hasText: 'Regularization 해석 코드는 지표를 결론으로 바꾸는 흐름입니다' }).waitFor({ state: 'visible' });
  await page.locator('.core-code-summary', { hasText: 'weight decay가 같아 보이는 이유를 분리해서 해석하기' }).waitFor({ state: 'visible' });
  await page.locator('.mini-code', { hasText: "decay_objective = float(framework['weight_decay_regularized_objective_before_step'])" }).waitFor({ state: 'visible' });
  await page.locator('.mini-code', { hasText: 'OBSERVED_REPORT.write_text(observed_report' }).waitFor({ state: 'visible' });
  await assertCoreCodeStepLabelsAreNotManuallyNumbered(page);

  await selectStudyUnit(page, '04 NLP', '03 Machine Reading Comprehension');
  await page.getByRole('tab', { name: '프레임워크 실습 코드' }).click();
  await page.locator('.core-code-summary', { hasText: '긴 프레임워크 실습은 데이터→모델→평가 흐름만 먼저 보세요' }).waitFor({ state: 'visible' });
  await page.locator('.core-code-summary', { hasText: '판단 지표 계산: token_f1()' }).waitFor({ state: 'visible' });
  await page.locator('.mini-code', { hasText: 'def token_f1' }).waitFor({ state: 'visible' });
}

async function assertCoreCodeStepLabelsAreNotManuallyNumbered(page) {
  const labels = await page.locator('.core-code-summary li strong').evaluateAll((nodes) => nodes.map((node) => node.textContent.trim()));
  const duplicated = labels.filter((label) => /^\d+[.)]\s+/.test(label));
  if (duplicated.length) {
    throw new Error(`Core code labels should rely on <ol> numbering only, but found manual labels: ${duplicated.join(', ')}`);
  }
}

async function assertLearningRouteAndSelfChecks(page) {
  await selectStudyUnit(page, '06 Training Systems', '01 Torchrun and DDP Basics');
  const titleBeforeRouteChange = await page.locator('#detail-title').innerText();

  await page.locator('#route-select').selectOption('multimodal');
  await page.locator('.route-box', { hasText: '학습 경로 먼저 선택' }).waitFor({ state: 'visible' });
  await page.locator('.route-box', { hasText: '현재 읽던 단원은 유지' }).waitFor({ state: 'visible' });
  await page.locator('#route-card', { hasText: 'Multimodal/VLA 경로' }).waitFor({ state: 'visible' });
  await page.locator('#route-card', { hasText: '다음 단원 추천' }).waitFor({ state: 'visible' });
  await page.locator('#detail-title', { hasText: titleBeforeRouteChange }).waitFor({ state: 'visible' });

  await page.locator('.self-checklist [data-self-check]').first().check();
  await page.locator('[data-self-check-summary]', { hasText: '1/' }).waitFor({ state: 'visible' });
  await page.locator('.self-checklist', { hasText: '설명할 수 있다' }).waitFor({ state: 'visible' });
  await page.locator('#unit-note').fill('wrong-note: route persists after reload');
  await page.locator('#unit-note').dispatchEvent('change');

  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('#route-card', { hasText: 'Multimodal/VLA 경로' }).waitFor({ state: 'visible' });
  await page.locator('#detail-title', { hasText: titleBeforeRouteChange }).waitFor({ state: 'visible' });
  const checkedAfterReload = await page.locator('.self-checklist [data-self-check]').first().isChecked();
  if (!checkedAfterReload) {
    throw new Error('self-check progress should persist after reload');
  }
  const noteAfterReload = await page.locator('#unit-note').inputValue();
  if (noteAfterReload !== 'wrong-note: route persists after reload') {
    throw new Error(`unit note should persist after reload, got: ${noteAfterReload}`);
  }

  await page.locator('#route-card [data-route-next]').click();
  await page.locator('#detail-title', { hasText: '01 Tensor Shapes' }).waitFor({ state: 'visible' });
}

async function runDesktopQa(browser, baseUrl) {
  const { context, externalRequests } = await newLocalOnlyContext(browser, baseUrl, { viewport: { width: 1440, height: 980 } });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => document.fonts?.ready);
  await page.locator('#runtime-badge', { hasText: '실행 가능' }).waitFor({ state: 'visible' });
  await assertTrackMarkdownRendered(page);
  await assertNoHorizontalOverflow(page, 'desktop-readme');
  await assertLearningRouteAndSelfChecks(page);
  await assertInjectedPythonComments(page, '기초 실습 코드');
  await assertRunButton(page, '기초 실습 코드', 'matmul_shape');
  await assertQuizAndWrongNotes(page);
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-run-output.png'), fullPage: false });
  await page.getByText('코드 읽기 안내').waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '이 파일은 무엇인가' }).waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '실행하면 남는 결과' }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: /기초 실습 실행 완료로 표시/ }).click();
  await expectTabComplete(page, '기초 실습 코드');
  await assertInjectedPythonComments(page, '프레임워크 실습 코드');
  await page.getByText('프레임워크 실습 코드: 실제 도구').waitFor({ state: 'visible' });
  await assertNoHorizontalOverflow(page, 'desktop-framework');
  await assertInjectedPythonComments(page, '결과 해석 코드');
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-code-reader.png'), fullPage: false });
  await page.getByRole('button', { name: /10 VLA/ }).click();
  await page.getByRole('button', { name: /01 VLA Vision-Language-Action Grounding/ }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: /01 VLA Vision-Language-Action Grounding/ }).click();
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('.track-card[aria-pressed="true"]', { hasText: /10 VLA/ }).waitFor({ state: 'visible' });
  await page.locator('.unit-card[aria-current="true"]', { hasText: /01 VLA Vision-Language-Action Grounding/ }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: '학습 안내' }).click();
  await page.getByText('02 Study Guide').waitFor({ state: 'visible' });
  await assertBridgeResources(page);
  await assertMlRunnerResources(page);
  await assertGuideAndQuizPersonalization(page);
  await assertCoreCodeSummaries(page);

  const metrics = await assertNoHorizontalOverflow(page, 'desktop');
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-study-guide.png'), fullPage: false });
  await context.close();
  assertLocalOnlyRequests(externalRequests, 'desktop');

  if (consoleErrors.length) {
    throw new Error(`Console errors during desktop QA:\n${consoleErrors.join('\n')}`);
  }
  return metrics;
}

async function expectTabComplete(page, tabName) {
  const tab = page.getByRole('tab', { name: tabName });
  await tab.waitFor({ state: 'visible' });
  await expectAttribute(tab, 'data-complete', 'true');
  await tab.locator('.tab-done-mark', { hasText: '✓' }).waitFor({ state: 'visible' });
}

async function expectAttribute(locator, name, value) {
  await locator.evaluate(
    (element, [attributeName, expected]) => {
      if (element.getAttribute(attributeName) !== expected) {
        throw new Error(`${attributeName} expected ${expected}, got ${element.getAttribute(attributeName)}`);
      }
    },
    [name, value],
  );
}

async function runResponsiveQa(browser, baseUrl) {
  const context = await browser.newContext({ viewport: { width: 1024, height: 900 } });
  const page = await context.newPage();
  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => document.fonts?.ready);
  await page.getByRole('tab', { name: '단원 안내' }).waitFor({ state: 'visible' });
  await assertReaderBeforeGuideOnNarrow(page, 'tablet');
  const metrics = await assertNoHorizontalOverflow(page, 'tablet');
  await page.screenshot({ path: path.join(OUT_DIR, 'tablet-reader.png'), fullPage: false });
  await context.close();
  return metrics;
}

async function runMobileQa(browser, baseUrl) {
  const { context, externalRequests } = await newLocalOnlyContext(browser, baseUrl, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => document.fonts?.ready);
  await page.getByRole('tab', { name: '단원 안내' }).waitFor({ state: 'visible' });
  await assertReaderBeforeGuideOnNarrow(page, 'mobile');
  await assertInjectedPythonComments(page, '결과 해석 코드');
  await page.getByText('결과 해석 코드: 실행 결과').waitFor({ state: 'visible' });
  const metrics = await assertNoHorizontalOverflow(page, 'mobile');
  await page.screenshot({ path: path.join(OUT_DIR, 'mobile-code-reader.png'), fullPage: false });
  await context.close();
  assertLocalOnlyRequests(externalRequests, 'mobile');
  return metrics;
}

async function runStaticServerRunHelpQa(browser, baseUrl) {
  const { context, externalRequests } = await newLocalOnlyContext(browser, baseUrl, { viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await selectStudyUnit(page, '00 Foundations', '01 Tensor Shapes');
  await page.getByRole('tab', { name: '기초 실습 코드' }).click();
  await page.locator('[data-run-code]', { hasText: '기초 실습 코드 실행' }).click();
  const output = page.locator('[data-run-output]');
  await output.locator('text=읽기 전용 서버').waitFor({ state: 'visible' });
  const text = await output.innerText();
  if (!text.includes('501 Unsupported method')) {
    throw new Error(`static server help should explain the 501 POST failure:\n${text}`);
  }
  if (text.includes('<!DOCTYPE HTML>')) {
    throw new Error(`static server help should not expose the raw HTML error body:\n${text}`);
  }
  await page.screenshot({ path: path.join(OUT_DIR, 'static-server-run-help.png'), fullPage: false });
  await context.close();
  assertLocalOnlyRequests(externalRequests, 'static-server-help');
  return { explained501: true };
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const staticServerHelp = await withPlainHttpServer(async (baseUrl) => {
    const browser = await chromium.launch({ headless: true });
    try {
      return await runStaticServerRunHelpQa(browser, baseUrl);
    } finally {
      await browser.close();
    }
  });
  await withStaticServer(async (baseUrl) => {
    const browser = await chromium.launch({ headless: true });
    try {
      const desktop = await runDesktopQa(browser, baseUrl);
      const tablet = await runResponsiveQa(browser, baseUrl);
      const mobile = await runMobileQa(browser, baseUrl);
      console.log(JSON.stringify({ ok: true, baseUrl, outDir: OUT_DIR, staticServerHelp, desktop, tablet, mobile }, null, 2));
    } finally {
      await browser.close();
    }
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
