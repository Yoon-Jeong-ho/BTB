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
  await insights.locator('text=좋은 결과 기준').waitFor({ state: 'visible' });
  await insights.locator('text=다음 질문').waitFor({ state: 'visible' });
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
  await page.getByRole('tab', { name: 'dataset.py' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: 'experiment.py' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: 'run_stage.py' }).waitFor({ state: 'visible' });
  if (await page.getByRole('tab', { name: 'scratch_lab.py' }).count()) {
    throw new Error('01_ml should not show missing scratch_lab.py tab');
  }
  await page.getByRole('tab', { name: 'dataset.py' }).click();
  await page.locator('.code-explanation', { hasText: 'dataset.py는 ML stage의 입력 표를 만드는 코드입니다.' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: 'experiment.py' }).click();
  await page.locator('.code-explanation', { hasText: 'experiment.py는 ML stage의 실제 실험 흐름입니다.' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: 'run_stage.py' }).click();
  await page.locator('.code-block code', { hasText: '# 코드 읽기 힌트:' }).waitFor({ state: 'visible' });
  await page.locator('.code-block code', { hasText: 'run_stage(device)' }).waitFor({ state: 'visible' });
  await page.locator('[data-run-code]', { hasText: 'run_stage.py 실행' }).waitFor({ state: 'visible' });
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

  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('#route-card', { hasText: 'Multimodal/VLA 경로' }).waitFor({ state: 'visible' });
  await page.locator('#detail-title', { hasText: titleBeforeRouteChange }).waitFor({ state: 'visible' });
  const checkedAfterReload = await page.locator('.self-checklist [data-self-check]').first().isChecked();
  if (!checkedAfterReload) {
    throw new Error('self-check progress should persist after reload');
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
  await assertTrackMarkdownRendered(page);
  await assertNoHorizontalOverflow(page, 'desktop-readme');
  await assertLearningRouteAndSelfChecks(page);
  await assertInjectedPythonComments(page, 'scratch_lab.py');
  await assertRunButton(page, 'scratch_lab.py', 'matmul_shape');
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-run-output.png'), fullPage: false });
  await page.getByText('코드 읽기 안내').waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '이 파일은 무엇인가' }).waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '실행하면 남는 결과' }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: /scratch lab 완료 표시/ }).click();
  await assertInjectedPythonComments(page, 'framework_lab.py');
  await page.getByText('framework_lab.py는 같은 아이디어').waitFor({ state: 'visible' });
  await assertNoHorizontalOverflow(page, 'desktop-framework');
  await assertInjectedPythonComments(page, 'analysis.py');
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

  const metrics = await assertNoHorizontalOverflow(page, 'desktop');
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-study-guide.png'), fullPage: false });
  await context.close();
  assertLocalOnlyRequests(externalRequests, 'desktop');

  if (consoleErrors.length) {
    throw new Error(`Console errors during desktop QA:\n${consoleErrors.join('\n')}`);
  }
  return metrics;
}

async function runResponsiveQa(browser, baseUrl) {
  const context = await browser.newContext({ viewport: { width: 1024, height: 900 } });
  const page = await context.newPage();
  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => document.fonts?.ready);
  await page.getByRole('tab', { name: 'README' }).waitFor({ state: 'visible' });
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
  await page.getByRole('tab', { name: 'README' }).waitFor({ state: 'visible' });
  await assertInjectedPythonComments(page, 'analysis.py');
  await page.getByText('analysis.py는 실행 결과').waitFor({ state: 'visible' });
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
  await page.getByRole('tab', { name: 'scratch_lab.py' }).click();
  await page.locator('[data-run-code]', { hasText: 'scratch_lab.py 실행' }).click();
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
