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
  for (const token of ['# 학습자용 한글 주석', '# 이 파일은 무엇인가:', '# 어떻게 읽으면 좋은가:', '# 실행하면 남는 결과:', '# 아래부터 원본 Python 코드입니다.']) {
    if (!codeText.includes(token)) {
      throw new Error(`${tabName}: missing injected Korean code comment ${token}`);
    }
  }
  const firstLines = codeText.trimStart().split('\n').slice(0, 8).join('\n');
  if (!firstLines.includes('# 이 파일은 무엇인가:')) {
    throw new Error(`${tabName}: Korean guidance comments should start the displayed code block`);
  }
  if (await page.locator('.learner-comment').count()) {
    throw new Error(`${tabName}: guidance must be injected into code, not rendered as a separate learner-comment block`);
  }
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
  await assertResourceDocument(page, 'Decoder generation bridge', 'KV-cache intuition');

  await selectStudyUnit(page, '09 Multimodal', '02 Image Captioning');
  await assertResourceDocument(page, 'Multimodal generation bridge', 'Grounding failure vs retrieval failure');

  await selectStudyUnit(page, '10 VLA', '01 VLA Vision-Language-Action Grounding');
  await assertResourceDocument(page, 'RL to VLA bridge', 'Behavior cloning vs RL vs offline RL');
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
  await assertNoHorizontalOverflow(page, 'desktop-readme');
  await assertInjectedPythonComments(page, 'scratch_lab.py');
  await page.getByText('코드 읽기 안내').waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '이 파일은 무엇인가' }).waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '실행하면 남는 결과' }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: /scratch lab 체크/ }).click();
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
  await page.getByRole('button', { name: 'Study guide' }).click();
  await page.getByText('02 Study Guide').waitFor({ state: 'visible' });
  await assertBridgeResources(page);

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

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  await withStaticServer(async (baseUrl) => {
    const browser = await chromium.launch({ headless: true });
    try {
      const desktop = await runDesktopQa(browser, baseUrl);
      const tablet = await runResponsiveQa(browser, baseUrl);
      const mobile = await runMobileQa(browser, baseUrl);
      console.log(JSON.stringify({ ok: true, baseUrl, outDir: OUT_DIR, desktop, tablet, mobile }, null, 2));
    } finally {
      await browser.close();
    }
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
