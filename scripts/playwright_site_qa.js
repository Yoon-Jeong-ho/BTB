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
  const metrics = await page.evaluate(() => ({
    innerWidth: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    readerWidth: document.querySelector('.reader-panel')?.getBoundingClientRect().width || 0,
    contentWidth: document.querySelector('.lesson-content')?.getBoundingClientRect().width || 0,
    navWidth: document.querySelector('.track-panel')?.getBoundingClientRect().width || 0,
  }));
  if (metrics.scrollWidth > metrics.innerWidth + 1) {
    throw new Error(`${label}: horizontal overflow ${metrics.scrollWidth} > ${metrics.innerWidth}`);
  }
  if (metrics.innerWidth >= 1200 && metrics.readerWidth < 850) {
    throw new Error(`${label}: reader is too narrow (${metrics.readerWidth}px)`);
  }
  if (metrics.innerWidth >= 1200 && metrics.navWidth > 380) {
    throw new Error(`${label}: left navigation is too wide (${metrics.navWidth}px)`);
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

async function runDesktopQa(browser, baseUrl) {
  const { context, externalRequests } = await newLocalOnlyContext(browser, baseUrl, { viewport: { width: 1440, height: 980 } });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => document.fonts?.ready);
  await page.getByRole('tab', { name: 'scratch_lab.py' }).click();
  await page.getByText('코드 읽기 안내').waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '이 파일은 무엇인가' }).waitFor({ state: 'visible' });
  await page.locator('.code-explanation dt', { hasText: '실행하면 남는 결과' }).waitFor({ state: 'visible' });
  await page.getByText('# 학습자용 한글 주석').waitFor({ state: 'visible' });
  await page.getByRole('button', { name: /scratch lab 체크/ }).click();
  await page.getByRole('tab', { name: 'framework_lab.py' }).click();
  await page.getByText('framework_lab.py는 같은 아이디어').waitFor({ state: 'visible' });
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-code-reader.png'), fullPage: false });
  await page.getByRole('button', { name: 'Study guide' }).click();
  await page.getByText('02 Study Guide').waitFor({ state: 'visible' });

  const metrics = await assertNoHorizontalOverflow(page, 'desktop');
  await page.screenshot({ path: path.join(OUT_DIR, 'desktop-study-guide.png'), fullPage: false });
  await context.close();
  assertLocalOnlyRequests(externalRequests, 'desktop');

  if (consoleErrors.length) {
    throw new Error(`Console errors during desktop QA:\n${consoleErrors.join('\n')}`);
  }
  return metrics;
}

async function runMobileQa(browser, baseUrl) {
  const { context, externalRequests } = await newLocalOnlyContext(browser, baseUrl, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  await page.goto(`${baseUrl}/web/`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => document.fonts?.ready);
  await page.getByRole('tab', { name: 'README' }).waitFor({ state: 'visible' });
  await page.getByRole('tab', { name: 'analysis.py' }).click();
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
      const mobile = await runMobileQa(browser, baseUrl);
      console.log(JSON.stringify({ ok: true, baseUrl, outDir: OUT_DIR, desktop, mobile }, null, 2));
    } finally {
      await browser.close();
    }
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
