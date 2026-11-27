/*
 * 示例页脚本的行为守卫（node 版）。
 *
 * 用法：node card-check.js <index.html 路径>
 *
 * 做法：把页面里的内联 <script> 抽出来，在最小 DOM 桩里执行，然后**逐张调用能力卡片**，
 * 校验两件事：
 *   1. 请求格式与卡片字段匹配 —— 带文件的卡片必须发 multipart，带文本域的 JSON 卡片必须发 JSON 请求体
 *   2. 调用过程不抛异常、不出现「请求体不是合法 JSON」
 *
 * 这类错误在浏览器里才会炸（页面整体静默失效），构建期完全看不见，
 * 所以用 node 在 CI 里把它变成一条会红的断言。
 */

/* eslint-disable no-undef */

const fs = require('fs');

const htmlPath = process.argv[2];
if (!htmlPath) {
  console.error('用法: node card-check.js <index.html 路径>');
  process.exit(2);
}

const noop = () => {};

function fakeEl(extra) {
  return Object.assign({
    innerHTML: '', textContent: '', value: '', files: [], disabled: false, type: '',
    dataset: {}, style: {},
    classList: {add: noop, remove: noop, contains: () => false},
    setAttribute: noop, getAttribute: () => null, appendChild: noop,
    querySelector: () => fakeEl(), querySelectorAll: () => [],
    closest: () => fakeEl(), addEventListener: noop, removeEventListener: noop
  }, extra || {});
}

global.document = {
  addEventListener: noop,
  querySelector: () => fakeEl(),
  querySelectorAll: () => [],
  createElement: () => fakeEl(),
  body: fakeEl()
};
global.window = global;
global.addEventListener = noop;
global.URL = {createObjectURL: () => ''};

let lastCall = null;
global.fetch = async (path, opts) => {
  lastCall = {path: path, opts: opts};
  return {headers: {get: () => 'application/json'}, json: async () => ({ok: true})};
};

const html = fs.readFileSync(htmlPath, 'utf8');
const matched = /<script[^>]*>([\s\S]*?)<\/script>/.exec(html);
if (!matched) {
  console.error('页面里没有找到内联 <script> 块');
  process.exit(2);
}
const script = matched[1];

/* 把能力卡片数组与调度函数暴露出来 */
const exportCaps = '\n;globalThis.__CAPS = {FACE_CAPS, DOC_CAPS, IMAGE_CAPS, DETECT_CAPS,'
  + ' MISC_CAPS, SCENARIO_CAPS, runCap};';
new Function(script + exportCaps)();
const CAPS = globalThis.__CAPS;

function fieldValue(cap, kind) {
  const field = cap.f.filter(f => f.y === kind)[0];
  return field && field.v ? field.v : '{}';
}

async function checkCard(group, cap) {
  const problems = [];
  const hasTextarea = cap.f.some(f => f.y === 'textarea');
  const hasFile = cap.f.some(f => f.y === 'file');
  const outEl = {innerHTML: '', textContent: 'out'};
  const card = {
    dataset: {out: cap.o, path: cap.p},
    querySelector: (sel) => {
      if (sel === 'textarea') {
        return hasTextarea ? {value: fieldValue(cap, 'textarea')} : null;
      }
      if (sel === '.out') {
        return outEl;
      }
      if (sel === '.run') {
        return {innerHTML: '', disabled: false};
      }
      return null;
    },
    querySelectorAll: () => []
  };

  lastCall = null;
  try {
    await CAPS.runCap(card);
  } catch (e) {
    problems.push('调用抛异常: ' + e.name + ': ' + e.message);
  }

  if (!lastCall) {
    problems.push('没有发起请求');
  } else {
    const isJsonBody = typeof lastCall.opts.body === 'string';
    const isForm = lastCall.opts.body instanceof FormData;
    if (hasFile && !isForm) {
      problems.push('带文件的卡片应以 multipart 发送，实际 ' + (isJsonBody ? 'JSON' : typeof lastCall.opts.body));
    } else if (!hasFile && cap.o === 'json' && !isJsonBody) {
      problems.push('JSON 卡片应以 JSON 请求体发送');
    } else if (!hasFile && cap.o !== 'json' && !isForm) {
      problems.push('应以 multipart 发送');
    }
  }
  if (/请求体不是合法 JSON|Cannot read/.test(outEl.innerHTML)) {
    problems.push('页面报错: ' + outEl.innerHTML.replace(/<[^>]*>/g, '').slice(0, 120));
  }
  if (problems.length) {
    console.log('❌ ' + group + ' → ' + cap.t);
    problems.forEach(p => console.log('     ' + p));
  }
  return problems.length;
}

(async () => {
  let total = 0;
  let bad = 0;
  for (const group of Object.keys(CAPS).filter(k => k.endsWith('_CAPS'))) {
    for (const cap of CAPS[group]) {
      total++;
      bad += await checkCard(group, cap);
    }
  }
  console.log('共 ' + total + ' 张能力卡片，异常 ' + bad + ' 张');
  process.exit(bad ? 1 : 0);
})();
