/**
 * ShieldTab background service worker (Manifest V3).
 *
 * Flow:
 *  1. webNavigation.onBeforeNavigate fires before a page loads.
 *  2. Check chrome.storage.local cache first (avoid re-hitting backend for
 *     recently-seen URLs).
 *  3. If not cached, POST the URL to the ShieldTab backend /predict.
 *  4. If malicious, redirect the tab to warning.html with the verdict info
 *     before the real page has a chance to render.
 */

importScripts("config.js");

const BACKEND_URL = SHIELDTAB_CONFIG.BACKEND_URL + "/predict";
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes
const REQUEST_TIMEOUT_MS = 2500; // fail open (allow) if backend is slow/unreachable

// Skip internal/browser pages -- nothing to check there.
function shouldSkip(url) {
  return (
    !url ||
    url.startsWith("chrome://") ||
    url.startsWith("chrome-extension://") ||
    url.startsWith("about:") ||
    url.startsWith("edge://") ||
    url.startsWith("file://")
  );
}

async function getCached(url) {
  const key = "verdict:" + url;
  const data = await chrome.storage.local.get(key);
  const entry = data[key];
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL_MS) return null;
  return entry.verdict;
}

async function setCached(url, verdict) {
  const key = "verdict:" + url;
  await chrome.storage.local.set({
    [key]: { verdict, timestamp: Date.now() },
  });
}

async function checkUrl(url) {
  const cached = await getCached(url);
  if (cached) return cached;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error("Backend error " + res.status);
    const verdict = await res.json();
    await setCached(url, verdict);
    return verdict;
  } catch (err) {
    clearTimeout(timeout);
    console.warn("ShieldTab: backend check failed, failing open:", err.message);
    return null; // fail open -- never block due to a backend/network issue
  }
}

async function incrementStat(field) {
  const today = new Date().toISOString().slice(0, 10);
  const key = "stats:" + today;
  const data = await chrome.storage.local.get([key]);
  const stats = data[key] || { blocked: 0, checked: 0 };
  stats[field] += 1;
  await chrome.storage.local.set({ [key]: stats });
}

async function isProtectionEnabled() {
  const data = await chrome.storage.local.get(["protectionEnabled"]);
  return data.protectionEnabled !== false; // default true
}

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  // Only act on the top-level frame (not iframes/ads embedded in a safe page)
  if (details.frameId !== 0) return;
  const url = details.url;
  if (shouldSkip(url)) return;
  if (!(await isProtectionEnabled())) return;

  await incrementStat("checked");
  const verdict = await checkUrl(url);
  if (verdict && verdict.is_malicious) {
    await incrementStat("blocked");
    const warningUrl =
      chrome.runtime.getURL("warning.html") +
      "?url=" + encodeURIComponent(url) +
      "&type=" + encodeURIComponent(verdict.threat_type) +
      "&confidence=" + encodeURIComponent(verdict.confidence);

    chrome.tabs.update(details.tabId, { url: warningUrl });
  }
});

// Let a user "proceed anyway" from the warning page without being
// re-blocked on that exact URL for this session.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "PROCEED_ANYWAY") {
    setCached(msg.url, { is_malicious: false, threat_type: "user_override" }).then(() => {
      chrome.tabs.update(sender.tab.id, { url: msg.url });
      sendResponse({ ok: true });
    });
    return true; // keep the message channel open for the async response
  }
});
