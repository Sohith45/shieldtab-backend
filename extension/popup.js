const BACKEND_HEALTH_URL = SHIELDTAB_CONFIG.BACKEND_URL + "/health";

async function checkBackend() {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch(BACKEND_HEALTH_URL, { signal: AbortSignal.timeout(1500) });
    const data = await res.json();
    if (data.status === "ok" && data.models_loaded) {
      dot.classList.remove("offline");
      text.textContent = "Protection active";
    } else {
      dot.classList.add("offline");
      text.textContent = "Backend starting up...";
    }
  } catch (err) {
    dot.classList.add("offline");
    text.textContent = "Backend unreachable (fail-open mode)";
  }
}

async function loadStats() {
  const today = new Date().toISOString().slice(0, 10);
  const data = await chrome.storage.local.get(["stats:" + today]);
  const stats = data["stats:" + today] || { blocked: 0, checked: 0 };
  document.getElementById("blockedCount").textContent = stats.blocked;
  document.getElementById("checkedCount").textContent = stats.checked;
}

async function loadToggleState() {
  const data = await chrome.storage.local.get(["protectionEnabled"]);
  const enabled = data.protectionEnabled !== false; // default true
  document.getElementById("protectionToggle").checked = enabled;
}

document.getElementById("protectionToggle").addEventListener("change", (e) => {
  chrome.storage.local.set({ protectionEnabled: e.target.checked });
});

checkBackend();
loadStats();
loadToggleState();
