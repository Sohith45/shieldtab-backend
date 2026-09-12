const params = new URLSearchParams(window.location.search);
const blockedUrl = params.get("url") || "";
const threatType = params.get("type") || "unknown";
const confidence = params.get("confidence") || "";

document.getElementById("urlBox").textContent = blockedUrl;
document.getElementById("threatType").textContent = threatType;
document.getElementById("confidenceBadge").textContent = confidence
  ? `${Math.round(parseFloat(confidence) * 100)}% confidence`
  : "";
document.getElementById("goBackBtn").addEventListener("click", () => {
  if (window.history.length > 1) {
    window.history.back();
  } else {
    window.location.href = "https://www.google.com";
  }
});

document.getElementById("proceedBtn").addEventListener("click", () => {
  if (!blockedUrl) return;
  const confirmed = confirm(
    "Are you sure you want to proceed? This site was flagged as potentially " +
      threatType + ". Only continue if you trust this source."
  );
  if (!confirmed) return;

  chrome.runtime.sendMessage(
    { type: "PROCEED_ANYWAY", url: blockedUrl },
    () => {
      // background.js handles the actual tab redirect
    }
  );
});

document.getElementById("explainBtn").addEventListener("click", async () => {
  const btn = document.getElementById("explainBtn");
  const content = document.getElementById("explainContent");

  if (content.style.display !== "none") {
    content.style.display = "none";
    btn.textContent = "Why was this flagged?";
    return;
  }

  content.style.display = "block";
  content.innerHTML = '<span class="loading">Looking up similar known threats...</span>';
  btn.textContent = "Hide explanation";

  try {
    const res = await fetch(SHIELDTAB_CONFIG.BACKEND_URL + "/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: blockedUrl, threat_type: threatType }),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error("Backend error " + res.status);
    const data = await res.json();

    let html = `<p>${escapeHtml(data.summary)}</p>`;
    if (data.similar_threats && data.similar_threats.length > 0) {
      html += '<p style="margin-top:10px;font-weight:600;">Similar known threats:</p>';
      for (const t of data.similar_threats) {
        html += `<div class="similar-item">
          <span class="sim-badge">${Math.round(t.similarity * 100)}% similar · ${escapeHtml(t.type)}</span><br>
          ${escapeHtml(t.url)}
        </div>`;
      }
    }
    content.innerHTML = html;
  } catch (err) {
    content.innerHTML = '<span class="loading">Explanation unavailable right now.</span>';
  }
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
