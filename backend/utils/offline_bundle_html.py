# Embedded HTML/CSS/JS shipped inside the downloadable OpenVINO bundle.
# Personalization: __DISPLAY_NAME__ in OFFLINE_HOME_HTML is replaced server-side after html.escape.

OFFLINE_HOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EasyDefect</title>
  <style>
    :root {
      --bg: #020617;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --primary: #2dd4bf;
      --panel: #0f172a;
      --border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1000px 500px at 10% -20%, #1e293b 0%, var(--bg) 60%);
      color: var(--text);
    }
    .wrap {
      max-width: 560px;
      margin: 0 auto;
      padding: 52px 24px;
      text-align: center;
    }
    .brand { font-weight: 800; letter-spacing: 0.06em; color: var(--primary); font-size: 0.9rem; margin-bottom: 14px; }
    h1 { font-size: 1.85rem; font-weight: 700; margin: 0 0 16px; line-height: 1.25; }
    .lead { color: var(--muted); font-size: 1.05rem; line-height: 1.6; margin-bottom: 32px; }
    .btn {
      display: inline-block;
      padding: 14px 32px;
      background: var(--primary);
      color: #021617;
      font-weight: 700;
      text-decoration: none;
      border-radius: 10px;
    }
    .note { margin-top: 36px; font-size: 13px; color: var(--muted); line-height: 1.55; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="brand">EASYDEFECT</div>
    <h1>Welcome __DISPLAY_NAME__ to Easy Defect</h1>
    <p class="lead">
      You&apos;re offline with your exported model. Open testing to classify product images — no separation into &quot;normal&quot; vs &quot;defect&quot; uploads; EasyDefect labels each photo for you.
    </p>
    <a class="btn" href="test.html">Go to testing</a>
    <p class="note">
      Tip: Always start the local server before opening testing (see <strong>README.txt</strong>). Use <strong>Open_Testing_Page</strong> to launch browser + server together.
    </p>
  </div>
</body>
</html>
"""

OFFLINE_TEST_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EasyDefect – Testing</title>
  <style>
    :root {
      --bg: #020617;
      --panel: #0f172a;
      --panel-border: #1f2937;
      --muted: #94a3b8;
      --text: #e2e8f0;
      --primary: #2dd4bf;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1000px 500px at 10% -20%, #1e293b 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
    }
    .top {
      padding: 16px 24px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }
    .top a.home { color: var(--muted); text-decoration: none; font-size: 14px; }
    .top a.home:hover { color: var(--primary); }
    .top .ttl { font-weight: 700; letter-spacing: 0.04em; color: var(--primary); }
    main { padding: 28px 20px 48px; max-width: 980px; margin: 0 auto; }
    main h1 { text-align: center; margin-bottom: 6px; }
    main .sub { text-align: center; color: var(--muted); margin-bottom: 20px; }
    .error-text { text-align: center; color: #fca5a5; min-height: 1.25em; }
    .panel {
      background: var(--panel);
      padding: 24px;
      border-radius: 14px;
      border: 1px solid var(--panel-border);
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .image-card {
      position: relative;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--panel-border);
      background: #111827;
    }
    .image-card.preview-def { border-color: #ef4444; }
    .image-card.preview-ok { border-color: #22c55e; }
    .image-card img {
      width: 100%;
      height: 120px;
      object-fit: cover;
      display: block;
    }
    .delete-btn {
      position: absolute;
      top: 5px;
      right: 5px;
      background: rgba(0, 0, 0, 0.72);
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      cursor: pointer;
      line-height: 22px;
      padding: 0;
      font-size: 13px;
    }
    .result-line {
      margin: 0;
      padding: 8px;
      text-align: center;
      font-size: 13px;
      color: var(--muted);
      border-top: 1px solid var(--panel-border);
      background: #0d1324;
      line-height: 1.3;
    }
    .file-name {
      margin: 0 0 4px;
      color: #cbd5e1;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .detected-label {
      margin: 0;
      font-weight: 700;
    }
    .normal-label { color: #86efac; }
    .defective-label { color: #fca5a5; }
    .btn {
      min-width: 220px;
      padding: 12px 22px;
      border-radius: 10px;
      border: none;
      cursor: pointer;
      font-weight: 700;
      margin-top: 16px;
    }
    .btn.primary { background: var(--primary); color: #021617; }
    .btn:disabled { opacity: 0.55; cursor: not-allowed; }
    .actions { text-align: center; }
    .note { margin-top: 12px; font-size: 13px; color: var(--muted); text-align: center; }
    input[type=file] {
      width: 100%;
      color: #cbd5e1;
      margin-bottom: 8px;
    }
    @media (max-width: 520px) { .btn { width: 100%; } }
  </style>
</head>
<body>
  <header class="top">
    <span class="ttl">EasyDefect</span>
    <a class="home" href="index.html">← Welcome / Home</a>
  </header>
  <main>
    <h1>Test images</h1>
    <p class="sub">Upload any product photos in one list (no separate normal vs defective piles). Results appear under each thumbnail.</p>
    <p id="errorMsg" class="error-text"></p>
    <div class="panel">
      <input id="filesInput" type="file" accept=".jpg,.jpeg,.png,.bmp" multiple>
      <div class="actions">
        <button id="testBtn" class="btn primary" type="button">Run test</button>
      </div>
      <p class="note">After testing, each image shows <strong>Detected as: Normal</strong> or <strong>Defective</strong> at the bottom.</p>
      <div id="grid" class="image-grid"></div>
    </div>
  </main>
  <script>
    const state = { images: [], isTesting: false };
    const el = {
      error: document.getElementById("errorMsg"),
      filesInput: document.getElementById("filesInput"),
      grid: document.getElementById("grid"),
      testBtn: document.getElementById("testBtn"),
    };

    function labelClass(pred) {
      if (pred === "Normal") return "result-line normal-label";
      if (pred === "Defective") return "result-line defective-label";
      return "result-line";
    }

    function cardBorderClass(pred, done) {
      if (!done) return "";
      if (pred === "Defective") return "preview-def";
      if (pred === "Normal") return "preview-ok";
      return "";
    }

    function caption(img, tested) {
      const safeName = escapeHtml(img.file.name);
      let result = "Not tested yet";
      if (tested && img.prediction === "Normal") result = "Detected as: Normal";
      else if (tested && img.prediction === "Defective") result = "Detected as: Defective";
      else if (tested) result = "Detected as: Error";
      return `<p class="file-name" title="${safeName}">${safeName}</p><p class="detected-label">${result}</p>`;
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function renderGrid() {
      el.grid.innerHTML = "";
      state.images.forEach((img, i) => {
        const tested = img.prediction !== "Not tested";
        const card = document.createElement("div");
        card.className = "image-card " + cardBorderClass(img.prediction, tested);
        const capCls = tested ? labelClass(img.prediction) : "result-line";
        card.innerHTML = `
          <img src="${img.url}" alt="upload">
          <button type="button" class="delete-btn" data-index="${i}">\u2715</button>
          <div class="${capCls}">${caption(img, tested)}</div>
        `;
        el.grid.appendChild(card);
      });
    }

    function render() {
      renderGrid();
      el.testBtn.disabled = state.isTesting;
      el.filesInput.disabled = state.isTesting;
      el.testBtn.textContent = state.isTesting ? "Working\u2026" : "Run test";
    }

    function pushFiles(files) {
      Array.from(files || []).forEach((file) => {
        state.images.push({
          file,
          url: URL.createObjectURL(file),
          prediction: "Not tested",
        });
      });
      render();
    }

    el.filesInput.addEventListener("change", (e) => pushFiles(e.target.files));

    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".delete-btn");
      if (!btn || state.isTesting) return;
      const i = Number(btn.dataset.index);
      URL.revokeObjectURL(state.images[i].url);
      state.images.splice(i, 1);
      render();
    });

    el.testBtn.addEventListener("click", async () => {
      el.error.textContent = "";
      if (state.images.length === 0) {
        el.error.textContent = "Choose at least one image.";
        return;
      }
      state.isTesting = true;
      render();
      const fd = new FormData();
      state.images.forEach((img) => fd.append("files", img.file));
      try {
        const res = await fetch("/predict", { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          el.error.textContent = data.error || "Testing failed.";
          return;
        }
        state.images = state.images.map((img) => {
          const found = data.results ? data.results.find((x) => x.filename === img.file.name) : null;
          return { ...img, prediction: found ? found.prediction : "Error" };
        });
        render();
      } catch (err) {
        el.error.textContent = "Could not reach the local server.";
      } finally {
        state.isTesting = false;
        render();
      }
    });

    render();
  </script>
</body>
</html>
"""
