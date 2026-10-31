#!/usr/bin/env python
"""Local demo page: one SP-10 test clip per sign language, each with a Run
button that runs the real model on that video.

    python scripts/demo_app.py            # then open http://localhost:8765

Clips are listed in scripts/demo_clips.json (test set, never seen in training)
and downloaded from SpreadTheSign into demo/videos/ on first start.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)  # configs use repo-relative paths

import torch  # noqa: E402

from translate_video import load, translate  # noqa: E402

CLIPS = json.load(open(os.path.join(ROOT, "scripts", "demo_clips.json"), encoding="utf-8"))
VIDEO_DIR = os.path.join(ROOT, "demo", "videos")


def video_path(clip):
    return os.path.join(VIDEO_DIR, clip["id"] + ".mp4")


def make_handler(models):
    lock = threading.Lock()  # ponytail: one inference at a time; fine for a single-presenter demo
    by_id = {c["id"]: c for c in CLIPS}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/clips":
                self._send(200, json.dumps(CLIPS, ensure_ascii=False).encode("utf-8"), "application/json")
            elif self.path.startswith("/video/") and self.path[7:-4] in by_id:
                with open(video_path(by_id[self.path[7:-4]]), "rb") as f:
                    self._send(200, f.read(), "video/mp4")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            clip = by_id.get(self.path[5:]) if self.path.startswith("/run/") else None
            if clip is None:
                return self._send(404, b"not found", "text/plain")
            t0 = time.time()
            with lock:
                result = translate(models, video_path(clip))
            result["seconds"] = round(time.time() - t0, 1)
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json")

        def log_message(self, fmt, *args):
            if "/run/" in self.path:
                sys.stderr.write("run %s\n" % self.path[5:])

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/proposed_lid_selfcond.yaml")
    p.add_argument("--model-dir", default="demo/ckpt/proposed/seed3")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "best.ckpt")):
        sys.exit(f"No checkpoint in {args.model_dir}. Download mlslt_checkpoints.zip from the v1.0 release "
                 "and unzip proposed/seed3 into demo/ckpt/ (README §6).")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for c in CLIPS:
        if not os.path.exists(video_path(c)):
            print("downloading", c["id"])
            urllib.request.urlretrieve(c["video_url"], video_path(c))

    print("loading model ...")
    models = load(args.config, args.model_dir, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(models))
    print(f"demo ready: http://localhost:{args.port}")
    server.serve_forever()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign Translation Demo</title>
<style>
  :root { --bg:#f6f5f2; --card:#ffffff; --ink:#1c1c1a; --muted:#6b6a65; --line:#e3e1db;
          --accent:#2d5be3; --good:#1f7a4d; --warn:#a86512; --chip:#efeee9; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151514; --card:#1f1f1d; --ink:#ecebe6; --muted:#a09f99; --line:#33332f;
            --accent:#7d9bff; --good:#5cc592; --warn:#e2a556; --chip:#2a2a27; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { max-width:1200px; margin:0 auto; padding:32px 16px 8px; }
  h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
  header p { margin:4px 0; color:var(--muted); max-width:820px; }
  .bar { display:flex; gap:10px; align-items:center; margin-top:14px; }
  main { max-width:1200px; margin:0 auto; padding:16px; display:grid; gap:16px;
         grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden;
          display:flex; flex-direction:column; }
  video { width:100%; aspect-ratio:4/3; background:#000; display:block; object-fit:contain; }
  .body { padding:12px 14px 14px; display:flex; flex-direction:column; gap:10px; flex:1; }
  .top { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
  .lang { font-weight:600; }
  .code { font:12px ui-monospace, monospace; color:var(--muted); background:var(--chip);
          padding:2px 6px; border-radius:6px; }
  .label { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:2px; }
  .text { font-size:16px; }
  .out { min-height:44px; }
  .out .text { font-weight:600; }
  .meta { font-size:12px; color:var(--muted); margin-top:2px; }
  .ok { color:var(--good); } .no { color:var(--warn); }
  .section { grid-column:1/-1; font-size:17px; margin:18px 0 0; padding-top:18px; border-top:1px solid var(--line); }
  button { font:inherit; font-weight:600; border:0; border-radius:8px; padding:9px 14px; cursor:pointer;
           background:var(--accent); color:#fff; margin-top:auto; }
  button.ghost { background:transparent; color:var(--accent); border:1px solid var(--accent); }
  button:disabled { opacity:.55; cursor:progress; }
  button:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
</style>
</head>
<body>
<header>
  <h1>Sign language → English</h1>
  <p>One clip from SpreadTheSign (SP-10 dev or test set) for each of the 10 sign languages the model was trained on.
     None of these sentences appear in training. <b>Run</b> sends the video through the real model
     (DINOv2 features → Sign2(LID+Text) with LID self-conditioning) on this computer.</p>
  <p>In the top 8 the model gets the meaning right, though usually in slightly different words: it can
     only answer with sentences it saw in training. In the bottom 2 (Russian, Bulgarian) it gets one key word wrong.</p>
  <div class="bar"><button class="ghost" id="all">Run all</button><span id="status" class="meta"></span></div>
</header>
<main id="grid"></main>
<script>
const NAMES = {csl:"Chinese",ukl:"Ukrainian",rsl:"Russian",bqn:"Bulgarian",icl:"Icelandic",
               gsg:"German",ise:"Italian",swl:"Swedish",lls:"Lithuanian",bfi:"British"};
const grid = document.getElementById("grid");
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

async function run(clip, btn, out) {
  btn.disabled = true; btn.textContent = "Running…";
  out.innerHTML = '<div class="meta">Extracting features and decoding…</div>';
  try {
    const r = await (await fetch("/run/" + clip.id, {method: "POST"})).json();
    const langOk = r.sign_language === clip.sign_language;
    const guess = r.sign_language ? NAMES[r.sign_language] + " (" + r.sign_language + ")" : "none";
    out.innerHTML = '<div class="label">Model output</div><div class="text">' + esc(r.translation || "(empty)") +
      '</div><div class="meta">Sign language guess: <span class="' + (langOk ? "ok" : "no") + '">' + guess +
      '</span> · ' + r.frames + ' frames · ' + r.seconds + ' s</div>';
  } catch (e) {
    out.innerHTML = '<div class="meta no">Run failed: is scripts/demo_app.py still running?</div>';
  }
  btn.disabled = false; btn.textContent = "Run again";
}

fetch("/clips").then(r => r.json()).then(clips => {
  const runners = [];
  let shownPartial = false;
  for (const c of clips) {
    if (c.kind === "partial" && !shownPartial) {
      shownPartial = true;
      const h = document.createElement("h2");
      h.className = "section";
      h.textContent = "Partial results: one key word wrong";
      grid.appendChild(h);
    }
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML =
      '<video src="/video/' + c.id + '.mp4" controls muted loop playsinline preload="auto"></video>' +
      '<div class="body"><div class="top"><span class="lang">' + esc(c.name) + '</span><span class="code">' + c.sign_language +
      '</span></div><div><div class="label">Expected</div><div class="text">' + esc(c.expected) + '</div></div>' +
      '<div class="out" aria-live="polite"></div><button>Run</button></div>';
    const btn = card.querySelector("button"), out = card.querySelector(".out");
    btn.setAttribute("aria-label", "Run the model on the " + c.name + " clip");
    btn.onclick = () => { card.querySelector("video").play(); run(c, btn, out); };
    runners.push(() => run(c, btn, out));
    grid.appendChild(card);
  }
  document.getElementById("all").onclick = async (e) => {
    e.target.disabled = true;
    for (let i = 0; i < runners.length; i++) {
      document.getElementById("status").textContent = "Running " + (i + 1) + " of " + runners.length + "…";
      await runners[i]();
    }
    document.getElementById("status").textContent = "Done.";
    e.target.disabled = false;
  };
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
