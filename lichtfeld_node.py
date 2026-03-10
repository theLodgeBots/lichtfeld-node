#!/usr/bin/env python3
"""
lichtfeld-node: Windows GPU splat processing node using LichtFeld Studio.

Polls Firestore for splat_status=300 tickets, downloads dataset,
runs COLMAP + LichtFeld Studio training (with real-time viewer),
aligns to ARKit, uploads results.

Runs natively on Windows with an NVIDIA GPU — no Docker needed.

Usage:
    python lichtfeld_node.py --lfs "C:/LichtFeld-Studio/bin/LichtFeld-Studio.exe"

Requirements:
    - Windows 10/11 with NVIDIA GPU (RTX 2060+)
    - NVIDIA driver 570+
    - COLMAP installed and on PATH
    - LichtFeld Studio binary (download from GitHub releases)
    - Python 3.10+ with deps: pip install -r requirements.txt
    - Firebase service account JSON
"""

import argparse
import datetime
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter


# ─── Status Codes ────────────────────────────────────────────────────────────

class SplatStatus:
    NOT_REQUESTED = 0
    REQUESTED = 300
    ASSIGNED = 400
    PROCESSING = 410
    SUCCESS = 500
    FAILED = 950

class NodeStatus:
    ERROR = -1
    UNKNOWN = 0
    AVAILABLE = 1
    PROCESSING = 2


# ─── Dashboard ───────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lichtfeld-node</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
         background: #0a0a0a; color: #e5e5e5; padding: 24px; }
  .header { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .status-dot { width: 12px; height: 12px; border-radius: 50%; }
  .status-dot.available { background: #22c55e; box-shadow: 0 0 8px #22c55e80; }
  .status-dot.processing { background: #a855f7; box-shadow: 0 0 8px #a855f780; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
  .card { background: #141414; border: 1px solid #262626; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
  .stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a1a; }
  .stat:last-child { border-bottom: none; }
  .stat .label { color: #888; font-size: 13px; }
  .stat .value { font-size: 13px; font-weight: 500; }
  .progress-bar { width: 100%; height: 6px; background: #262626; border-radius: 3px; overflow: hidden; margin-top: 8px; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #f59e0b, #ef4444); border-radius: 3px; transition: width 0.3s; }
  .log { background: #0d0d0d; border: 1px solid #1a1a1a; border-radius: 8px; padding: 12px;
         font-family: monospace; font-size: 11px; color: #888;
         max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
  .log .line.info { color: #fbbf24; }
  .log .line.success { color: #22c55e; }
  .log .line.error { color: #ef4444; }
  .log .line.progress { color: #f59e0b; }
</style>
</head>
<body>
<div class="header">
  <div class="status-dot" id="statusDot"></div>
  <h1>🔆 lichtfeld-node</h1>
  <span id="statusText" style="font-size:13px;color:#888;margin-left:auto"></span>
</div>
<div class="card">
  <h2>Node Info</h2>
  <div class="stat"><span class="label">Machine</span><span class="value" id="machine">—</span></div>
  <div class="stat"><span class="label">GPU</span><span class="value" id="gpu">—</span></div>
  <div class="stat"><span class="label">Trainer</span><span class="value" id="trainer">LichtFeld Studio</span></div>
  <div class="stat"><span class="label">Node ID</span><span class="value" id="nodeId">—</span></div>
  <div class="stat"><span class="label">Uptime</span><span class="value" id="uptime">—</span></div>
  <div class="stat"><span class="label">Jobs Completed</span><span class="value" id="jobCount">0</span></div>
</div>
<div class="card" id="currentJobCard" style="display:none">
  <h2>Current Job</h2>
  <div class="stat"><span class="label">Ticket</span><span class="value" id="currentTicket">—</span></div>
  <div class="stat"><span class="label">Strategy</span><span class="value" id="currentStrategy">—</span></div>
  <div class="stat"><span class="label">Stage</span><span class="value" id="currentStage">—</span></div>
  <div class="stat"><span class="label">Progress</span><span class="value" id="currentProgress">—</span></div>
  <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
</div>
<div class="card">
  <h2>Recent Jobs</h2>
  <div id="recentJobs"><span style="color:#666;font-size:13px">No jobs yet</span></div>
</div>
<div class="card">
  <h2>Log</h2>
  <div class="log" id="logBox"></div>
</div>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('statusDot').className = 'status-dot ' + d.state;
    document.getElementById('statusText').textContent = d.state === 'processing' ? 'Training...' : 'Awaiting tickets';
    document.getElementById('machine').textContent = d.machine_name || '—';
    document.getElementById('gpu').textContent = d.gpu || '—';
    document.getElementById('nodeId').textContent = (d.node_id || '').substring(0, 8) + '...';
    document.getElementById('uptime').textContent = d.uptime || '—';
    document.getElementById('jobCount').textContent = d.jobs_completed || 0;
    const card = document.getElementById('currentJobCard');
    if (d.current_job) {
      card.style.display = '';
      document.getElementById('currentTicket').textContent = d.current_job.ticket_id || '—';
      document.getElementById('currentStrategy').textContent = d.current_job.strategy || 'mcmc';
      document.getElementById('currentStage').textContent = d.current_job.stage || '—';
      document.getElementById('currentProgress').textContent = (d.current_job.progress||0) + '%';
      document.getElementById('progressFill').style.width = (d.current_job.progress||0) + '%';
    } else { card.style.display = 'none'; }
    const rj = document.getElementById('recentJobs');
    if (d.recent_jobs && d.recent_jobs.length) {
      rj.innerHTML = d.recent_jobs.map(j =>
        '<div style="padding:6px 0;border-bottom:1px solid #1a1a1a;font-size:13px">' +
        '<span style="font-family:monospace;color:#fbbf24">' + j.ticket_id.substring(0,8) + '...</span> ' +
        (j.success ? '✅' : '❌') + ' ' +
        '<span style="color:#666">' + j.duration + 's</span></div>'
      ).join('');
    }
    const lr = await fetch('/api/log');
    const ld = await lr.json();
    const box = document.getElementById('logBox');
    box.innerHTML = (ld.lines || []).map(l => {
      let cls = '';
      if (l.includes('✅')) cls = 'success';
      else if (l.includes('❌') || l.includes('ERROR')) cls = 'error';
      else if (l.includes('Step') || l.includes('progress') || l.includes('%')) cls = 'progress';
      else if (l.includes('🔔') || l.includes('Processing') || l.includes('🔆')) cls = 'info';
      return '<div class="line ' + cls + '">' + l + '</div>';
    }).join('');
    box.scrollTop = box.scrollHeight;
  } catch(e) {}
}
refresh(); setInterval(refresh, 3000);
</script>
</body></html>"""


# ─── Node ────────────────────────────────────────────────────────────────────

class LichtFeldNode:
    def __init__(self, lfs_path, port=8788, data_dir=None, max_steps=30000,
                 downsample=1, strategy="mcmc", colmap_path="colmap",
                 headless=False):
        self.lfs_path = Path(lfs_path)
        if not self.lfs_path.exists():
            raise FileNotFoundError(f"LichtFeld Studio not found: {self.lfs_path}")

        self.port = port
        self.max_steps = max_steps
        self.downsample = downsample
        self.default_strategy = strategy
        self.colmap_path = colmap_path
        self.headless = headless

        # Default data dir: ./data next to script
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.machine_name = platform.node()
        self.gpu_name = self._get_gpu_name()
        self.node_id = None
        self.state = "available"
        self.current_job = None
        self.jobs_completed = 0
        self.recent_jobs = []
        self.start_time = time.time()
        self.log_lines = []
        self._lock = threading.Lock()

        # Firebase
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "./service-account.json")
        if not Path(sa_path).exists():
            raise FileNotFoundError(
                f"Firebase service account not found: {sa_path}\n"
                "Set GOOGLE_APPLICATION_CREDENTIALS or place service-account.json in working dir."
            )
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred, {
            "storageBucket": "mobilescannerphotogrammetry.firebasestorage.app",
        })
        self.db = firestore.client()
        self.bucket = storage.bucket()

        self.node_id = self._register_node()
        self._log(f"🔆 lichtfeld-node started — {self.gpu_name}")
        self._log(f"   Trainer: {self.lfs_path}")
        self._log(f"   Node: {self.node_id[:8]}...")
        self._log(f"   Steps: {self.max_steps}, Downsample: {self.downsample}x, Strategy: {self.default_strategy}")
        self._log(f"   Viewer: {'disabled' if self.headless else 'enabled (real-time)'}")

    @staticmethod
    def _get_gpu_name():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass
        return "Unknown GPU"

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[-400:]
        print(line, flush=True)

    def _register_node(self):
        node_id = str(uuid.uuid4())
        self.db.collection("nodes").document(node_id).set({
            "machine_name": self.machine_name,
            "node_type": "splat-gpu-lichtfeld",
            "gpu": self.gpu_name,
            "trainer": "LichtFeld Studio",
            "node_status": NodeStatus.AVAILABLE,
            "state": "available",
            "port": self.port,
            "heartbeat": firestore.SERVER_TIMESTAMP,
            "jobs_completed": 0,
            "date_created": firestore.SERVER_TIMESTAMP,
            "date_modified": firestore.SERVER_TIMESTAMP,
        })
        return node_id

    # ─── Firestore Listener ──────────────────────────────────────────────────

    def _on_snapshot(self, col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name in ('ADDED', 'MODIFIED'):
                data = change.document.to_dict()
                if data.get('splat_status') == SplatStatus.REQUESTED:
                    tid = change.document.id
                    self._log(f"🔔 Splat requested: {tid}")
                    threading.Thread(target=self._safe_process, args=(tid, data), daemon=True).start()

    def _safe_process(self, ticket_id, ticket_data):
        if not self._lock.acquire(blocking=False):
            self._log(f"⏳ Busy, {ticket_id} will be picked up next")
            return
        try:
            self._process(ticket_id)
        except Exception as e:
            self._log(f"ERROR {ticket_id}: {e}")
            import traceback
            self._log(traceback.format_exc())
            self._fail(ticket_id, str(e))
        finally:
            self.state = "available"
            self.current_job = None
            self.db.collection("nodes").document(self.node_id).update({
                "node_status": NodeStatus.AVAILABLE, "state": "available",
                "date_modified": firestore.SERVER_TIMESTAMP,
            })
            self._lock.release()
            self._check_pending()

    def _check_pending(self):
        try:
            pending = list(self.db.collection("tickets")
                .where(filter=FieldFilter("splat_status", "==", SplatStatus.REQUESTED))
                .limit(1).get())
            if pending:
                doc = pending[0]
                self._log(f"🔔 Pending ticket: {doc.id}")
                threading.Thread(target=self._safe_process, args=(doc.id, doc.to_dict()), daemon=True).start()
        except Exception:
            pass

    # ─── Processing Pipeline ─────────────────────────────────────────────────

    def _process(self, ticket_id):
        self.state = "processing"
        t0 = time.time()

        # Read ticket to get per-job config
        ticket_doc = self.db.collection("tickets").document(ticket_id).get()
        ticket_data = ticket_doc.to_dict() if ticket_doc.exists else {}
        strategy = ticket_data.get("splat_strategy", self.default_strategy)
        if strategy not in ("default", "mcmc"):
            strategy = self.default_strategy

        self.current_job = {"ticket_id": ticket_id, "stage": "Starting", "progress": 0, "strategy": strategy}

        # Claim ticket
        self.db.collection("tickets").document(ticket_id).update({
            "splat_status": SplatStatus.ASSIGNED,
            "splat_date_start": firestore.SERVER_TIMESTAMP,
            "splat_node_id": self.node_id,
            "splat_trainer": "lichtfeld",
            "date_modified": firestore.SERVER_TIMESTAMP,
        })
        self.db.collection("nodes").document(self.node_id).update({
            "node_status": NodeStatus.PROCESSING, "state": "processing",
            "date_modified": firestore.SERVER_TIMESTAMP,
        })

        work = self.data_dir / ticket_id.upper()
        work.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Download
            self._update(ticket_id, "Downloading", 2)
            self._download(ticket_id, work)

            # 2. Convert images
            self._update(ticket_id, "Converting images", 8)
            images_dir = self._convert_images(work)

            # 3. COLMAP
            self._update(ticket_id, "COLMAP: features", 12)
            self._run_colmap(ticket_id, work, images_dir)

            # 4. LichtFeld training
            self._update(ticket_id, f"Training LichtFeld ({strategy})", 30)
            ply_path = self._train(ticket_id, work, strategy=strategy)

            # 5. Align to ARKit
            self._update(ticket_id, "Aligning", 91)
            aligned = self._align(work, ply_path)

            # 6. Convert to .splat
            self._update(ticket_id, "Converting", 93)
            splat_path = self._to_splat(aligned, work / "splat.splat")

            # 7. Upload
            self._update(ticket_id, "Uploading", 96)
            url = self._upload(ticket_id, splat_path, aligned)

            # Done
            elapsed = round(time.time() - t0)
            self.db.collection("tickets").document(ticket_id).update({
                "splat_status": SplatStatus.SUCCESS,
                "splat_progress": 100,
                "splat_strategy": strategy,
                "splat_trainer": "lichtfeld",
                "url_splat": url,
                "splat_date_end": firestore.SERVER_TIMESTAMP,
                "splat_duration": elapsed,
                "date_modified": firestore.SERVER_TIMESTAMP,
            })
            self._log(f"✅ {ticket_id} complete! ({elapsed}s)")
            self.jobs_completed += 1
            self.recent_jobs.insert(0, {"ticket_id": ticket_id, "success": True, "duration": elapsed})

        except Exception as e:
            elapsed = round(time.time() - t0)
            self.recent_jobs.insert(0, {"ticket_id": ticket_id, "success": False, "duration": elapsed})
            raise
        finally:
            # Cleanup
            if work.exists():
                shutil.rmtree(str(work), ignore_errors=True)
            if len(self.recent_jobs) > 20:
                self.recent_jobs = self.recent_jobs[:20]

    def _update(self, ticket_id, stage, progress):
        if self.current_job:
            self.current_job["stage"] = stage
            self.current_job["progress"] = progress
        self._log(f"  [{progress}%] {stage}")
        try:
            self.db.collection("tickets").document(ticket_id).update({
                "splat_status": SplatStatus.PROCESSING,
                "splat_progress": progress, "splat_stage": stage,
                "date_modified": firestore.SERVER_TIMESTAMP,
            })
        except Exception:
            pass

    def _fail(self, ticket_id, error):
        self._log(f"❌ {ticket_id}: {error}")
        try:
            self.db.collection("tickets").document(ticket_id).update({
                "splat_status": SplatStatus.FAILED,
                "splat_error": error,
                "splat_trainer": "lichtfeld",
                "splat_date_end": firestore.SERVER_TIMESTAMP,
                "date_modified": firestore.SERVER_TIMESTAMP,
            })
        except Exception:
            pass

    # ─── Download ────────────────────────────────────────────────────────────

    def _download(self, ticket_id, work):
        tid = ticket_id.upper()
        blob = self.bucket.blob(f"captures/{tid}/{tid}_data.zip")
        if not blob.exists():
            raise RuntimeError("Dataset not found in Storage")

        zip_path = work / "data.zip"
        blob.download_to_filename(str(zip_path))
        size_mb = zip_path.stat().st_size / 1024 / 1024
        self._log(f"  Downloaded {size_mb:.1f} MB")

        with zipfile.ZipFile(str(zip_path)) as zf:
            for member in zf.namelist():
                fn = os.path.basename(member)
                if not fn or "__MACOSX" in member:
                    continue
                with zf.open(member) as src, open(str(work / fn), "wb") as dst:
                    shutil.copyfileobj(src, dst)
        zip_path.unlink()

    # ─── Convert Images ──────────────────────────────────────────────────────

    def _convert_images(self, work):
        images_dir = work / "images"
        images_dir.mkdir(exist_ok=True)

        # Copy JPGs
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
            for f in work.glob(ext):
                if f.parent == work:
                    shutil.copy2(str(f), str(images_dir / f.name))

        # Convert HEIC
        heics = sorted(list(work.glob("*.heic")) + list(work.glob("*.HEIC")))
        if heics:
            try:
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                for i, heic in enumerate(heics):
                    jpg_out = images_dir / heic.with_suffix(".jpg").name
                    if not jpg_out.exists():
                        img = Image.open(str(heic))
                        img.save(str(jpg_out), "JPEG", quality=95)
                    if (i + 1) % 20 == 0:
                        self._log(f"  Converted {i+1}/{len(heics)} HEIC")
            except ImportError:
                self._log("  WARNING: pillow-heif not installed, skipping HEIC conversion")

        total = sum(1 for _ in images_dir.iterdir() if _.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        self._log(f"  {total} images ready")
        if total == 0:
            raise RuntimeError("No images found in dataset")
        return images_dir

    # ─── COLMAP ──────────────────────────────────────────────────────────────

    def _run_colmap(self, ticket_id, work, images_dir):
        db_path = work / "database.db"
        sparse_dir = work / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        t0 = time.time()

        # Feature extraction (GPU)
        self._update(ticket_id, "COLMAP: features", 12)
        r = subprocess.run([
            self.colmap_path, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
            "--FeatureExtraction.use_gpu", "1",
            "--SiftExtraction.max_image_size", "3200",
            "--SiftExtraction.max_num_features", "16384",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        self._log(f"  COLMAP feature_extractor exit code: {r.returncode}")
        if r.returncode != 0:
            raise RuntimeError(f"Feature extraction failed (exit {r.returncode}): {r.stdout[-500:]}")
        # Log last few lines of output
        for line in r.stdout.strip().split("\n")[-3:]:
            self._log(f"  {line.strip()}")

        # Matching (GPU)
        self._update(ticket_id, "COLMAP: matching", 16)
        r = subprocess.run([
            self.colmap_path, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.guided_matching", "1",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        self._log(f"  COLMAP exhaustive_matcher exit code: {r.returncode}")
        if r.returncode != 0:
            raise RuntimeError(f"Matching failed (exit {r.returncode}): {r.stdout[-500:]}")

        # Mapper
        self._update(ticket_id, "COLMAP: mapping", 22)
        mapper_out = sparse_dir / "mapper_out"
        mapper_out.mkdir(exist_ok=True)
        r = subprocess.run([
            self.colmap_path, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(mapper_out),
            "--Mapper.ba_global_max_num_iterations", "100",
            "--Mapper.ba_global_max_refinements", "5",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        self._log(f"  COLMAP mapper exit code: {r.returncode}")
        if r.returncode != 0:
            raise RuntimeError(f"Mapper failed (exit {r.returncode}): {r.stdout[-500:]}")

        # Pick best model
        model_0 = sparse_dir / "0"
        model_0.mkdir(exist_ok=True)
        best, best_count = None, 0
        for d in mapper_out.iterdir():
            if d.is_dir() and (d / "images.bin").exists():
                n = struct.unpack("<Q", open(d / "images.bin", "rb").read(8))[0]
                if n > best_count:
                    best, best_count = d, n
        if not best:
            raise RuntimeError("COLMAP produced no models")

        for f in best.iterdir():
            if f.name not in ("frames.bin", "rigs.bin"):
                shutil.copy2(str(f), str(model_0 / f.name))
        shutil.rmtree(str(mapper_out))
        self._log(f"  COLMAP: {best_count} images registered")

        # Undistort
        self._update(ticket_id, "COLMAP: undistort", 26)
        undist = work / "undistorted"
        r = subprocess.run([
            self.colmap_path, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(model_0),
            "--output_path", str(undist),
            "--output_type", "COLMAP",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        if r.returncode == 0 and (undist / "images").exists():
            shutil.rmtree(str(images_dir))
            (undist / "images").rename(images_dir)
            if (undist / "sparse").exists():
                shutil.rmtree(str(model_0))
                model_0.mkdir()
                for f in (undist / "sparse").iterdir():
                    shutil.copy2(str(f), str(model_0 / f.name))
            shutil.rmtree(str(undist), ignore_errors=True)
            self._log("  Undistorted images ready")
        else:
            self._log("  Undistortion skipped (non-critical)")

        elapsed = round(time.time() - t0)
        self._log(f"  COLMAP complete in {elapsed}s")

    # ─── LichtFeld Training ─────────────────────────────────────────────────

    def _train(self, ticket_id, work, strategy="mcmc"):
        result_dir = work / "lfs_output"
        result_dir.mkdir(exist_ok=True)

        self._log(f"  Training LichtFeld ({strategy}): {self.max_steps} steps, {self.downsample}x downsample")
        t0 = time.time()

        cmd = [
            str(self.lfs_path),
            "-d", str(work),
            "-o", str(result_dir),
            "-i", str(self.max_steps),
            "--strategy", strategy,
            "--bilateral-grid",
            "--enable-mip",
            "--train",
            "--no-splash",
        ]

        if self.downsample > 1:
            cmd.extend(["-r", str(self.downsample)])

        if strategy == "mcmc":
            cmd.extend(["--max-cap", "2000000"])

        if self.headless:
            cmd.append("--headless")

        self._log(f"  CMD: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        import re
        last_pct = 30
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Parse iteration progress from LichtFeld output
            # LFS logs lines like: "Iteration 5000/30000 ..."
            m = re.search(r'[Ii]teration\s+(\d+)\s*/\s*(\d+)', line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                pct_raw = int(100 * step / total) if total > 0 else 0
                pct = min(90, 30 + int(60 * pct_raw / 100))
                if pct > last_pct + 3:
                    last_pct = pct
                    self._update(ticket_id, f"Training ({pct_raw}%)", pct)
                if step > 0 and step % 5000 == 0:
                    self._log(f"  Step {step}/{total}")

            # Also try percentage pattern
            m2 = re.search(r'(\d+)%', line)
            if m2 and not m:
                pct_raw = int(m2.group(1))
                pct = min(90, 30 + int(60 * pct_raw / 100))
                if pct > last_pct + 3:
                    last_pct = pct
                    self._update(ticket_id, f"Training ({pct_raw}%)", pct)

            # Log key lines
            if any(k in line.lower() for k in ["scene scale", "number of", "gaussians",
                                                  "psnr", "error", "traceback", "saved",
                                                  "loss", "training complete", "writing"]):
                self._log(f"  {line[:200]}")

        proc.wait(timeout=7200)
        elapsed = round(time.time() - t0)
        self._log(f"  LichtFeld training: {elapsed}s (exit {proc.returncode})")

        if proc.returncode != 0:
            raise RuntimeError(f"LichtFeld training failed (exit {proc.returncode})")

        # Find PLY output
        # LichtFeld saves to output_dir/point_cloud/iteration_XXXXX/point_cloud.ply
        # or directly as .ply files in the output directory
        ply = self._find_ply(result_dir)
        if not ply:
            raise RuntimeError("No PLY output from LichtFeld training")

        self._log(f"  PLY: {ply.stat().st_size / 1024 / 1024:.1f} MB")
        return ply

    def _find_ply(self, result_dir):
        """Find the best PLY file from LichtFeld output."""
        # Check common output locations
        candidates = []

        # Recursive search for .ply files
        for ply in result_dir.rglob("*.ply"):
            candidates.append(ply)

        if not candidates:
            return None

        # Return the largest/newest PLY
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # ─── Alignment ───────────────────────────────────────────────────────────

    def _align(self, work, ply_path):
        """Align from COLMAP space to ARKit space via Procrustes."""
        frames = sorted(work.glob("frame_*.json"))
        if not frames:
            self._log("  No ARKit frames — skipping alignment")
            return ply_path

        images_bin = work / "sparse" / "0" / "images.bin"
        if not images_bin.exists():
            self._log("  No COLMAP model — skipping alignment")
            return ply_path

        # Read COLMAP cameras
        colmap_cams = {}
        with open(images_bin, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                struct.unpack("<I", f.read(4))
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                struct.unpack("<I", f.read(4))
                name = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name += c
                name = name.decode()
                np2d = struct.unpack("<Q", f.read(8))[0]
                f.read(np2d * 24)
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                colmap_cams[name] = -R.T @ t

        # Read ARKit cameras
        arkit_cams = {}
        for fp in frames:
            with open(fp) as fj:
                data = json.load(fj)
            if "cameraPoseARFrame" not in data:
                continue
            c2w = np.array(data["cameraPoseARFrame"]).reshape(4, 4, order="C")
            idx = int(fp.stem.split("_")[1])
            img_name = f"IMG_{idx:04d}.jpg"
            frame_name = f"frame_{idx:05d}.jpg"
            if frame_name in colmap_cams and img_name not in colmap_cams:
                arkit_cams[frame_name] = c2w[:3, 3]
            else:
                arkit_cams[img_name] = c2w[:3, 3]

        common = sorted(set(colmap_cams) & set(arkit_cams))
        self._log(f"  Alignment: {len(common)} matched cameras")
        if len(common) < 4:
            self._log("  Too few matches — skipping alignment")
            return ply_path

        src = np.array([colmap_cams[k] for k in common])
        dst = np.array([arkit_cams[k] for k in common])

        # Procrustes
        mu_s, mu_d = src.mean(0), dst.mean(0)
        sc, dc = src - mu_s, dst - mu_d
        ss = np.sqrt((sc ** 2).sum() / len(src))
        sd = np.sqrt((dc ** 2).sum() / len(dst))
        s = sd / ss
        H = (sc / ss).T @ (dc / sd)
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        t_vec = mu_d - s * R @ mu_s

        # Verify
        transformed = s * (src @ R.T) + t_vec
        errors = np.linalg.norm(transformed - dst, axis=1)

        # Refine with inliers
        inlier = errors < np.percentile(errors, 90)
        if inlier.sum() >= 4:
            mu_s, mu_d = src[inlier].mean(0), dst[inlier].mean(0)
            sc, dc = src[inlier] - mu_s, dst[inlier] - mu_d
            ss = np.sqrt((sc ** 2).sum() / inlier.sum())
            sd = np.sqrt((dc ** 2).sum() / inlier.sum())
            s = sd / ss
            H = (sc / ss).T @ (dc / sd)
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R = Vt.T @ np.diag([1, 1, d]) @ U.T
            t_vec = mu_d - s * R @ mu_s
            transformed = s * (src @ R.T) + t_vec
            errors = np.linalg.norm(transformed - dst, axis=1)

        self._log(f"  Scale: {s:.4f}, Mean error: {errors.mean():.4f}")

        # Apply to PLY
        aligned_path = work / "aligned.ply"
        self._transform_ply(ply_path, aligned_path, s, R, t_vec)
        return aligned_path

    def _transform_ply(self, in_path, out_path, s, R, t):
        """Apply rigid transform to gaussian splat PLY."""
        with open(in_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii")
                header_lines.append(line)
                if line.strip() == "end_header":
                    break
            header = "".join(header_lines)
            data = f.read()

        properties = []
        vertex_count = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        names = [n for _, n in properties]
        fmt_map = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I"}
        fmt = "<" + "".join(fmt_map.get(dt, "f") for dt, _ in properties)
        vsize = struct.calcsize(fmt)

        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")

        # Rotation and scale indices (may not exist in all PLY formats)
        has_rot = all(f"rot_{i}" in names for i in range(4))
        has_scale = all(f"scale_{i}" in names for i in range(3))
        irot = [names.index(f"rot_{i}") for i in range(4)] if has_rot else None
        iscale = [names.index(f"scale_{i}") for i in range(3)] if has_scale else None
        log_s = np.log(s)

        out = bytearray()
        for i in range(vertex_count):
            vals = list(struct.unpack(fmt, data[i*vsize:(i+1)*vsize]))

            # Position
            pos = np.array([vals[ix], vals[iy], vals[iz]])
            new_pos = s * R @ pos + t
            vals[ix], vals[iy], vals[iz] = new_pos

            # Rotation
            if irot:
                q = np.array([vals[irot[1]], vals[irot[2]], vals[irot[3]], vals[irot[0]]])  # xyzw
                q_new = (Rotation.from_matrix(R) * Rotation.from_quat(q)).as_quat()
                vals[irot[0]] = q_new[3]  # w
                vals[irot[1]] = q_new[0]
                vals[irot[2]] = q_new[1]
                vals[irot[3]] = q_new[2]

            # Scale
            if iscale:
                for si in iscale:
                    vals[si] += log_s

            out.extend(struct.pack(fmt, *vals))

        with open(out_path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(out)

    # ─── Export .splat ───────────────────────────────────────────────────────

    def _to_splat(self, ply_path, out_path):
        """Convert PLY to .splat format."""
        with open(ply_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii")
                header_lines.append(line)
                if line.strip() == "end_header":
                    break
            data = f.read()

        properties = []
        vertex_count = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        names = [n for _, n in properties]
        fmt_map = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I"}
        fmt = "<" + "".join(fmt_map.get(dt, "f") for dt, _ in properties)
        vsize = struct.calcsize(fmt)

        def idx(n): return names.index(n) if n in names else -1

        ix, iy, iz = idx("x"), idx("y"), idx("z")
        iscale = [idx(f"scale_{i}") for i in range(3)]
        irot = [idx(f"rot_{i}") for i in range(4)]
        iopacity = idx("opacity")
        ish = [idx(f"f_dc_{i}") for i in range(3)]
        SH_C0 = 0.28209479177387814

        out = bytearray()
        for i in range(vertex_count):
            v = struct.unpack(fmt, data[i*vsize:(i+1)*vsize])

            x, y, z = v[ix], v[iy], v[iz]
            sx = math.exp(v[iscale[0]]) if iscale[0] >= 0 else 0.01
            sy = math.exp(v[iscale[1]]) if iscale[1] >= 0 else 0.01
            sz = math.exp(v[iscale[2]]) if iscale[2] >= 0 else 0.01

            r = max(0, min(255, int((0.5 + SH_C0 * v[ish[0]]) * 255))) if ish[0] >= 0 else 128
            g = max(0, min(255, int((0.5 + SH_C0 * v[ish[1]]) * 255))) if ish[1] >= 0 else 128
            b = max(0, min(255, int((0.5 + SH_C0 * v[ish[2]]) * 255))) if ish[2] >= 0 else 128

            raw_op = v[iopacity] if iopacity >= 0 else 0
            a = max(0, min(255, int(1.0 / (1.0 + math.exp(-raw_op)) * 255)))

            qw = v[irot[0]] if irot[0] >= 0 else 1
            qx = v[irot[1]] if irot[1] >= 0 else 0
            qy = v[irot[2]] if irot[2] >= 0 else 0
            qz = v[irot[3]] if irot[3] >= 0 else 0
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if norm > 0:
                qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

            out.extend(struct.pack("<3f3f4B4B",
                x, y, z, sx, sy, sz,
                r, g, b, a,
                int((qw*0.5+0.5)*255), int((qx*0.5+0.5)*255),
                int((qy*0.5+0.5)*255), int((qz*0.5+0.5)*255),
            ))

        with open(out_path, "wb") as f:
            f.write(out)

        self._log(f"  .splat: {len(out) / 1024 / 1024:.1f} MB ({vertex_count} gaussians)")
        return out_path

    # ─── Upload ──────────────────────────────────────────────────────────────

    def _upload(self, ticket_id, splat_path, ply_path):
        tid = ticket_id.upper()
        zip_name = f"{tid}_splat.zip"
        zip_path = splat_path.parent / zip_name

        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(splat_path), "splat.splat")
            if ply_path.exists():
                final_ply = splat_path.parent / "splat.ply"
                if ply_path != final_ply:
                    shutil.copy2(str(ply_path), str(final_ply))
                zf.write(str(final_ply), "splat.ply")

        blob_name = f"captures/{tid}/{zip_name}"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(zip_path))
        blob.make_public()
        url = f"https://storage.googleapis.com/{self.bucket.name}/{blob_name}"
        self._log(f"  Uploaded: {url}")
        return url

    # ─── Heartbeat ───────────────────────────────────────────────────────────

    def _heartbeat(self):
        try:
            up = int(time.time() - self.start_time)
            h, rem = divmod(up, 3600)
            m, s = divmod(rem, 60)
            self.db.collection("nodes").document(self.node_id).update({
                "heartbeat": firestore.SERVER_TIMESTAMP,
                "uptime": f"{h}h {m}m" if h else f"{m}m {s}s",
                "state": self.state,
                "jobs_completed": self.jobs_completed,
                "current_ticket": self.current_job.get("ticket_id") if self.current_job else None,
                "current_stage": self.current_job.get("stage") if self.current_job else None,
                "current_progress": self.current_job.get("progress", 0) if self.current_job else 0,
                "date_modified": firestore.SERVER_TIMESTAMP,
            })
        except Exception:
            pass

    # ─── Dashboard Server ────────────────────────────────────────────────────

    def _serve_dashboard(self):
        node = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(DASHBOARD_HTML.encode())
                elif self.path == '/api/status':
                    up = int(time.time() - node.start_time)
                    h, rem = divmod(up, 3600)
                    m, s = divmod(rem, 60)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "state": node.state,
                        "machine_name": node.machine_name,
                        "gpu": node.gpu_name,
                        "node_id": node.node_id,
                        "uptime": f"{h}h {m}m" if h else f"{m}m {s}s",
                        "jobs_completed": node.jobs_completed,
                        "current_job": node.current_job,
                        "recent_jobs": node.recent_jobs[:10],
                    }).encode())
                elif self.path == '/api/log':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"lines": node.log_lines[-100:]}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        class S(HTTPServer):
            allow_reuse_address = True
        S(('0.0.0.0', self.port), H).serve_forever()

    # ─── Run ─────────────────────────────────────────────────────────────────

    def run(self):
        threading.Thread(target=self._serve_dashboard, daemon=True).start()
        self._log(f"📊 Dashboard at http://localhost:{self.port}")

        query = self.db.collection("tickets").where(
            filter=FieldFilter("splat_status", "==", SplatStatus.REQUESTED))
        query.on_snapshot(self._on_snapshot)
        self._log("🚀 Listening for splat requests...")

        try:
            while True:
                time.sleep(30)
                self._heartbeat()
        except KeyboardInterrupt:
            self._log("Shutting down...")
            self.db.collection("nodes").document(self.node_id).update({
                "node_status": NodeStatus.UNKNOWN, "state": "offline",
                "date_modified": firestore.SERVER_TIMESTAMP,
            })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="lichtfeld-node: GPU splat processing node using LichtFeld Studio")
    p.add_argument("--lfs", required=True, help="Path to LichtFeld-Studio executable")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--data-dir", default=None, help="Working directory (default: ./data)")
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--strategy", choices=["default", "mcmc"], default="mcmc",
                   help="Training strategy (default: mcmc)")
    p.add_argument("--colmap", default="colmap", help="Path to COLMAP executable")
    p.add_argument("--headless", action="store_true",
                   help="Run LichtFeld without viewer (headless mode)")
    args = p.parse_args()

    LichtFeldNode(
        lfs_path=args.lfs,
        port=args.port,
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        downsample=args.downsample,
        strategy=args.strategy,
        colmap_path=args.colmap,
        headless=args.headless,
    ).run()
