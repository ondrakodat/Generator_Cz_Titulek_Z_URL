const form = document.querySelector("#batchForm");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const resetButton = document.querySelector("#resetButton");
const statusBox = document.querySelector("#status");
const stageText = document.querySelector("#stageText");
const overallProgressText = document.querySelector("#overallProgressText");
const overallProgressFill = document.querySelector("#overallProgressFill");
const currentProgressText = document.querySelector("#currentProgressText");
const currentProgressFill = document.querySelector("#currentProgressFill");
const humanLog = document.querySelector("#humanLog");
const technicalLog = document.querySelector("#technicalLog");
const queueList = document.querySelector("#queueList");
const urlsInput = document.querySelector("#urlsInput");

let pollTimer = null;

function clampProgress(value) {
  return Math.max(0, Math.min(100, Number(value || 0)));
}

function renderHumanLog(items) {
  humanLog.innerHTML = "";
  for (const item of items || []) {
    const li = document.createElement("li");
    li.textContent = item;
    humanLog.appendChild(li);
  }
}

function statusLabel(job) {
  if (job.status === "queued") return "Ceka";
  if (job.status === "running") return job.stage || "Pracuji";
  if (job.status === "done") return "Hotovo";
  if (job.status === "translation_failed") return "Preklad selhal";
  if (job.status === "error") return "Chyba";
  if (job.status === "stopped") return "Zastaveno";
  return job.stage || job.status;
}

function downloadLink(job, filename, text, className, enabled) {
  const link = document.createElement("a");
  link.className = className;
  link.href = `/batch/download/${encodeURIComponent(job.id)}/${encodeURIComponent(filename)}`;
  link.textContent = text;
  link.setAttribute("aria-disabled", enabled ? "false" : "true");
  return link;
}

function metricsText(metrics) {
  if (!metrics) return "";
  const crop = metrics.used_crop
    ? `crop ${Number(metrics.used_crop.x1).toFixed(2)}-${Number(metrics.used_crop.x2).toFixed(2)} / ${Number(metrics.used_crop.y1).toFixed(2)}-${Number(metrics.used_crop.y2).toFixed(2)}`
    : "crop ?";
  return [
    crop,
    `frames ${metrics.total_frames_sampled || 0}`,
    `with subtitles ${metrics.frames_with_subtitles || 0}`,
    `skipped empty ${metrics.skipped_without_subtitles || 0}`,
    `skipped same ${metrics.skipped_same_subtitle || 0}`,
    `OCR calls ${metrics.actual_ocr_calls || 0}`,
    `original blocks ${metrics.subtitle_blocks || 0}`,
    `translated blocks ${metrics.translated_blocks || 0}`,
    `translation ${metrics.translation_status || "pending"}`,
  ].join(" | ");
}

function renderQueue(jobs) {
  queueList.innerHTML = "";
  if (!jobs || jobs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.textContent = "Zatim neni spustena zadna davka.";
    queueList.appendChild(empty);
    return;
  }

  for (const job of jobs) {
    const item = document.createElement("article");
    item.className = "queue-item";

    const head = document.createElement("div");
    head.className = "queue-head";

    const titleWrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "queue-title";
    title.textContent = job.id.replace("_", " ").toUpperCase();
    const url = document.createElement("div");
    url.className = "queue-url";
    url.textContent = job.url;
    titleWrap.append(title, url);

    const badge = document.createElement("span");
    badge.className = `queue-status ${job.status === "done" ? "done" : job.status === "error" || job.status === "translation_failed" ? "error" : ""}`;
    badge.textContent = statusLabel(job);

    head.append(titleWrap, badge);

    const stage = document.createElement("div");
    stage.className = "queue-stage";
    stage.textContent = job.error || `${job.stage || "Ceka"} (${clampProgress(job.progress)} %)`;

    const metrics = document.createElement("div");
    metrics.className = "queue-stage";
    metrics.textContent = metricsText(job.metrics);

    const actions = document.createElement("div");
    actions.className = "queue-actions";
    actions.append(
      downloadLink(job, "video_cz.mp4", "Video", "download-main", Boolean(job.outputs && job.outputs.video)),
      downloadLink(job, "subtitles_original_raw.srt", "Original raw SRT", "download-secondary", Boolean(job.outputs && job.outputs.srt_raw)),
      downloadLink(job, "subtitles_original_context_fixed.srt", "Original context fixed SRT", "download-secondary", Boolean(job.outputs && job.outputs.srt_context)),
      downloadLink(job, "subtitles_cs.srt", "Czech SRT", "download-secondary", Boolean(job.outputs && job.outputs.srt_cs)),
      downloadLink(job, "ocr_report.json", "OCR report", "download-secondary", Boolean(job.outputs && job.outputs.ocr_report)),
      downloadLink(job, "translation_report.json", "Translation report", "download-secondary", Boolean(job.outputs && job.outputs.translation_report)),
      downloadLink(job, "log.txt", "Log", "download-secondary", Boolean(job.outputs && job.outputs.log)),
    );

    item.append(head, stage, metrics, actions);
    queueList.appendChild(item);
  }
}

function renderStatus(data) {
  const overall = clampProgress(data.overall_progress);
  const runningJob = (data.jobs || []).find((job) => job.status === "running");
  const current = clampProgress(runningJob ? runningJob.progress : overall);

  overallProgressFill.style.width = `${overall}%`;
  overallProgressText.textContent = `${overall} %`;
  currentProgressFill.style.width = `${current}%`;
  currentProgressText.textContent = `${current} %`;
  stageText.textContent = data.current_stage || "Pripraveno";
  technicalLog.textContent = (data.technical_log || []).join("\n");
  technicalLog.scrollTop = technicalLog.scrollHeight;
  renderHumanLog(data.human_log || []);
  renderQueue(data.jobs || []);

  startButton.disabled = Boolean(data.running);
  stopButton.disabled = !data.running;
  statusBox.textContent = data.running ? "Pracuji" : overall >= 100 ? "Hotovo" : "Pripraveno";

  if (pollTimer && !data.running) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollStatus() {
  const response = await fetch("/batch/status");
  const data = await response.json();
  renderStatus(data);
}

async function startPolling() {
  await pollStatus();
  if (!pollTimer) {
    pollTimer = setInterval(pollStatus, 1500);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  startButton.disabled = true;
  statusBox.textContent = "Spoustim";
  humanLog.innerHTML = "";
  technicalLog.textContent = "";
  overallProgressFill.style.width = "0%";
  currentProgressFill.style.width = "0%";
  overallProgressText.textContent = "0 %";
  currentProgressText.textContent = "0 %";
  stageText.textContent = "Pripravuji...";

  const response = await fetch("/batch/start", {
    method: "POST",
    body: new FormData(form),
  });

  if (!response.ok) {
    const data = await response.json();
    renderHumanLog([data.error || "Davku se nepodarilo spustit."]);
    statusBox.textContent = "Chyba";
    startButton.disabled = false;
    return;
  }

  await startPolling();
});

stopButton.addEventListener("click", async () => {
  stopButton.disabled = true;
  await fetch("/batch/stop", { method: "POST" });
  await pollStatus();
});

resetButton.addEventListener("click", () => {
  form.reset();
  urlsInput.focus();
  overallProgressFill.style.width = "0%";
  currentProgressFill.style.width = "0%";
  overallProgressText.textContent = "0 %";
  currentProgressText.textContent = "0 %";
  stageText.textContent = "Pripraveno";
  statusBox.textContent = "Pripraveno";
  humanLog.innerHTML = "";
  technicalLog.textContent = "";
  renderQueue([]);
});

pollStatus();
