"""
MANIFOLD LENS  (Stable AI Flow v25)
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.

The diagnosis (Antti's, formalized)
-----------------------------------
Two manifolds: the real world, and the diffusion weights. v23/v24 tried to
stabilize a loop that is broken by construction, in two independent ways:

  1. CARTOON DRIFT. Feeding the dream back into img2img iterates the map
     x -> Diffuse(x). Iterated img2img is a contraction toward the manifold's
     attractors (smooth, saturated, illustration-like modes). More phase-lock
     gravity = more iterations of the contraction = more cartoon. The lock
     CAUSED the drift it was meant to prevent.

  2. IDENTITY FLICKER. Each frame is an independent stochastic query into a
     diffuse REGION of the manifold ("clint eastwood" is a region, not a
     point). Fresh noise every frame = a different sample from that region
     every frame. No amount of pixel-space locking fixes a sampling problem.

The fix (the unitvae4 instinct, completed)
------------------------------------------
Split the roles. The diffusion manifold is the MEMORIES: queried slowly,
always from the CLEAN webcam frame (never its own output -> no feedback ->
no contraction -> no cartoon drift). A small U-Net student is the PRESENT:
a deterministic function trained online by distillation on (webcam, dream)
pairs from a replay buffer.

Why this is the "stabilized moving average" you asked for, exactly:
an L1/MSE-trained student converges to the conditional mean of the teacher's
mapping, E[dream | frame]. SGD over the streaming pairs IS an exponential
moving average over the manifold's answers — but stored in weights, as a
mapping, not in pixels as a frame. So when the arm moves, the shirt shifts,
the light changes: the lens renders them in the learned style, because it
learned how this scene MAPS, not how one frame LOOKED.

  - Deterministic: same frame in -> bit-identical frame out. Zero resampling
    flicker by construction. (Verified in test: max|out1-out2| = 0.)
  - Locked teacher seed (default ON): SDXL-Turbo at fixed noise is nearly
    deterministic given the input, so the pairs are consistent and the lens
    locks identity hard instead of averaging over the region.
  - Polyak/EMA weights for display: the literal moving average, smoothing
    the learning itself.
  - Prompt change -> replay buffer clears, weights stay: the lens re-grinds
    itself live, morphing smoothly from one manifold region to another.
  - The chiral field (v24, per-pixel depth-2, zero parameters) now schedules
    the teacher: harvest new pairs when the pose is NEW (field mass), wake on
    motion onset (depth-2 chirp), and it still drives X-ray + the
    moves/still compositing modes.

PhaseLocker is gone: it stabilized a feedback loop that no longer exists.

Mechanism verified headless before shipping (test_lens.py): a frozen random
conv stack played the teacher; the student's held-out L1 dropped to a small
fraction of the do-nothing baseline within a few hundred online steps, and
the EMA copy tracked it. The trainer pattern itself is unitvae4's, kept.

Controls worth knowing:
  Freeze Lens  - stop teaching; the mapping is now a fixed object. Save it,
                 load it tomorrow: a bottled way of seeing.
  Show Teacher - inset of the latest raw (frame, dream) pair: watch the
                 flickery teacher vs the stable lens, the whole thesis on
                 screen at once.

Requires: torch, diffusers, opencv-python, pillow, numpy.
If you OOM: TRAIN_RES = 384 and/or LENS_CH = 24 below.
"""

import os
import time
import random
from collections import deque
from threading import Thread, Lock

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageTk
from tkinter import (Tk, Label, Scale, HORIZONTAL, Frame, Checkbutton,
                     BooleanVar, Button, StringVar, Entry, Radiobutton,
                     filedialog, Canvas, Scrollbar)

FIELD_SIZE = 96
TRAIN_RES = 512        # lens training/render resolution
LENS_CH = 32           # lens width (~6M params); 24 if VRAM is tight
BUFFER_MAX = 96        # replay pairs kept (uint8, CPU RAM)
BATCH = 2
TEACHER_SEED = 1234


# ============================================================
# 1. CHIRAL FIELD  (v24, per-pixel depth-2, zero trained parameters)
# ============================================================
class ChiralField2D:
    """z(t)=x(t)+ix(t-d); w1=z*conj(z_lag): Im w1 = chiral flow (motion);
    w2=w1*conj(w1_lag): Im w2 = chirp (onset). Calibrated in v24:
    real motion E2/E1^2 ~ O(10), steady flicker ~ O(0.5); gate mass_ratio
    ~1.1 frozen vs ~8 moving; onset_ratio wakes within ~2 frames."""

    def __init__(self, size=96, d=1, t1=2, t2=3, ema=0.85, mu_ema=0.95):
        self.size = size
        self.d, self.t1, self.t2 = d, t1, t2
        self.N = d + t1 + t2 + 1
        self.ema = ema
        self.mu_ema = mu_ema
        self.buf = []
        self.mu = None
        self.E1 = None
        self.E2 = None

    def _xc(self, o):
        return self.buf[-1 - o]

    def _w1(self, o):
        d, t1 = self.d, self.t1
        zr0, zi0 = self._xc(o), self._xc(o + d)
        zr1, zi1 = self._xc(o + t1), self._xc(o + t1 + d)
        return zr0 * zr1 + zi0 * zi1, zi0 * zr1 - zr0 * zi1

    def update(self, gray):
        g = gray.astype(np.float32)
        if self.mu is None:
            self.mu = g.copy()
            self.E1 = np.zeros_like(g)
            self.E2 = np.zeros_like(g)
        self.mu = self.mu_ema * self.mu + (1.0 - self.mu_ema) * g
        self.buf.append(g - self.mu)
        if len(self.buf) > self.N:
            self.buf.pop(0)
        if len(self.buf) < self.N:
            return None
        w1r0, w1i0 = self._w1(0)
        w1r2, w1i2 = self._w1(self.t2)
        w2i = w1i0 * w1r2 - w1r0 * w1i2
        a = 1.0 - self.ema
        self.E1 = self.ema * self.E1 + a * np.abs(w1i0)
        self.E2 = self.ema * self.E2 + a * np.abs(w2i)
        return True

    def motion_mask(self, thr=6.0, sens=12.0, flicker_shield=True, shield_div=8.0):
        if self.E1 is None:
            return None
        noise = np.median(self.E1) + 1e-7
        m = np.tanh(np.maximum(0.0, self.E1 / noise - thr) / sens)
        if flicker_shield:
            m = m * np.clip((self.E2 / (self.E1 ** 2 + 1e-9)) / shield_div, 0.05, 1.0)
        return m

    def stats(self):
        if self.E1 is None:
            return {"mass_ratio": 0.0, "onset_ratio": 0.0}
        return {
            "mass_ratio": float(self.E1.mean() / (np.median(self.E1) + 1e-9)),
            "onset_ratio": float(self.E2.mean() / (np.median(self.E2) + 1e-9)),
        }


# ============================================================
# 2. THE LENS  (student U-Net: the present)
# ============================================================
class _Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        g = max(1, min(8, cout // 4))
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(g, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1),     nn.GroupNorm(g, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ManifoldLens(nn.Module):
    """Deterministic image->image function distilled from the manifold."""

    def __init__(self, ch=LENS_CH):
        super().__init__()
        self.e1 = _Block(3, ch)
        self.e2 = _Block(ch, ch * 2, stride=2)
        self.e3 = _Block(ch * 2, ch * 4, stride=2)
        self.e4 = _Block(ch * 4, ch * 8, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d3 = _Block(ch * 8 + ch * 4, ch * 4)
        self.d2 = _Block(ch * 4 + ch * 2, ch * 2)
        self.d1 = _Block(ch * 2 + ch, ch)
        self.head = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        b = self.e4(s3)
        y = self.d3(torch.cat([self.up(b), s3], 1))
        y = self.d2(torch.cat([self.up(y), s2], 1))
        y = self.d1(torch.cat([self.up(y), s1], 1))
        return torch.sigmoid(self.head(y))


class ReplayBuffer:
    """Recent (input, target) uint8 pairs on CPU. Pose diversity across the
    buffer is what makes the lens a mapping, not a memorized frame."""

    def __init__(self, maxlen=BUFFER_MAX):
        self.maxlen = maxlen
        self.items = []
        self.lock = Lock()

    def push(self, inp_u8, tgt_u8):
        with self.lock:
            self.items.append((inp_u8, tgt_u8))
            if len(self.items) > self.maxlen:
                self.items.pop(0)

    def sample(self, n):
        with self.lock:
            if len(self.items) == 0:
                return None, None
            batch = random.sample(self.items, min(n, len(self.items)))
        x = torch.stack([b[0] for b in batch]).float() / 255.0
        y = torch.stack([b[1] for b in batch]).float() / 255.0
        return x, y

    def clear(self):
        with self.lock:
            self.items.clear()

    def __len__(self):
        return len(self.items)


@torch.no_grad()
def ema_update(ema_model, model, decay=0.995):
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.lerp_(p, 1.0 - decay)
    for be, b in zip(ema_model.buffers(), model.buffers()):
        be.copy_(b)


# ============================================================
# 3. THE APP
# ============================================================
PRESETS = [
    ("Marble",  "marble statue, museum lighting, detailed sculpture"),
    ("Anime",   "anime, cel shaded, vivid colors, studio quality"),
    ("Oil",     "oil painting, rembrandt lighting, thick brushstrokes"),
    ("Cyborg",  "cyberpunk android, neon, intricate machinery"),
    ("Zombie",  "zombie, horror film still, decayed, cinematic"),
    ("1600s",   "renaissance portrait, oil on canvas, chiaroscuro"),
]


class ManifoldLensApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Manifold Lens v1  (the manifold is the memories, the lens is the present)")
        self.master.geometry("980x800")
        self.master.configure(bg='#0a0a12')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_lock = Lock()
        self.frame_lock = Lock()

        self.field = ChiralField2D(size=FIELD_SIZE)
        self.buffer = ReplayBuffer()

        self.lens = ManifoldLens().to(self.device)
        self.lens_ema = ManifoldLens().to(self.device)
        self.lens_ema.load_state_dict(self.lens.state_dict())
        self.lens_ema.eval()
        self.opt = torch.optim.Adam(self.lens.parameters(), lr=2e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        self.params = {
            "strength": 0.5,     # teacher dream strength (applied ONCE, never compounded)
            "lr": 2e-4,
            "opacity": 1.0,
            "sens": 6.0,
        }
        self.mode = StringVar(value="lens")
        self.shield_var = BooleanVar(value=True)
        self.xray_var = BooleanVar(value=False)
        self.teacher_pip_var = BooleanVar(value=True)
        self.seed_lock_var = BooleanVar(value=True)

        self.cap = None
        self.pipe = None
        self.running = False
        self.frozen = False
        self.current_webcam = None
        self.mask_small = None
        self.gate_stats = {"mass_ratio": 0.0, "onset_ratio": 0.0}
        self.lens_frame = None          # latest lens render (RGB 512)
        self.last_pair = None           # (input u8 HWC, target u8 HWC) for PiP
        self.last_prompt = None
        self.loss_ema = None
        self.train_steps = 0
        self.harvests = 0
        self._mass_since_harvest = 0.0
        self._last_harvest_t = 0.0
        self._lens_times = deque(maxlen=20)
        self.lens_fps = 0.0
        self.recorder = None
        os.makedirs("captures", exist_ok=True)
        os.makedirs("lenses", exist_ok=True)

        self.setup_gui()
        self.update_video()
        self.master.after(100, lambda: Thread(target=self.load_model, daemon=True).start())

    # ---------------- model ----------------
    def load_model(self):
        try:
            self.master.after(0, lambda: self.status_var.set("loading SDXL-Turbo teacher..."))
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.master.after(0, lambda: self.start_button.config(state='normal', text="Start the Lens"))
            self.master.after(0, lambda: self.status_var.set("ready"))
        except Exception as e:
            print(f"Model Load Error: {e}")
            self.master.after(0, lambda: self.status_var.set(f"model error: {e}"))

    def update_param(self, name, value):
        self.params[name] = float(value)

# ---------------- GUI ----------------
    def setup_gui(self):
        BG, PAN, FG, ACC = '#0a0a12', '#12121e', '#cfd0e0', '#42f5a1'
        main = Frame(self.master, bg=BG)
        main.pack(fill='both', expand=True)
        
        # --- Scrollable Sidebar Container ---
        sidebar_container = Frame(main, width=340, bg=PAN)
        sidebar_container.pack(side='left', fill='y', padx=6, pady=6)
        sidebar_container.pack_propagate(False)

        sidebar_canvas = Canvas(sidebar_container, bg=PAN, highlightthickness=0)
        sidebar_scrollbar = Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        
        # This is the actual frame that will hold all the widgets
        sidebar = Frame(sidebar_canvas, bg=PAN)
        
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_scrollbar.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)
        
        # Anchor the inner frame to the top-left of the canvas (width slightly reduced to fit scrollbar)
        sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=320)
        
        # Update the canvas scroll region when the inner frame changes size
        sidebar.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        )

        # Optional: Add mouse wheel scrolling support
        def _on_mousewheel(event):
            # Handles Windows/Mac (event.delta) and Linux (event.num)
            if hasattr(event, 'num') and event.num == 5 or event.delta < 0:
                sidebar_canvas.yview_scroll(1, "units")
            elif hasattr(event, 'num') and event.num == 4 or event.delta > 0:
                sidebar_canvas.yview_scroll(-1, "units")
                
        sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        sidebar_canvas.bind_all("<Button-4>", _on_mousewheel)
        sidebar_canvas.bind_all("<Button-5>", _on_mousewheel)
        # ------------------------------------

        Label(sidebar, text="MANIFOLD LENS", bg=PAN, fg=ACC,
              font=("Consolas", 13, "bold")).pack(pady=(12, 2))
        Label(sidebar, text="the manifold remembers · the lens sees", bg=PAN,
              fg='#6b6b85', font=("Consolas", 8)).pack(pady=(0, 10))

        self.start_button = Button(sidebar, text="Loading teacher...", command=self.toggle_run,
                                   bg='#1d3a2c', fg=ACC, font=('Consolas', 11, 'bold'),
                                   state='disabled', height=2, relief='flat')
        self.start_button.pack(fill='x', padx=10, pady=4)

        self.freeze_button = Button(sidebar, text="❄ Freeze Lens (stop teaching)",
                                    command=self.toggle_freeze, bg='#23233a', fg=FG,
                                    font=('Consolas', 9), relief='flat')
        self.freeze_button.pack(fill='x', padx=10, pady=2)

        lrow = Frame(sidebar, bg=PAN); lrow.pack(fill='x', padx=10, pady=2)
        Button(lrow, text="💾 Save Lens", command=self.save_lens, bg='#23233a', fg=FG,
               font=('Consolas', 9), relief='flat').pack(side='left', expand=True, fill='x', padx=(0, 3))
        Button(lrow, text="📂 Load Lens", command=self.load_lens, bg='#23233a', fg=FG,
               font=('Consolas', 9), relief='flat').pack(side='left', expand=True, fill='x', padx=(3, 0))

        Label(sidebar, text="Prompt (change it: the lens re-grinds live)", bg=PAN, fg=FG,
              anchor='w', font=("Consolas", 8)).pack(fill='x', padx=10, pady=(10, 0))
        self.prompt_var = StringVar(value=PRESETS[0][1])
        Entry(sidebar, textvariable=self.prompt_var, bg='#23233a', fg=FG,
              insertbackground=FG, relief='flat').pack(fill='x', padx=10, pady=(2, 4))
        prow = Frame(sidebar, bg=PAN); prow.pack(fill='x', padx=8)
        for i, (name, prompt) in enumerate(PRESETS):
            Button(prow, text=name, command=lambda p=prompt: self.prompt_var.set(p),
                   bg='#23233a', fg=FG, font=('Consolas', 8), relief='flat',
                   width=6).grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky='ew')

        Label(sidebar, text="Compositing (chiral field decides)", bg=PAN, fg='#6b6b85',
              font=("Consolas", 9)).pack(fill='x', padx=10, pady=(12, 0))
        for text, val in [("Lens everywhere", "lens"),
                          ("Lens where it moves", "moves"),
                          ("Hold still to transform", "still")]:
            Radiobutton(sidebar, text=text, variable=self.mode, value=val,
                        bg=PAN, fg=FG, selectcolor='#23233a', activebackground=PAN,
                        activeforeground=FG, font=('Consolas', 9), anchor='w').pack(fill='x', padx=14)

        def slider(label, pname, lo, hi, res, default):
            Label(sidebar, text=label, bg=PAN, fg=FG, anchor='w',
                  font=("Consolas", 8)).pack(fill='x', padx=10, pady=(8, 0))
            s = Scale(sidebar, from_=lo, to=hi, resolution=res, orient=HORIZONTAL,
                      bg=PAN, fg=FG, troughcolor='#23233a', highlightthickness=0,
                      command=lambda v, p=pname: self.update_param(p, v))
            s.set(default)
            s.pack(fill='x', padx=10)

        slider("Teacher Dream Strength (applied once, never compounded)",
               "strength", 0.2, 0.9, 0.05, 0.5)
        slider("Lens Learning Rate", "lr", 1e-5, 1e-3, 1e-5, 2e-4)
        slider("Lens Opacity", "opacity", 0.0, 1.0, 0.05, 1.0)
        slider("Field Sensitivity (lower = more)", "sens", 2.0, 20.0, 0.5, 6.0)

        trow = Frame(sidebar, bg=PAN); trow.pack(fill='x', padx=10, pady=(10, 2))
        for text, var in [("Lock teacher seed (identity lock)", self.seed_lock_var),
                          ("Flicker Shield (E2/E1²)", self.shield_var),
                          ("X-Ray (show the field)", self.xray_var),
                          ("Show Teacher PiP (inset)", self.teacher_pip_var)]: 
            Checkbutton(trow, text=text, variable=var, bg=PAN, fg=FG,
                        selectcolor='#23233a', activebackground=PAN,
                        activeforeground=FG, font=('Consolas', 9)).pack(anchor='w')

        Label(sidebar, text="Main Display View", bg=PAN, fg='#6b6b85',
              font=("Consolas", 9)).pack(fill='x', padx=10, pady=(12, 0))
        
        self.main_view = StringVar(value="lens")
        for text, val in [("Student Lens (Smooth)", "lens"),
                          ("Teacher (Raw SDXL)", "teacher"),
                          ("Webcam (Reality)", "webcam")]:
            Radiobutton(sidebar, text=text, variable=self.main_view, value=val,
                        bg=PAN, fg=FG, selectcolor='#23233a', activebackground=PAN,
                        activeforeground=FG, font=('Consolas', 9), anchor='w').pack(fill='x', padx=14)

        crow = Frame(sidebar, bg=PAN); crow.pack(fill='x', padx=10, pady=8)
        Button(crow, text="📸 Snapshot", command=self.snapshot, bg='#23233a', fg=FG,
               font=('Consolas', 9), relief='flat').pack(side='left', expand=True, fill='x', padx=(0, 3))
        self.rec_button = Button(crow, text="⏺ Record", command=self.toggle_record,
                                 bg='#23233a', fg=FG, font=('Consolas', 9), relief='flat')
        self.rec_button.pack(side='left', expand=True, fill='x', padx=(3, 0))

        # Padding at the bottom for scroll comfort
        Frame(sidebar, bg=PAN, height=20).pack(fill='x')

        video_area = Frame(main, bg='black')
        video_area.pack(side='right', fill='both', expand=True, padx=6, pady=6)
        self.panel = Label(video_area, bg='black')
        self.panel.pack(expand=True)

        self.status_var = StringVar(value="init...")
        Label(self.master, textvariable=self.status_var, relief='flat', anchor='w',
              bg=BG, fg=ACC, font=("Consolas", 9)).pack(side='bottom', fill='x')

    # ---------------- camera + display ----------------
    def update_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if ret:
            small = cv2.resize(frame, (FIELD_SIZE, FIELD_SIZE))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            self.field.update(gray)
            mask = self.field.motion_mask(thr=self.params["sens"],
                                          flicker_shield=self.shield_var.get())
            stats = self.field.stats()
            self._mass_since_harvest = max(self._mass_since_harvest, stats["mass_ratio"])
            with self.frame_lock:
                self.current_webcam = frame
                self.mask_small = mask
                self.gate_stats = stats

            webcam_512 = cv2.cvtColor(cv2.resize(frame, (TRAIN_RES, TRAIN_RES)), cv2.COLOR_BGR2RGB)
            disp = self.compose_display(webcam_512, mask)
            if self.recorder is not None:
                self.recorder.write(cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                cv2.circle(disp, (24, 24), 8, (255, 59, 107), -1)
            img = ImageTk.PhotoImage(image=Image.fromarray(disp))
            self.panel.imgtk = img
            self.panel.config(image=img)
        self.master.after(20, self.update_video)

    def compose_display(self, webcam_512, mask):
        # 1. Run the lens (we keep this running in the background to maintain the FPS counter and keep states fresh)
        lens_out = None
        if self.running and self.gpu_lock.acquire(blocking=False):
            try:
                with torch.no_grad():
                    x = torch.from_numpy(webcam_512).permute(2, 0, 1).unsqueeze(0)
                    x = (x.float() / 255.0).to(self.device)
                    with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                        y = self.lens_ema(x)
                    lens_out = (y[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                self._lens_times.append(time.time())
                if len(self._lens_times) > 1:
                    self.lens_fps = (len(self._lens_times) - 1) / (
                        self._lens_times[-1] - self._lens_times[0] + 1e-6)
                with self.frame_lock:
                    self.lens_frame = lens_out
            finally:
                self.gpu_lock.release()
                
        if lens_out is None:
            with self.frame_lock:
                lens_out = self.lens_frame

        # --- 2. VIEW SELECTION ROUTING ---
        view = self.main_view.get()

        if view == "webcam":
            # Just show reality
            disp = webcam_512.copy()
            
        elif view == "teacher":
            # Show the raw SDXL target (if it exists yet, otherwise fallback to webcam)
            if self.last_pair is not None:
                disp = self.last_pair[1].copy() 
            else:
                disp = webcam_512.copy()
                
        else:
            # view == "lens": The standard student lens + chiral compositing
            if lens_out is None:
                disp = webcam_512.copy()
            else:
                op = self.params["opacity"]
                mode = self.mode.get()
                if mask is None or mode == "lens":
                    D = np.full((TRAIN_RES, TRAIN_RES, 1), op, np.float32)
                else:
                    M = cv2.resize(mask, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_LINEAR)
                    M = np.clip(cv2.GaussianBlur(M, (0, 0), 10.0), 0, 1)[:, :, None]
                    D = op * (M if mode == "moves" else (1.0 - M))
                disp = (lens_out.astype(np.float32) * D +
                        webcam_512.astype(np.float32) * (1.0 - D)).clip(0, 255).astype(np.uint8)

        # 3. Apply Overlays (X-Ray and Teacher PiP always draw on top of whatever view you selected)
        if self.xray_var.get() and mask is not None:
            heat = cv2.applyColorMap((np.clip(mask, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)
            heat = cv2.cvtColor(cv2.resize(heat, (TRAIN_RES, TRAIN_RES)), cv2.COLOR_BGR2RGB)
            disp = cv2.addWeighted(disp, 0.5, heat, 0.5, 0)

        if self.teacher_pip_var.get() and self.last_pair is not None:
            inp, tgt = self.last_pair
            s = 128
            strip = np.concatenate([cv2.resize(inp, (s, s)), cv2.resize(tgt, (s, s))], axis=1)
            disp[8:8 + s, TRAIN_RES - 2 * s - 8:TRAIN_RES - 8] = strip
            cv2.rectangle(disp, (TRAIN_RES - 2 * s - 8, 8), (TRAIN_RES - 8, 8 + s), (66, 245, 161), 1)
            
        return disp

    # ---------------- teacher: harvest pairs from the manifold ----------------
    def teacher_loop(self):
        while self.running:
            if self.pipe is None or self.frozen:
                time.sleep(0.2)
                continue
            prompt = self.prompt_var.get()
            if prompt != self.last_prompt:
                if self.last_prompt is not None:
                    self.buffer.clear()
                    self.loss_ema = None
                self.last_prompt = prompt

            now = time.time()
            need_bootstrap = len(self.buffer) < 24
            pose_is_new = self._mass_since_harvest > 2.5
            slow_refresh = (now - self._last_harvest_t) > 5.0
            if not (now - self._last_harvest_t > 0.4 and
                    (need_bootstrap or pose_is_new or slow_refresh)):
                time.sleep(0.05)
                continue

            with self.frame_lock:
                frame = None if self.current_webcam is None else self.current_webcam.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            webcam_512 = cv2.cvtColor(cv2.resize(frame, (TRAIN_RES, TRAIN_RES)), cv2.COLOR_BGR2RGB)

            try:
                strength = self.params["strength"]
                steps = max(2, int(np.ceil(1.0 / max(0.05, strength))))
                gen = (torch.Generator(device=self.device).manual_seed(TEACHER_SEED)
                       if self.seed_lock_var.get() else None)
                with self.gpu_lock:
                    result = self.pipe(
                        prompt=prompt,
                        image=Image.fromarray(webcam_512),  # ALWAYS the clean frame
                        strength=strength,
                        guidance_scale=0.0,
                        num_inference_steps=steps,
                        generator=gen,
                    ).images[0]
                tgt = np.array(result)
                self.buffer.push(
                    torch.from_numpy(webcam_512).permute(2, 0, 1).contiguous(),
                    torch.from_numpy(tgt).permute(2, 0, 1).contiguous(),
                )
                self.last_pair = (webcam_512, tgt)
                self.harvests += 1
                self._last_harvest_t = now
                self._mass_since_harvest = 0.0
            except Exception as e:
                print(f"Teacher error: {e}")
                time.sleep(0.5)

    # ---------------- student: distill the mapping ----------------
    def student_loop(self):
        while self.running:
            if self.frozen or len(self.buffer) < 8:
                time.sleep(0.1)
                continue
            try:
                bx, by = self.buffer.sample(BATCH)
                if bx is None:
                    continue
                bx, by = bx.to(self.device), by.to(self.device)
                for g in self.opt.param_groups:
                    g["lr"] = self.params["lr"]
                with self.gpu_lock:
                    self.lens.train()
                    self.opt.zero_grad()
                    with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                        loss = (self.lens(bx) - by).abs().mean()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    ema_update(self.lens_ema, self.lens, 0.995)
                l = loss.item()
                self.loss_ema = l if self.loss_ema is None else 0.95 * self.loss_ema + 0.05 * l
                self.train_steps += 1
                if self.train_steps % 5 == 0:
                    st = self.gate_stats
                    self.master.after(0, lambda: self.status_var.set(
                        f"{'FROZEN' if self.frozen else 'GRINDING'}"
                        f" | pairs {len(self.buffer)} | step {self.train_steps}"
                        f" | L1 {self.loss_ema:.3f}"
                        f" | lens {self.lens_fps:.0f} fps | harvests {self.harvests}"
                        f" | gate {st['mass_ratio']:.1f}"))
            except Exception as e:
                print(f"Student error: {e}")
                time.sleep(0.5)

    # ---------------- buttons ----------------
    def toggle_run(self):
        if self.running:
            self.running = False
            self.start_button.config(text="Start the Lens", bg='#1d3a2c')
        else:
            self.running = True
            self.start_button.config(text="Stop the Lens", bg='#3a1622')
            Thread(target=self.teacher_loop, daemon=True).start()
            Thread(target=self.student_loop, daemon=True).start()

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.freeze_button.config(
            text="🔥 Unfreeze Lens (resume teaching)" if self.frozen
            else "❄ Freeze Lens (stop teaching)")
        self.status_var.set("lens frozen: the mapping is now a fixed object"
                            if self.frozen else "teaching resumed")

    def save_lens(self):
        path = filedialog.asksaveasfilename(title="Save Lens", initialdir="lenses",
                                            defaultextension=".pth",
                                            filetypes=[("Lens", "*.pth")])
        if path:
            torch.save({"lens": self.lens.state_dict(),
                        "ema": self.lens_ema.state_dict(),
                        "prompt": self.prompt_var.get()}, path)
            self.status_var.set(f"lens saved: {path}")

    def load_lens(self):
        path = filedialog.askopenfilename(title="Load Lens", initialdir="lenses",
                                          filetypes=[("Lens", "*.pth")])
        if path:
            ck = torch.load(path, map_location=self.device)
            self.lens.load_state_dict(ck["lens"])
            self.lens_ema.load_state_dict(ck["ema"])
            if "prompt" in ck:
                self.prompt_var.set(ck["prompt"])
                self.last_prompt = ck["prompt"]
            self.status_var.set(f"lens loaded: {path}")

    def snapshot(self):
        with self.frame_lock:
            disp = self.lens_frame
        if disp is not None:
            path = time.strftime("captures/lens_%Y%m%d_%H%M%S.png")
            cv2.imwrite(path, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            self.status_var.set(f"saved {path}")

    def toggle_record(self):
        if self.recorder is None:
            path = time.strftime("captures/lens_%Y%m%d_%H%M%S.mp4")
            self.recorder = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            25, (TRAIN_RES, TRAIN_RES))
            self.rec_button.config(text="⏹ Stop", bg='#3a1622')
        else:
            self.recorder.release()
            self.recorder = None
            self.rec_button.config(text="⏺ Record", bg='#23233a')
            self.status_var.set("recording saved")


if __name__ == "__main__":
    root = Tk()
    app = ManifoldLensApp(root)
    root.mainloop()
