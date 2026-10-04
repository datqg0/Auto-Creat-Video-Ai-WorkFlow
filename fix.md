# Fix List - Auto-Create-Video-AI Pipeline

Muc do uu tien: HIGH | MEDIUM | LOW

---

## [FIX-01] HIGH - Code AI: frame count sai, MoviePy doc past EOF

File: src/ai_code_runner.py

Van de: Warning MoviePy 2764800 bytes wanted but 0 bytes read at frame 539/529.
Nguyen nhan: encoder khong flush du frame cuoi khi luu mp4, MoviePy doc past EOF.

Fix:
1. Them extra_args=[-pix_fmt,yuv420p,-movflags,+faststart] vao ani.save() trong _PRELUDE comment.
2. Them rule bat buoc vao prompt generate_matplotlib_from_prompt dong 164:
   MUST use: ani.save(OUT_PATH, fps=FPS, writer=ffmpeg, extra_args=[-pix_fmt,yuv420p,-movflags,+faststart])
3. Them try/except quanh bg.subclipped() trong compositor._code_clip() dong 134.

---

## [FIX-02] LOW - Manim: _MANIM_QUALITY_FLAG dat SAU ham dung no

File: src/ai_code_runner.py dong 526-531

Fix: Di chuyen dict _MANIM_QUALITY_FLAG len ngay truoc ham run_manim_code().

---

## [FIX-03] HIGH - TTS: Short audio bi cat giua cau khi vuot 179s

File: config.yaml dong 190, 198; src/compositor.py dong 297-302

Van de: Neu LLM sinh du tu, TTS doc qua 175s -> compositor cat cung ve 179s -> audio bi cat giua cau.

Fix:
- Giam VieNeu speed: 0.90 (hien tai 0.95)
- Edge TTS rate: -8% (hien tai -6%)
- Them audio fade-out 0.5s khi cat Short de tranh am thanh dot ngot.

---

## [FIX-04] HIGH - Prompt Matplotlib: thieu rang buoc nhip cham rai

File: src/ai_code_runner.py - generate_matplotlib_from_prompt() dong 143-166

Van de: Animation dang chay QUA NHANH. AI nhet qua nhieu buoc vao animation ngan,
nguoi xem khong kip ngam.

Fix: Them vao prompt cac quy tac nhip cung hon:
  - SLOW DOWN: Each conceptual step MUST be visible for at least 0.8-1.5 seconds.
  - Max 3-4 key transitions for the animation. Do NOT cram more steps.
  - After each major state change, insert a PAUSE (empty frames 0.5-1.0s).
  - Use np.linspace for gradual/smooth changes, NOT instant jumps.

---

## [FIX-05] HIGH - Prompt Manim: rang buoc nhip chua du, self.wait() bi bo qua

File: src/ai_code_runner.py - generate_manim_from_prompt() dong 207-265

Van de: Manim animation cung bi nhip nhanh. AI hay bo qua self.wait() va nhoi nhieu
self.play() lien tiep, khong co khoang nghi giua cac buoc.

Fix: Them rule manh hon vao prompt:
  RHYTHM RULE (NON-NEGOTIABLE): Between EVERY self.play(), insert self.wait(0.6-1.0).
  Max 4-5 play() calls for the full scene. Budget time carefully.
  WRONG: self.play(A); self.play(B); self.play(C)   -- Too fast!
  RIGHT: self.play(A, run_time=1.2); self.wait(0.8); self.play(B, run_time=1.0); self.wait(0.6);

---

## [FIX-06] MEDIUM - Script writer: vi du trong prompt chua du rock-solid

File: src/script_writer.py - _build_prompt() dong 73-76

Van de: LLM doi khi van sinh vi du truu tuong du da co rule yeu cau cu the.

Fix: Them 2-3 mau vi du cu the vao prompt:
  DUNG: Mang [5,2,8,1,9] -> Merge Sort -> ket qua [1,2,5,8,9] sau 12 lan so sanh vs 25 cua Bubble Sort.
  SAI: thuat toan sap xep hieu qua hon -- qua mo ho, bi cam.

---

## [FIX-07] MEDIUM - visual_prompt mau con thieu phan nhip chuyen dong

File: src/script_writer.py dong 177-184

Van de: LLM van hay viet visual_prompt ngan gon, thieu chi tiet ve du lieu, buoc chuyen, nhip.

Fix: Them vi du mau bat buoc day du du lieu + buoc chuyen + nhip vao prompt:
  Mang 8 phan tu [3,7,1,9,5,2,8,4] tren nen toi.
  Phase 1 (0-2s): Reveal tung phan tu tu trai, moi fade-in 0.15s.
  Phase 2 (2-5s): left di chuyen sang phai; o dang xet bat Hong; o da xong chuyen Xanh.
  Phase 3 (5-7s): Tim thay target=5, nhap nhay 3 lan, hien FOUND mau Xanh.
  Tong 7s. Moi buoc dung 0.6s de nguoi xem kip nhin.

---

## [FIX-08] MEDIUM - Short min_words qua thap, video thuc te ngan hon target

File: src/script_writer.py dong 638

Van de: min_words = approx_words * 0.65 -> voi duration=175s: approx=379 tu, min=246 tu.
Neu LLM chi viet 250 tu ma pass threshold -> video thuc te chi ~115s thay vi 175s.

Fix: Tang nguong minimum len 0.80 cho short mode:
  min_words = approx_words * 0.75  (long mode)
  min_words = approx_words * 0.80  (short mode)
Short 175s: min_words = 303 tu (du ~140s @ 130wpm + pause).

---

## [FIX-09] MEDIUM - Short prompt thieu rule nhip cham cho animation

File: src/script_writer.py - _build_short_prompt() dong 262-266

Van de: Chi co "Nen co 2-3 scene animation" nhung khong yeu cau nhip cham,
khien Short animation cung bi nhanh nhu Long.

Fix: Them vao _build_short_prompt:
  - Moi scene animation PHAI co visual_prompt mo ta nhip CHAM RAI: moi buoc dung >=0.8s.
  - Toi da 3-4 buoc chuyen dong cho moi animation scene trong Short.

---

## [FIX-10] HIGH - compositor.py: W, H, FPS cache tai import time

File: src/compositor.py dong 43-45

Van de:
  W = CONFIG["visual"]["width"]   # = 1920 tai luc import
  H = CONFIG["visual"]["height"]  # = 1080 tai luc import

apply_mode() thay doi config SAU khi module da import. Vi W, H la hang so module-level,
chung KHONG update. Khi chay --mode short sau khi da cache module,
compositor van render 1920x1080 thay vi 1080x1920.

CANH BAO: Day la bug tiem an nghiem trong, nen fix som.

Fix: Doi cac bien module-level thanh ham:
  def _W(): return CONFIG["visual"]["width"]
  def _H(): return CONFIG["visual"]["height"]
  def _FPS(): return CONFIG["visual"]["fps"]
Roi thay W->_W(), H->_H(), FPS->_FPS() khap file compositor.py.

---

## [FIX-11] HIGH - visual_engine.py: W, H cung cache tai import time

File: src/visual_engine.py dong 23-24

Van de: Tuong tu FIX-10. W va H duoc gan tai import time,
khong reflect khi apply_mode() thay doi kich thuoc cho Short.

Fix: Thay tat ca W, H bang CONFIG["visual"]["width"], CONFIG["visual"]["height"] truc tiep.

---

## [FIX-12] MEDIUM - ai_code_runner.py: timeout Matplotlib = 90s qua ngan cho CI

File: src/ai_code_runner.py dong 292

Van de: timeout = 90s. Voi animation 175s x 30fps = 5250 frames, render Matplotlib
co the mat 120-180s. Neu timeout -> AI repair cung timeout -> fallback ve anh tinh
-> nhieu scene khong co animation.

Fix: Tang default timeout len 150s:
  run_ai_code(..., timeout=150, ...)

---

## [FIX-13] MEDIUM - Manim runner: bien stderr co the chua duoc dinh nghia

File: src/ai_code_runner.py dong 466-511

Van de: Neu proc la None (timeout) o attempt dau tien, dong 511
(stderr = proc.stderr if proc else stderr) dung bien stderr chua duoc dinh nghia -> NameError.

Fix: Them khoi tao truoc vong lap:
  stderr_val = ""
  for attempt in range(max_repairs + 1):
      ...

---

## [FIX-14] LOW - System prompt thieu vi du an du Feynman cu the

File: src/script_writer.py - _SYSTEM dong 14-22

Van de: System prompt da tot nhung thieu vi du cu the ve cach an du doi thuong.

Fix: Them vao _SYSTEM vi du an du dung chuan:
  Hash Table: Thu thu thong minh: dan nhan dau ngan -> tim ngay O(1).
  TCP/IP: Gui buu pham chia nhieu goi co dia chi, so thu tu; nguoi nhan ghep lai.
  Mutex: Chia khoa nha ve sinh: chi 1 nguoi co chia khoa -> khong ai vao trung.

---

## [FIX-15] LOW - GitHub Actions: khong luu artifact khi pipeline loi

File: .github/workflows/create-video.yml dong 154-160

Van de: Step "Luu artifact khi test" chi chay khi no_upload == true. Khi pipeline loi,
output video da render mot phan bi mat, kho debug.

Fix: Them step upload artifact khi failure():
  - name: Upload debug artifacts on failure
    if: failure()
    uses: actions/upload-artifact@v4
    with:
      name: debug-output
      path: output/**/
      retention-days: 3

---

## [FIX-16] HIGH - _clean_narration_markdown() dung sai bien text thay vi t

File: src/script_writer.py dong 413-416

Van de:
  t = re.sub(URL_PATTERN, "", text)        # xu ly URL -> luu vao t
  t = re.sub(LINK_PATTERN, r"\1", t)      # xu ly link -> luu vao t
  t = re.sub(BOLD_PATTERN, r"\1", text)   # BUG! Dung text (bien goc) thay vi t
  t = re.sub(ITALIC_PATTERN, r"\1", t)

Dong 415 dung text (bien goc, chua xu ly URL) thay vi t (da xoa URL) -> URL bi giu lai.

Fix dong 415: thay text thanh t:
  t = re.sub(BOLD_PATTERN, r"\1", t)   # doi text thanh t

---

## [FIX-17] LOW - TTS speed VieNeu vs Edge khong dong nhat

File: config.yaml dong 190, 198

Van de: VieNeu speed: 0.95 (cham 5%) nhung Edge rate: -6% (cham 6%).
Khi fallback giua cac scene, nhip doc se thay doi nhe.

Fix: Dong nhat ca hai:
  VieNeu: speed: 0.90
  Edge: rate: -10%

---

## [FIX-18] LOW - requirements.txt: co the thieu json_repair

File: requirements.txt, src/script_writer.py dong 329

Van de: script_writer.py dung import json_repair nhung goi nay co the khong co trong
requirements.txt -> loi khi cai dependencies.

Fix: Kiem tra va them neu thieu:
  json-repair>=0.30

---

## Tom tat uu tien thuc hien

| Ma      | Mo ta ngan                              | Uu tien | File chinh              |
|---------|-----------------------------------------|---------|-------------------------|
| FIX-01  | MoviePy frame EOF warning               | HIGH    | ai_code_runner.py       |
| FIX-03  | TTS Short bi cat audio giua cau         | HIGH    | config.yaml, compositor |
| FIX-04  | Animation Matplotlib qua nhanh          | HIGH    | ai_code_runner.py       |
| FIX-05  | Animation Manim qua nhanh               | HIGH    | ai_code_runner.py       |
| FIX-10  | W/H cache - Short render sai kich thuoc | HIGH    | compositor.py           |
| FIX-11  | W/H cache - Short render sai kich thuoc | HIGH    | visual_engine.py        |
| FIX-16  | clean_narration dung sai bien text      | HIGH    | script_writer.py        |
| FIX-06  | Vi du trong prompt chua du cu the       | MEDIUM  | script_writer.py        |
| FIX-07  | visual_prompt mau thieu phan nhip       | MEDIUM  | script_writer.py        |
| FIX-08  | Short min_words qua thap                | MEDIUM  | script_writer.py        |
| FIX-09  | Short animation thieu rule nhip cham    | MEDIUM  | script_writer.py        |
| FIX-12  | Matplotlib timeout 90s qua ngan         | MEDIUM  | ai_code_runner.py       |
| FIX-13  | Manim stderr undefined khi proc=None    | MEDIUM  | ai_code_runner.py       |
| FIX-02  | _MANIM_QUALITY_FLAG dat sai vi tri      | LOW     | ai_code_runner.py       |
| FIX-14  | System prompt thieu vi du Feynman       | LOW     | script_writer.py        |
| FIX-15  | Khong upload artifact khi pipeline loi  | LOW     | create-video.yml        |
| FIX-17  | TTS speed VieNeu vs Edge khong nhat quan| LOW     | config.yaml             |
| FIX-18  | json_repair thieu trong requirements    | LOW     | requirements.txt        |
