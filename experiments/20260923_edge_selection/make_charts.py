import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

FONT_R = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_B = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(FONT_R)
fm.fontManager.addfont(FONT_B)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_R).get_name()
plt.rcParams["axes.unicode_minus"] = False

from paths import IMG_OUT

OUT = str(IMG_OUT)          # 저장소 안 docs/images/20260923_edge_report
os.makedirs(OUT, exist_ok=True)

GOOD = "#1f8a52"
BAD = "#c1432b"
ACCENT = "#2f5fd8"
WARN = "#a3730a"
GRID = "#d7dbdf"
INK = "#14181d"
MUTED = "#5b6570"

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# ---------- 차트 1: 5개 설정 std 비교 (로그) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
labels = ["① 기본\n(margin .40)", "② 범위만\n박스기준", "③ 영역만\n5%축소", "④ 둘 다\n(채택)", "⑤ 마진 0\n(박스그대로)"]
vals = [145.260, 27.631, 30.666, 0.932, 1.421]
colors = [BAD, BAD, BAD, GOOD, BAD]
bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3)
ax.set_yscale("log")
ax.set_ylabel("31점 산포 std (px, 로그축)", fontsize=11, color=INK)
ax.set_title("A2 한 장 — 탐색영역·인정범위 조합 5가지", fontsize=13, color=INK, fontweight="bold", pad=14)
style(ax)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}px", (b.get_x()+b.get_width()/2, v), xytext=(0,6),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/chart1_5configs.png", facecolor="white")
plt.close(fig)

# ---------- 차트 2: 14장, 기본설정 vs 수정설정 대표값 ----------
shots = np.arange(1, 15)
default_med = [1155.1,1288.0,1488.7,1157.0,1157.7,1157.5,1157.5,1157.9,1157.6,1156.8,1157.8,1157.0,1493.4,1157.8]
fixed_med   = [1157.0,1157.0,1157.07,1157.01,1157.0,1157.6,1157.2,1157.4,1157.0,1157.3,1157.6,1157.2,1158.0,1157.6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=170)
c1 = [BAD if abs(v-1157) > 10 else MUTED for v in default_med]
ax1.bar(shots, default_med, color=c1, width=0.62, zorder=3)
ax1.axhline(1157, color=GRID, linewidth=1, linestyle="--", zorder=2)
ax1.set_title("기본 설정 (margin .40, roi기준)", fontsize=12.5, color=INK, fontweight="bold")
ax1.set_ylabel("대표값 median (px)", fontsize=10.5, color=INK)
ax1.set_xlabel("장 번호", fontsize=10.5, color=INK)
ax1.set_xticks(shots)
style(ax1)
for x, v in zip(shots, default_med):
    if abs(v-1157) > 10:
        ax1.annotate(f"{v:.0f}", (x, v), xytext=(0,6), textcoords="offset points",
                     ha="center", fontsize=9, color=BAD, fontweight="bold")

ax2.bar(shots, fixed_med, color=GOOD, width=0.62, zorder=3)
ax2.set_ylim(1154, 1160)
ax2.axhline(1157, color=GRID, linewidth=1, linestyle="--", zorder=2)
ax2.set_title("수정 설정 (margin .05 + box기준) — 같은 세로축 단위(px)", fontsize=12.5, color=INK, fontweight="bold")
ax2.set_xlabel("장 번호", fontsize=10.5, color=INK)
ax2.set_xticks(shots)
style(ax2)

fig.suptitle("시편1 14장 전체 — 기본 vs 수정 설정 대표값(px) 비교", fontsize=13.5, color=INK, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/chart2_14shots_before_after.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------- 차트 3: 라인피팅 결합 — 컷 내부 std, 14장 ----------
mad_std  = [1.06,0.93,0.49,0.24,1.10,0.63,1.09,1.31,1.35,0.55,1.46,1.17,0.27,1.24]
line_std = [0.143,0.465,0.175,0.221,0.170,0.818,0.149,0.072,0.002,0.024,0.101,0.489,0.237,0.069]

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=170)
w = 0.38
ax.bar(shots - w/2, mad_std, width=w, color=ACCENT, label="수정 설정만 (MAD)", zorder=3)
ax.bar(shots + w/2, line_std, width=w, color=GOOD, label="+ 라인 피팅 추가", zorder=3)
ax.set_xticks(shots)
ax.set_xlabel("장 번호", fontsize=11, color=INK)
ax.set_ylabel("컷 내부 31점 std (px)", fontsize=11, color=INK)
ax.set_title("시편1 14장 — 컷 내부 산포: 수정 설정만 vs 라인 피팅 추가", fontsize=13.5, color=INK, fontweight="bold", pad=14)
ax.legend(frameon=False, fontsize=10.5, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/chart3_linefit_within_shot.png", facecolor="white")
plt.close(fig)

# ---------- 차트 4: 라인피팅 결합 — 장간 반복 sd 비교 (핵심지표) ----------
fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=170)
cats = ["수정 설정만\n(MAD)", "+ 라인 피팅"]
sds = [0.31, 0.411]
colors4 = [GOOD, WARN]
bars = ax.bar(cats, sds, color=colors4, width=0.55, zorder=3)
ax.set_ylabel("14장 간 반복 sd (px)", fontsize=11, color=INK)
ax.set_title("핵심 지표 — 장간 반복성\n(낮을수록 좋음)", fontsize=12.5, color=INK, fontweight="bold", pad=12)
style(ax)
for b, v, um in zip(bars, sds, ["약 5.4um", "약 7.1um"]):
    ax.annotate(f"{v:.2f}px\n({um})", (b.get_x()+b.get_width()/2, v), xytext=(0,8),
                textcoords="offset points", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.set_ylim(0, 0.55)
fig.tight_layout()
fig.savefig(f"{OUT}/chart4_linefit_crossshot.png", facecolor="white")
plt.close(fig)

print("saved:", os.listdir(OUT))

# ---------- 차트 5: A2 기본설정 31개 측정선의 지름 분포 — 왜 시스템이 못 걸렀나 ----------
dia = np.array([1156.4,1483.8,1155.2,1484.2,1483.3,1288.0,1286.5,1156.9,1158.3,1485.1,
                1288.0,1485.7,1154.1,1154.1,1157.1,1484.4,1485.2,1397.8,1342.4,1157.5,
                1342.6,1154.0,1485.6,1484.8,1156.9,1486.5,1156.1,1153.9,1385.9,1154.8,1154.5])
wrong = np.array([0,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,0], dtype=bool)
idx = np.arange(1, 32)
med_all = np.median(dia)
mad = np.median(np.abs(dia - med_all))
sigma = 1.4826 * mad
lo, hi = med_all - 3*sigma, med_all + 3*sigma

fig, ax = plt.subplots(figsize=(11, 4.8), dpi=170)
ax.axhspan(lo, hi, color=WARN, alpha=0.10, zorder=1)
ax.axhline(lo, color=WARN, lw=1, ls=":", zorder=2)
ax.axhline(hi, color=WARN, lw=1, ls=":", zorder=2)
ax.axhline(1155.7, color=GOOD, lw=1.6, ls="--", zorder=2)
ax.axhline(med_all, color=BAD, lw=1.8, zorder=2)
ax.scatter(idx[~wrong], dia[~wrong], s=70, color=GOOD, zorder=4, label=f"실제 외곽을 잡은 점 ({(~wrong).sum()}개)")
ax.scatter(idx[wrong], dia[wrong], s=70, color=BAD, marker="X", zorder=4, label=f"반사띠+배경을 잡은 점 ({wrong.sum()}개)")
ax.annotate(f"이상치 허용 밴드 = 중앙값 ±3σ (σ는 MAD로 계산) = {lo:.0f} ~ {hi:.0f}px\n"
            f"두 무리로 갈리는 바람에 밴드가 ±{3*sigma:.0f}px(약 ±10mm)까지 벌어져서, 31개가 전부 통과한다",
            xy=(17, hi), xytext=(2.2, hi+60), fontsize=10.5, color=WARN,
            arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2))
ax.annotate(f"시스템이 고른 대표값 {med_all:.1f}px — 오답이 과반이라 여기로 끌려간다",
            xy=(24, med_all), xytext=(10.5, med_all+95), fontsize=10.5, color=BAD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2))
ax.annotate("진짜 값 1155.7px", xy=(3.5, 1155.7), xytext=(3.2, 1155.7-170), fontsize=10.5,
            color=GOOD, fontweight="bold", arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.2))
ax.set_xlabel("측정선 번호 (ROI 가로 방향 31개)", fontsize=11, color=INK)
ax.set_ylabel("그 측정선이 잰 지름 (px)", fontsize=11, color=INK, labelpad=8)
ax.set_title("A2 원본·기본설정 — 31개 측정선이 두 무리로 갈린다", fontsize=13.5, color=INK, fontweight="bold", pad=14)
ax.set_xlim(0, 32); ax.set_ylim(600, 2050)
ax.legend(frameon=False, fontsize=10.5, loc="lower left", ncol=2)
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/chart5_two_clusters.png", facecolor="white")
plt.close(fig)
print(f"chart5: median={med_all:.1f} MAD={mad:.1f} sigma={sigma:.1f} band={lo:.0f}~{hi:.0f}")
