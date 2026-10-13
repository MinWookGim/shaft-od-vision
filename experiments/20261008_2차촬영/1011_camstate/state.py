# 130장 전부의 카메라 상태를 고정물로 측정한다 (2026-10-11, 현장 메모 "화면바뀜·focus 새로" 대조용)
# 고정물: 자석 받침대 두 개. 실제로 움직인 적 없음(팀장 확인)
#   (1차 시도: 받침대 옆판(옅은 회색)과 백라이트(흰색) 경계 x 6600·9645 -> 대비가 약해 행마다 50~90px 흔들림. 버림)
#   왼쪽 xL: 행 4300~4900, x 6060~6200 에서 검은 몸통 -> 옆판 경계(밝기 변화율 최대, 포물선 소수점), 행별 중앙값
#   오른쪽 xR: 같은 행, x 10040~10160 에서 옆판 -> 검은 몸통 경계
#   창 아래 경계 yB: 열 7000~9300, y 4900~5050 에서 밝음 -> 어두움 (변화율 최대)
# 배율 자 = xR - xL (약 3970px). 핀(270px)보다 약 15배 길다
import os, csv, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'field', 'photos', '20261008_원전기계', 'full_result')
rows = list(csv.DictReader(open(os.path.join(HERE, '..', '1008_field', '_out', 'full_rows.csv'), encoding='utf-8-sig')))

def gpeak(prof):
    """밝기 변화율 절댓값이 가장 큰 곳(포물선 소수점)"""
    p = np.convolve(prof, np.ones(3) / 3, 'same'); d = np.abs(np.diff(p))[2:-2]; i = int(np.argmax(d))
    if 0 < i < len(d) - 1:
        a, b, c = d[i - 1], d[i], d[i + 1]; off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
    else: off = 0
    return i + 2 + 0.5 + off

def cross(prof, rising):
    """프로파일에서 50% 밝기를 처음 지나는 곳(소수점). rising=True 면 어두움->밝음"""
    lo, hi = np.percentile(prof, 5), np.percentile(prof, 95); thr = (lo + hi) / 2
    p = prof if rising else -prof; t = thr if rising else -thr
    i = int(np.argmax(p > t))
    if i == 0: return np.nan
    return i - 1 + (t - p[i - 1]) / (p[i] - p[i - 1])

out = csv.writer(open(os.path.join(HERE, '_out', 'state.csv'), 'w', newline='', encoding='utf-8-sig'))
out.writerow(['파일', '묶음', '시편', '초점', 'xL', 'xR', '간격', 'yB', 'xL_sd', 'xR_sd'])
for k, r in enumerate(rows):
    g = np.asarray(Image.open(os.path.join(D, r['파일'])).convert('L'), dtype=float)
    L = [6060 + gpeak(g[y, 6060:6200]) for y in range(4300, 4900, 4)]
    R = [10040 + gpeak(g[y, 10040:10160]) for y in range(4300, 4900, 4)]
    B = [4900 + gpeak(g[4900:5050, x]) for x in range(7000, 9300, 8)]
    xL, xR, yB = np.nanmedian(L), np.nanmedian(R), np.nanmedian(B)
    out.writerow([r['파일'], r['묶음'], r['시편'], r['초점'], f'{xL:.3f}', f'{xR:.3f}', f'{xR - xL:.3f}', f'{yB:.3f}',
                  f'{np.nanstd(L):.2f}', f'{np.nanstd(R):.2f}'])
    print(k, r['파일'], f'{xL:.2f} {xR:.2f} {xR - xL:.2f} {yB:.2f}', flush=True)
