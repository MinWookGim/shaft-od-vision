# 사진마다 샤프트 위/아래 경계 y, 샤프트 끝 x, 핀 왼쪽 경계 x 를 서브픽셀로 -> 재장착(놓임 변화) 판별
import sys, glob, os, cv2, numpy as np
cv2.setNumThreads(8)
d = sys.argv[1]
def cross(p, rising):
    # p: 1D 밝기 프로파일. 양 끝 10% 평균을 밝음/어두움 기준으로 50% 지점을 선형보간
    n = len(p); a, b = p[:n//10].mean(), p[-n//10:].mean(); h = (a + b) / 2
    s = np.where(np.diff(np.sign(p - h)) != 0)[0]
    if len(s) == 0: return np.nan
    i = s[0] if rising == (b > a) else s[0]
    return i + (h - p[i]) / (p[i+1] - p[i])
print('photo   top_y     bot_y     diam     end_x     pin_x    gap')
for f in sorted(glob.glob(os.path.join(d, '2026*_??????.jpg'))):
    g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if g.shape[1] != 16320: print(os.path.basename(f)[9:15], 'skip', g.shape); continue
    g = g.astype(np.float32)
    top = 3300 + cross(g[3300:3800, 9000:9800].mean(1), False)
    bot = 3900 + cross(g[3900:4400, 9000:9800].mean(1), True)
    end = 9900 + cross(g[3700:4000, 9900:10300].mean(0), True)
    pin = 10250 + cross(g[3700:4000, 10250:10500].mean(0), False)
    print(f'{os.path.basename(f)[9:15]}  {top:8.2f}  {bot:8.2f}  {bot-top:7.2f}  {end:8.2f}  {pin:8.2f}  {pin-end:6.2f}', flush=True)
