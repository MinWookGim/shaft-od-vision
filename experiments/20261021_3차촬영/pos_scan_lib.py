import numpy as np
def cross(p):
    # 양 끝 10% 평균의 중간값(50%)을 처음 지나는 지점, 선형보간
    n = len(p); a, b = p[:n//10].mean(), p[-n//10:].mean(); h = (a + b) / 2
    s = np.where(np.diff(np.sign(p - h)) != 0)[0]
    if len(s) == 0: return np.nan
    i = s[0]; return i + (h - p[i]) / (p[i+1] - p[i])
