# ③ 재장착 반복성 11.6µm(내) vs 5.2µm(팀원) 가르기: 정의 2가지 x 경계 원천 4가지
#   내 정의: 본시험 사진 사이 sd = 12시편 분산 평균의 제곱근, 측정 1회 sd = 재장착 한 장(5번 놓임) 분산 평균의 제곱근 (NO7~12)
#   팀원 정의: 사진 사이 = 정상 9시편 10장 sd 의 중앙값, 재장착 = 6번 놓임(본시험 놓임은 10장 중앙값 + 재장착 5장) sd 의 시편 중앙값 (NO7~12)
#   배율은 원천마다 16 / 핀 px 중앙값 (원래 내 grr.py 는 20.02/750 = 26.69µm/px 이었음)
import numpy as np, pandas as pd, sys
g = pd.read_csv(sys.argv[1], encoding='utf-8-sig'); g['file'] = g.filename
loc = pd.read_csv('../1021_main/local_edges.csv')
rc = pd.read_csv('../1027_repr/repr_cols.csv')
XS = np.linspace(8030, 8320, 31).round().astype(int); CL = [i for i, x in enumerate(XS) if 8060 <= x <= 8270]
D = rc[[f'b{i}' for i in range(31)]].to_numpy() - rc[[f't{i}' for i in range(31)]].to_numpy()
rc['mean31'] = np.nanmean(D, 1); rc['mean21'] = np.nanmean(D[:, CL], 1)
b = loc[['file', 'stage', 'no', 'pin', 'shaft']].merge(rc[['file', 'mean31', 'mean21']], on='file').merge(
    g[['file', 'bearing_cross50_px', 'pin_cross50_px']], on='file')
b['no'] = b.no.str[2:].astype(int); b = b[b.no != 1].sort_values('file')
b['mount'] = np.where(b.stage == 2, b.groupby(['stage', 'no']).cumcount() + 1, 0)
SRC = {'팀원 cross50': ('bearing_cross50_px', 'pin_cross50_px'), '내 32열 중앙값': ('shaft', 'pin'),
       '내 31열 평균': ('mean31', 'pin'), '내 21열 평균': ('mean21', 'pin')}
rows = []
for name, (sc, pc) in SRC.items():
    um = 16 / b[pc].median() * 1000; x = b[sc] * um
    main, rem = b.stage == 1, b.stage == 2
    mine_photo = np.sqrt(x[main].groupby(b.no[main]).var().mean())
    mine_1shot = np.sqrt(x[rem].groupby(b.no[rem]).var().mean())
    gon_photo = x[main & b.no.between(2, 10)].groupby(b.no).std().median()
    mv = x.groupby([b.no, b.mount]).median()
    mv = mv[mv.index.get_level_values(0) >= 7]
    gon_rem = mv.groupby(level=0).std().median()
    # 놓임 몫만(사진 몫 뺀 것): 재장착 한 장 분산 - 사진 분산
    mount_only = np.sqrt(max(mine_1shot ** 2 - mine_photo ** 2, 0))
    rows.append(dict(경계=name, 배율=round(um, 2), 사진사이_내정의=round(mine_photo, 1), 사진사이_팀원정의=round(gon_photo, 1),
                     재장착1장_내정의=round(mine_1shot, 1), 재장착6놓임_팀원정의=round(gon_rem, 1), 놓임몫=round(mount_only, 1),
                     GRR_내정의=f'{6 * mine_1shot / 20 * 100:.0f}%', GRR_팀원값=f'{6 * gon_rem / 20 * 100:.0f}%'))
pd.set_option('display.width', 250)
r = pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv('remount_cross.csv', index=False)
# 시편별 사진 사이 sd (팀원 정의의 중앙값이 분산 평균보다 작은 이유 확인)
for name, (sc, pc) in SRC.items():
    um = 16 / b[pc].median() * 1000; x = b[sc] * um
    print(name, '시편별 본시험 10장 sd:', ' '.join(f'{v:.1f}' for v in x[b.stage == 1].groupby(b.no[b.stage == 1]).std()))
