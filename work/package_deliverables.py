from pathlib import Path
import csv
import json
import zipfile
import importlib.metadata
import shutil

root=Path(__file__).resolve().parent.parent
out=root/'outputs'
work=root/'work'
# Move superseded calculations out of the deliverables directory. Both
# directories resolve within this task workspace; preserve all old files.
obsolete=work/'obsolete_results'
obsolete.mkdir(exist_ok=True)
for name in ['paper_swing_model.xml','paper_swing_motion.csv','paper_swing_result.json',
             'swing_calculation.json','swing_motion.csv','swing_verification.json']:
    source=(out/name).resolve()
    target=(obsolete/name).resolve()
    assert source.is_relative_to(out.resolve()) and target.is_relative_to(work.resolve())
    if source.exists():shutil.move(str(source),str(target))

stems=['swing-180-time','swing-rope-vs-rod','swing-assumption-comparison','swing-body-motion','swing-assumptions']
names=[f'{s}.{ext}' for s in stems for ext in ['png','svg']]
names+=['swing-180-calculation.md','swing-180-summary.csv','swing-180-results.json',
        'rod-ideal-data.csv','rod-with-resistance-data.csv','rope-ideal-data.csv','rope-with-resistance-data.csv']
public=json.loads((out/'swing-180-results.json').read_text(encoding='utf-8'))
checks={}
for name,case in public['main_cases'].items():
    with (out/f'{name}-data.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert rows
    assert all(float(rows[i]['time_s'])<float(rows[i+1]['time_s']) for i in range(len(rows)-1))
    assert case['max_hip_torque_Nm']<=800.00001 and case['max_knee_torque_Nm']<=400.00001
    assert case['max_hip_speed_rad_s']<=12 and case['max_knee_speed_rad_s']<=12
    if name.startswith('rope'):
        assert all(float(r['rope_tension_N'])>=0 for r in rows)
        assert not case['target_reached']
    else:assert case['target_reached']
    assert abs(min(float(r['foot_height_m']) for r in rows)-case['lowest_foot_m'])<1e-9
    checks[name]={'rows':len(rows),'end_s':float(rows[-1]['time_s']),'force_speed_tension_checks':'passed'}

readme='''θ=180° 도달 시간: 계산 결과 묶음

먼저 swing-180-calculation.md를 읽어 주세요.
PPT 삽입: PNG 또는 SVG 그래프 4개와 가정표 1개를 삽입합니다.
PNG: 3168×1782 (16:9). SVG: 글자를 윤곽선으로 저장한 벡터 파일.
원본 PPT는 수정하지 않았습니다.

주요 결과
봉, 공기·축 저항 없음: 27.1초.
봉, 공기저항·축마찰 포함: 39.5초.
줄 등가모델, 외부 저항 없음: 120초 미도달, 최대 112.1°.
줄 등가모델, 외부 저항 포함: 120초 미도달, 최대 106.3°.

줄은 좌석의 독립 회전을 생략한 장력 전용 탄성 등가모델입니다.
모든 결과는 명시한 가정과 제어 규칙에서의 예측이며 전역 최소시간이나 실측값이 아닙니다.

자료
swing-180-summary.csv: 주요 경우 요약.
*-data.csv: 시간에 따른 상태·에너지·힘·속력.
swing-180-results.json: 입력값·요약·검증.

선택: 재현
Python과 reproduce/requirements.txt의 패키지를 설치한 환경에서
reproduce/run_paper_analysis.py, reproduce/verify_rope_cases.py를 순서대로 실행합니다.
보존성 검사는 reproduce/check_conservation.py입니다.
CSV 마지막 행은 적분 단계 말의 상태이고, 도달 시간은 그 단계 안에서 보간한 값입니다.
현재 좌표·속도·토크 기록의 에너지 일 적분에는 시간 간격에 따른 수치 오차가 있습니다.
reproduce/의 스크립트는 분석 자료를 reproduce/paper_analysis/에 저장합니다.
'''
requirements='\n'.join(f'{package}=={importlib.metadata.version(package)}' for package in ['mujoco','numpy','scipy','matplotlib'])+'\n'
archive=out/'swing-180-results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr('먼저읽기.txt',readme)
    for name in names:
        path=out/name
        assert path.is_file() and path.stat().st_size>0
        z.write(path,name)
    for name in ['simulate_paper.py','run_paper_analysis.py','verify_rope_cases.py','check_conservation.py']:
        z.write(work/name,f'reproduce/{name}')
    z.writestr('reproduce/requirements.txt',requirements)
    z.writestr('delivery-checks.json',json.dumps(checks,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    count=len(z.namelist())
print(json.dumps({'archive_entries':count,'archive_bytes':archive.stat().st_size,'checks':checks},ensure_ascii=False,indent=2))
