from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
for relative in ['code/outputs','report/figs','submission']:
    p=root/relative
    if p.exists():shutil.rmtree(p)
for relative in ['report/report.pdf','report/generated.md']:
    (root/relative).unlink(missing_ok=True)
