from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

root=Path(__file__).resolve().parents[1]
out=root/'submission';out.mkdir(exist_ok=True)
files=[root/n for n in ['README.md','Makefile','requirements.txt','environment.yml','.gitignore','COL7(6)83Assignment4.pdf']]
files += [p for p in (root/'code').rglob('*') if p.is_file() and not {'outputs','__pycache__'} & set(p.relative_to(root).parts)]
files += [root/'report/report.md',root/'report/make_assets.py',root/'report/study_assets.py']
files += sorted((root/'code/outputs/study').glob('*.csv'))
files += sorted((root/'code/outputs/study').glob('*.json'))
with ZipFile(out/'COL7683_A4_code.zip','w',ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(root))
shutil.copy2(root/'report/report.pdf',out/'COL7683_A4_report.pdf')
print('Packaged',len(files),'source/input files in',out)
