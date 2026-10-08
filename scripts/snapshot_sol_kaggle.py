"""Read-only authenticated Kaggle snapshot; persist only explicit safe fields."""
import argparse
import contextlib
import io
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.submission.kaggle_io import _api,COMPETITION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError('Preserve the existing snapshot')
    api=_api()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        competitions=api.competitions_list(search=COMPETITION,page_size=20).competitions
        leaders=api.competition_leaderboard_view(COMPETITION,page_size=5) or []
        submissions=api.competition_submissions(COMPETITION,page_size=100) or []
        limits=api.competition_get_submission_limits(COMPETITION)
    matches=[c for c in competitions if c.ref.endswith('/'+COMPETITION)]
    assert len(matches)==1
    c=matches[0]
    result={'utc':datetime.now(timezone.utc).isoformat(),'source':'authenticated Kaggle SDK; whitelisted fields',
        'competition':{'title':c.title,'deadline':str(c.deadline),'entered':c.user_has_entered,
                       'user_rank':c.user_rank,'team_count':c.team_count},
        'remaining_now':limits.num_allowed_now,
        'top5':[{'team_id':s.team_id,'team_name':s.team_name,'score':s.score} for s in leaders if s],
        'submissions':[{'ref':str(s.ref),'file':s.file_name,'status':s.status.name,
                        'date':str(s.date),'public_score':s.public_score} for s in submissions if s],
        'selection_role':'Transfer diagnostic; never selects weights or parameters'}
    path.write_text(json.dumps(result,indent=2)+ '\n',encoding='utf-8')
    print(json.dumps({'utc':result['utc'],'user_rank':c.user_rank,'team_count':c.team_count,
                     'remaining_now':result['remaining_now'],'top_public':result['top5'][0]['score'] if result['top5'] else None}))
    return 0


if __name__=='__main__':raise SystemExit(main())
