"""Rebuild scoped paper tables from verified, archived per-pair scores."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from evaluation.reporting import summarize

ROOT=Path(__file__).resolve().parent
ROLES=('S','A','T','B','VP','Shared','Lead','Bar')


def load_artifacts(root=ROOT/'reproduction'):
    plan=json.loads((root/'plan.json').read_text())
    source=root/plan['source_file']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['source_sha256']:
        raise ValueError('Frozen per-pair artifact checksum mismatch')
    rows=list(csv.DictReader(io.StringIO(gzip.decompress(source.read_bytes()).decode())))
    inventory=json.loads((ROOT/'manifests/checkpoint_candidates.json').read_text())
    known_hashes={item['sha256'] for item in inventory['checkpoints']}
    groups=defaultdict(list);baselines={}
    for row in rows:
        if row['artifact_id'] not in plan['artifacts']:raise ValueError('Unknown score artifact')
        if row['checkpoint_sha256'] not in known_hashes:raise ValueError('Untracked result checkpoint')
        for name in ('si_sdr','mix_si_sdr','si_sdri'):row[name]=float(row[name])
        key=row['corpus'],row['track_id'],row['role']
        if key in baselines and not math.isclose(baselines[key],row['mix_si_sdr'],rel_tol=0,abs_tol=1e-6):
            raise ValueError('Mixture baseline differs across artifacts: '+str(key))
        baselines[key]=row['mix_si_sdr']
        groups[row['artifact_id']].append(row)
    summaries={}
    for name,meta in plan['artifacts'].items():
        if len(groups[name])!=meta['n_pairs']:raise ValueError('Frozen population count changed: '+name)
        summary=summarize(groups[name])
        if len(summary)!=2:raise ValueError('Artifact mixes corpora or protocols')
        summaries[name]={row['scope']:row for row in summary}
    return plan,groups,summaries


def build_tables(plan,summaries):
    result={}
    for table,entries in plan['tables'].items():
        output=[]
        for entry in entries:
            if 'artifacts' in entry:
                row={k:v for k,v in entry.items() if k!='artifacts'}
                for corpus,name in entry['artifacts'].items():
                    selected=summaries[name][entry['scope']]
                    criterion=(entry.get('validation_selection') or
                        ('combined' if entry['selection']=='Pooled' else entry['selection'].lower())+'_shared')
                    if criterion not in plan['artifacts'][name]['selection_criteria']:
                        raise ValueError('Result is not the frozen validation selection')
                    row[corpus]=selected['overall_si_sdri'];row[corpus+'_n']=selected['n_pairs']
                    if plan['artifacts'][name]['step']!=entry['step']:raise ValueError('Checkpoint step mismatch')
            else:
                name=entry['artifact'];overall=summaries[name]['shared'];inclusive=summaries[name]['inclusive']
                if entry.get('validation_selection') and entry['validation_selection'] not in plan['artifacts'][name]['selection_criteria']:
                    raise ValueError('Invalid target-domain checkpoint selection')
                row={k:v for k,v in entry.items() if k!='unavailable_roles'}
                row['Shared']=overall['overall_si_sdri'];row['shared_n']=overall['n_pairs']
                for role in ('S','A','T','B','VP','Lead','Bar'):
                    item=inclusive['per_role'].get(role)
                    row[role]=None if role in entry['unavailable_roles'] or item is None else item['si_sdri']
            output.append(row)
        result[table]=output
    return result


def draft_discrepancies(draft_cells,plan,tables):
    old=json.loads(Path(draft_cells).read_text());lookup={}
    for row in tables['cross_corpus']:
        meta=plan['artifacts'][row['artifact']]
        model=meta.get('historical_model') or row['artifact'].rsplit('_',2)[0]
        lookup[model,row['test']]=row
    differences=[]
    for cell in old:
        row=lookup.get((cell['model'],cell['dataset']))
        if row is None or row.get(cell['role']) is None:continue
        value=row[cell['role']]
        if float(f'{value:.1f}')!=float(cell['draft_value']):
            differences.append(dict(cell,artifact_value=value,artifact_rounded=f'{value:.1f}',
                note='Historical manuscript differs from the explicitly scoped source artifact; requires author reconciliation'))
    return differences


def write_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def format_value(value):
    if value is None:return '--'
    if isinstance(value,float):return f'{value:.1f}'
    return str(value).replace('_',r'\_').replace('&',r'\&')


def reproduce(output,source=ROOT/'reproduction',draft_cells=None):
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Output must be a new directory')
    plan,groups,summaries=load_artifacts(source);tables=build_tables(plan,summaries)
    differences=draft_discrepancies(draft_cells,plan,tables) if draft_cells else []
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reproduce-',dir=output.parent) as tmp:
        staging=Path(tmp)/'result';staging.mkdir()
        for name,rows in tables.items():
            write_csv(staging/(name+'.csv'),rows)
            if name=='cross_corpus':columns=['artifact','train','test',*ROLES]
            elif name=='target_conditioning':columns=['train','test','condition',*ROLES]
            elif name=='development_a':columns=['train','selection','step','JC','SK']
            else:columns=['train','step','SK','JC','AB']
            lines=['% '+ ' | '.join(columns)]
            lines += [' & '.join(format_value(row.get(c)) for c in columns)+r' \\' for row in rows]
            (staging/(name+'.tex')).write_text('\n'.join(lines)+'\n')
        write_csv(staging/'draft_discrepancies.csv',differences)
        (staging/'summary.json').write_text(json.dumps({
            'status':'ARTIFACT_REPRODUCTION_PASS','manuscript_status':('REQUIRES_RECONCILIATION' if differences else 'MATCHES_CHECKED_CELLS') if draft_cells else 'NOT_COMPARED',
            'source_sha256':plan['source_sha256'],'per_pair_rows':sum(map(len,groups.values())),
            'artifacts':len(groups),'table_rows':{k:len(v) for k,v in tables.items()},
            'historical_draft_discrepancies':len(differences),'limitations':plan['limitations']},indent=2)+'\n')
        staging.rename(output)
    return tables,differences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draft-cells',type=Path,help='Optional JSON manuscript cells for a separate comparison report')
    args=parser.parse_args();tables,differences=reproduce(args.output,draft_cells=args.draft_cells)
    print(f'Wrote {len(tables)} artifact-derived tables.' +
          (f' {len(differences)} draft discrepancies recorded.' if args.draft_cells else ' No manuscript was compared.'))


if __name__=='__main__':main()
