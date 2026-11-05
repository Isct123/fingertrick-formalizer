from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from .parser import parse_algorithm, parse_families
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
CASE_HEADERS = ('Case', 'Type', 'Phase', 'Target reps', 'Exact algorithm you use', 'Habitual physical-family sequence', 'Model-recommended physical sequence', 'Completed OK', 'Average OK time (ms)', 'Notes')
RUN_HEADERS = ('Run #', 'Phase', 'Case', 'Execution', 'Time (ms)', 'Result', 'Session', 'Rep', 'Notes')
HELD_OUT = {f'OLL {i}' for i in (4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 57)} | {f'PLL {s}' for s in ('Aa', 'E', 'Gc', 'Jb', 'Ua', 'V')}
ALL_CASES = {f'OLL {i}' for i in range(1, 58)} | {f'PLL {s}' for s in ('Aa', 'Ab', 'E', 'F', 'Ga', 'Gb', 'Gc', 'Gd', 'H', 'Ja', 'Jb', 'Na', 'Nb', 'Ra', 'Rb', 'T', 'Ua', 'Ub', 'V', 'Y', 'Z')}

@dataclass
class Case:
    case_name: str
    case_type: str
    phase: str
    target_reps: int
    exact_logical_algorithm: str
    habitual_physical_family_sequence: str
    model_recommended_sequence: str
    notes: str = ''

@dataclass
class Run:
    run_number: int
    phase: str
    case_name: str
    execution: str
    time_ms: float | None
    result: str
    session: str
    rep: int
    notes: str

def integer(value, label):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be an integer: {value!r}') from error
    if result < 1:
        raise ValueError(f'{label} must be positive')
    return result

def read_sheets(path):
    result = {}
    with ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS):
            target = rels[sheet.attrib[f'{{{REL}}}id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            rows = []
            for row in ET.fromstring(z.read(target)).findall('m:sheetData/m:row', NS):
                cells = {}
                for c in row.findall('m:c', NS):
                    col = re.match('[A-Z]+', c.attrib['r'])[0]
                    index = 0
                    for letter in col:
                        index = index * 26 + ord(letter) - 64
                    if c.find('m:f', NS) is not None:
                        value = '=' + c.findtext('m:f', '', NS)
                    elif c.attrib.get('t') == 'inlineStr':
                        value = ''.join(c.find('m:is', NS).itertext())
                    else:
                        value = c.findtext('m:v', '', NS)
                        if c.attrib.get('t') == 's':
                            value = shared[int(value)]
                    cells[index - 1] = value
                rows.append([cells.get(i, '') for i in range(max(cells, default=-1) + 1)])
            result[sheet.attrib['name']] = rows
    return result

def records(rows, headers):
    if not rows or tuple(rows[0][:len(headers)]) != headers:
        raise ValueError(f'Unexpected sheet headers; expected {headers}')
    return [dict(zip(headers, row + [''] * (len(headers) - len(row)))) for row in rows[1:] if any(row)]

def load_workbook(path):
    sheets = read_sheets(path)
    cases = []
    for row in records(sheets['Case Plan'], CASE_HEADERS):
        cases.append(Case(row['Case'], row['Type'], row['Phase'], integer(row['Target reps'], 'Target reps'), row[CASE_HEADERS[4]].strip(), row[CASE_HEADERS[5]].strip(), row[CASE_HEADERS[6]].strip(), row['Notes']))
    runs = []
    for row in records(sheets['Run Schedule'], RUN_HEADERS):
        time = float(row['Time (ms)']) if row['Time (ms)'].strip() else None
        if time is not None and (not math.isfinite(time) or time <= 0):
            raise ValueError(f"Run {row['Run #']}: time must be finite positive milliseconds")
        if row['Result'] not in ('', 'OK', 'Lockup', 'DNF'):
            raise ValueError(f"Run {row['Run #']}: unknown result")
        if row['Result'] == 'OK' and time is None:
            raise ValueError(f"Run {row['Run #']}: OK requires a measured time")
        runs.append(Run(integer(row['Run #'], 'Run #'), row['Phase'], row['Case'], row['Execution'], time, row['Result'], row['Session'], integer(row['Rep'], 'Rep'), row['Notes']))
    validate_plan(cases, runs)
    return (cases, runs)

def validate_plan(cases, runs=None):
    names = [c.case_name for c in cases]
    if len(names) != len(set(names)) or set(names) != ALL_CASES:
        raise ValueError('Expected exactly the 57 OLL and 21 PLL cases, without duplicates')
    for c in cases:
        expected_phase = 'HELD-OUT TEST' if c.case_name in HELD_OUT else 'TRAIN'
        if c.phase != expected_phase or c.target_reps != (20 if c.case_name in HELD_OUT else 8):
            raise ValueError(f'{c.case_name}: split/repetition plan differs from the handoff')
        if c.case_type != c.case_name.split()[0]:
            raise ValueError(f'{c.case_name}: incorrect case type')
    if runs is None:
        return
    if len(runs) != 840 or {r.run_number for r in runs} != set(range(1, 841)):
        raise ValueError('Expected 840 unique scheduled run numbers')
    lookup = {c.case_name: c for c in cases}
    repetitions = Counter()
    for r in runs:
        if r.case_name not in lookup or r.phase != lookup[r.case_name].phase:
            raise ValueError(f'Run {r.run_number}: case/phase mismatch')
        if r.execution not in (('Habitual',) if r.phase == 'TRAIN' else ('Habitual', 'Model')):
            raise ValueError(f'Run {r.run_number}: invalid execution group')
        limit = 8 if r.phase == 'TRAIN' else 10
        if r.rep not in range(1, limit + 1):
            raise ValueError(f'Run {r.run_number}: invalid repetition')
        repetitions[r.case_name, r.execution, r.rep] += 1
    expected = {(c.case_name, e, rep) for c in cases for e in (('Habitual',) if c.phase == 'TRAIN' else ('Habitual', 'Model')) for rep in range(1, 9 if c.phase == 'TRAIN' else 11)}
    if set(repetitions) != expected or any((n != 1 for n in repetitions.values())):
        raise ValueError('Missing or duplicated scheduled case/execution/repetition')

def load_cases(path):
    if Path(path).suffix.lower() == '.xlsx':
        return load_workbook(path)[0]
    with open(path, encoding='utf-8-sig', newline='') as f:
        data = list(csv.DictReader(f))
    cases = [Case(**{**row, 'target_reps': integer(row['target_reps'], 'target_reps')}) for row in data]
    validate_plan(cases)
    return cases

def export_plan(path, output):
    cases, runs = load_workbook(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (('case_plan', cases), ('timing_runs', runs)):
        rows = [asdict(v) for v in values]
        with (output / f'{name}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(source=str(path), cases=len(cases), train_cases=sum((c.phase == 'TRAIN' for c in cases)), held_out_cases=sum((c.phase == 'HELD-OUT TEST' for c in cases)), runs=len(runs), run_groups=dict(Counter((f'{r.phase}/{r.execution}' for r in runs))), filled_algorithms=sum((bool(c.exact_logical_algorithm) for c in cases)), filled_habitual_sequences=sum((bool(c.habitual_physical_family_sequence) for c in cases)), results=dict(Counter((r.result or 'unrecorded' for r in runs))))
    (output / 'workbook_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary

def audit_cases(cases):
    from .features import audit
    sequences, included, missing, invalid = ([], [], [], {})
    for c in cases:
        if c.phase != 'TRAIN':
            continue
        if not c.exact_logical_algorithm or not c.habitual_physical_family_sequence:
            missing.append(c.case_name)
            continue
        try:
            parse_algorithm(c.exact_logical_algorithm)
            seq = parse_families(c.habitual_physical_family_sequence)
            if not seq:
                raise ValueError('Empty family sequence')
        except ValueError as error:
            invalid[c.case_name] = str(error)
            continue
        sequences.append(seq)
        included.append(c.case_name)
    report = audit(sequences)
    report.update(included_training_cases=included, missing_training_cases=missing, invalid_training_cases=invalid, annotations_complete=not missing and (not invalid), physical_validation='Not established by family counts; run validate with exact starts and legality.', held_out_excluded=True, full_training_pool_rank_upper_bound=60, full_training_pool_additional_sequence_lower_bound=72)
    return report
