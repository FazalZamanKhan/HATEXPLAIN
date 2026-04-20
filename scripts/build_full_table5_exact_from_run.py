import csv
from pathlib import Path


def to_float_or_none(value):
    if value is None:
        return None
    value = value.strip()
    if value == '' or value.lower() == 'nan':
        return None
    return float(value)


def main():
    root = Path(__file__).resolve().parent.parent
    wide_path = root / 'paper_comparison_table_wide_latest.csv'

    # Table 5 rows in paper order
    rows = [
        ('CNN-GRU', 'LIME'),
        ('BiRNN', 'LIME'),
        ('BiRNN-Attn', 'Attn'),
        ('BiRNN-Attn', 'LIME'),
        ('BiRNN-HateXplain', 'Attn'),
        ('BiRNN-HateXplain', 'LIME'),
        ('BERT', 'Attn'),
        ('BERT', 'LIME'),
        ('BERT-HateXplain', 'Attn'),
        ('BERT-HateXplain', 'LIME'),
    ]

    metrics = [
        'acc',
        'macro_f1',
        'auroc',
        'gmb_sub',
        'gmb_bpsn',
        'gmb_bnsp',
        'iou_f1',
        'plaus_token_f1',
        'auprc',
        'comp',
        'suff',
    ]

    with wide_path.open() as f:
        wide = next(csv.DictReader(f))

    exact = {
        ('BERT-HateXplain', 'Attn'): {
            'acc': to_float_or_none(wide.get('rationale_accuracy')),
            'macro_f1': to_float_or_none(wide.get('rationale_macro_f1')),
            'auroc': to_float_or_none(wide.get('rationale_auroc')),
            'gmb_sub': to_float_or_none(wide.get('bias_subgroup_gmb_p-5')),
            'gmb_bpsn': to_float_or_none(wide.get('bias_bpsn_gmb_p-5')),
            'gmb_bnsp': to_float_or_none(wide.get('bias_bnsp_gmb_p-5')),
            'iou_f1': to_float_or_none(wide.get('rationale_iou_macro_f1')),
            'plaus_token_f1': to_float_or_none(wide.get('rationale_token_macro_f1')),
            'auprc': to_float_or_none(wide.get('rationale_token_auprc')),
            'comp': to_float_or_none(wide.get('rationale_comprehensiveness')),
            'suff': to_float_or_none(wide.get('rationale_sufficiency')),
        },
        ('BERT-HateXplain', 'LIME'): {
            'acc': to_float_or_none(wide.get('lime_accuracy')),
            'macro_f1': to_float_or_none(wide.get('lime_macro_f1')),
            'auroc': to_float_or_none(wide.get('lime_auroc')),
            'gmb_sub': to_float_or_none(wide.get('bias_subgroup_gmb_p-5')),
            'gmb_bpsn': to_float_or_none(wide.get('bias_bpsn_gmb_p-5')),
            'gmb_bnsp': to_float_or_none(wide.get('bias_bnsp_gmb_p-5')),
            'iou_f1': to_float_or_none(wide.get('lime_iou_macro_f1')),
            'plaus_token_f1': to_float_or_none(wide.get('lime_token_macro_f1')),
            'auprc': to_float_or_none(wide.get('lime_token_auprc')),
            'comp': to_float_or_none(wide.get('lime_comprehensiveness')),
            'suff': to_float_or_none(wide.get('lime_sufficiency')),
        },
    }

    full_csv = root / 'paper_table5_full_exact_from_run.csv'
    with full_csv.open('w', newline='') as f:
        fieldnames = ['model', 'token_method'] + metrics + ['availability_note']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model, token_method in rows:
            out_row = {'model': model, 'token_method': token_method}
            values = exact.get((model, token_method))

            if values is None:
                for m in metrics:
                    out_row[m] = 'N/A'
                out_row['availability_note'] = 'No run artifacts found in workspace for this row'
            else:
                for m in metrics:
                    v = values[m]
                    out_row[m] = f'{v:.6f}' if v is not None else 'N/A'
                if values['auroc'] is None:
                    out_row['availability_note'] = (
                        'AUROC unavailable from current run artifacts (class support issue)'
                    )
                else:
                    out_row['availability_note'] = 'Exact from run artifacts'

            writer.writerow(out_row)

    full_md = root / 'paper_table5_full_exact_from_run.md'
    with full_md.open('w') as f:
        f.write('# Table 5 (Exact Values From Current Run Artifacts)\n\n')
        f.write('Rows not executed in this workspace are marked as N/A.\n\n')
        f.write(
            '| Model | Token Method | Acc | Macro F1 | AUROC | GMB-Sub. | GMB-BPSN | GMB-BNSP | '
            'IOU F1 | Plausibility Token F1 | AUPRC | Comp. | Suff. | Note |\n'
        )
        f.write('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')

        for model, token_method in rows:
            values = exact.get((model, token_method))
            if values is None:
                row_vals = ['N/A'] * len(metrics)
                note = 'No run artifacts found in workspace for this row'
            else:
                row_vals = []
                for m in metrics:
                    v = values[m]
                    row_vals.append(f'{v:.6f}' if v is not None else 'N/A')
                note = 'Exact from run artifacts'
                if values['auroc'] is None:
                    note = 'AUROC unavailable from current run artifacts (class support issue)'

            f.write('| ' + ' | '.join([model, token_method] + row_vals + [note]) + ' |\n')

    print('created', full_csv)
    print('created', full_md)


if __name__ == '__main__':
    main()
