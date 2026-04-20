import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from Preprocess.dataCollect import get_annotated_data


PAPER_TABLE5_BERT_HATEXPLAIN = {
    'Attn': {
        'acc': 0.698,
        'macro_f1': 0.687,
        'auroc': 0.851,
        'gmb_sub': 0.807,
        'gmb_bpsn': 0.745,
        'gmb_bnsp': 0.763,
        'iou_f1': 0.120,
        'plaus_token_f1': 0.411,
        'auprc': 0.626,
        'comp': 0.424,
        'suff': 0.160,
    },
    'LIME': {
        'acc': 0.698,
        'macro_f1': 0.687,
        'auroc': 0.851,
        'gmb_sub': 0.807,
        'gmb_bpsn': 0.745,
        'gmb_bnsp': 0.763,
        'iou_f1': 0.112,
        'plaus_token_f1': 0.452,
        'auprc': 0.722,
        'comp': 0.500,
        'suff': 0.004,
    },
}

TABLE5_METRIC_ORDER = [
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

TABLE5_METRIC_LABELS = {
    'acc': 'Acc',
    'macro_f1': 'Macro F1',
    'auroc': 'AUROC',
    'gmb_sub': 'GMB-Sub.',
    'gmb_bpsn': 'GMB-BPSN',
    'gmb_bnsp': 'GMB-BNSP',
    'iou_f1': 'IOU F1',
    'plaus_token_f1': 'Plausibility Token F1',
    'auprc': 'AUPRC',
    'comp': 'Faithfulness Comp.',
    'suff': 'Faithfulness Suff.',
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def to_pretty(value):
    if value == '':
        return ''
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f'{float(value):.6f}'
    return 'nan'


def extract_explainability_metrics(blob, method_name: str, source_name: str):
    c = blob['classification_scores']
    prf = c['prf']
    iou = blob['iou_scores'][0]
    rat_prf = blob['rationale_prf']
    tok_prf = blob['token_prf']
    tok_soft = blob['token_soft_metrics']

    rows = [
        ('Explainability', method_name, 'classification_accuracy', c['accuracy']),
        ('Explainability', method_name, 'classification_macro_f1', prf['macro avg']['f1-score']),
        ('Explainability', method_name, 'classification_weighted_f1', prf['weighted avg']['f1-score']),
        ('Explainability', method_name, 'comprehensiveness', c['comprehensiveness']),
        ('Explainability', method_name, 'sufficiency', c['sufficiency']),
        ('Explainability', method_name, 'iou_macro_f1@0.5', iou['macro']['f1']),
        ('Explainability', method_name, 'iou_micro_f1@0.5', iou['micro']['f1']),
        ('Explainability', method_name, 'rationale_instance_macro_f1', rat_prf['instance_macro']['f1']),
        ('Explainability', method_name, 'rationale_instance_micro_f1', rat_prf['instance_micro']['f1']),
        ('Explainability', method_name, 'token_instance_macro_f1', tok_prf['instance_macro']['f1']),
        ('Explainability', method_name, 'token_instance_micro_f1', tok_prf['instance_micro']['f1']),
        ('Explainability', method_name, 'token_auprc', tok_soft['auprc']),
        ('Explainability', method_name, 'token_average_precision', tok_soft['average_precision']),
        ('Explainability', method_name, 'token_roc_auc', tok_soft['roc_auc_score']),
    ]

    records = []
    for section, method, metric, value in rows:
        records.append(
            {
                'section': section,
                'method': method,
                'metric': metric,
                'your_value': float(value),
                'paper_value': '',
                'delta_to_paper': '',
                'source_file': source_name,
            }
        )
    return records


def compute_multiclass_auroc_from_explanations(path: Path, class_order):
    y_true = []
    y_scores = []

    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rationales = row.get('rationales', [])
            if not rationales or 'truth' not in rationales[0]:
                continue

            truth = int(rationales[0]['truth'])
            score_map = row.get('classification_scores', {})
            if not all(label in score_map for label in class_order):
                continue

            y_true.append(truth)
            y_scores.append([float(score_map[label]) for label in class_order])

    if not y_true:
        return float('nan')

    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    y_one_hot = np.eye(len(class_order))[y_true]
    try:
        return float(roc_auc_score(y_one_hot, y_scores, multi_class='ovr', average='macro'))
    except Exception:
        return float('nan')


def generate_target_information(dataset):
    final_target_output = defaultdict(list)
    all_communities_selected = []

    for _, row in dataset.iterrows():
        all_targets = row['target1'] + row['target2'] + row['target3']
        community_dict = dict(Counter(all_targets))

        for key in community_dict:
            if community_dict[key] > 1:
                final_target_output[row['post_id']].append(key)
                all_communities_selected.append(key)

        if row['post_id'] not in final_target_output:
            final_target_output[row['post_id']].append('None')
            all_communities_selected.append('None')

    return final_target_output, all_communities_selected


def convert_to_score(label_name, label_dict):
    if label_name == 'non-toxic':
        return 1 - label_dict[label_name]
    return label_dict[label_name]


def to_binary_label(raw_label):
    if raw_label == 'normal':
        return 'non-toxic'
    if raw_label in ('hatespeech', 'offensive'):
        return 'toxic'
    return None


def bias_evaluation_metric(dataset, method, community):
    positive_ids = []
    negative_ids = []

    if method == 'subgroup':
        for _, row in dataset.iterrows():
            cats = row['final_target_category']
            label = to_binary_label(row['final_label'])
            if cats is None or label is None:
                continue
            if community in cats:
                if label == 'non-toxic':
                    negative_ids.append(row['post_id'])
                else:
                    positive_ids.append(row['post_id'])
    elif method == 'bpsn':
        for _, row in dataset.iterrows():
            cats = row['final_target_category']
            label = to_binary_label(row['final_label'])
            if cats is None or label is None:
                continue
            if community in cats:
                if label == 'non-toxic':
                    negative_ids.append(row['post_id'])
            else:
                if label != 'non-toxic':
                    positive_ids.append(row['post_id'])
    elif method == 'bnsp':
        for _, row in dataset.iterrows():
            cats = row['final_target_category']
            label = to_binary_label(row['final_label'])
            if cats is None or label is None:
                continue
            if community in cats:
                if label != 'non-toxic':
                    positive_ids.append(row['post_id'])
            else:
                if label == 'non-toxic':
                    negative_ids.append(row['post_id'])

    return {'positiveID': positive_ids, 'negativeID': negative_ids}


def maybe_delta(your_value, paper_value):
    if not isinstance(your_value, (int, float, np.floating)):
        return ''
    if not isinstance(paper_value, (int, float, np.floating)):
        return ''
    if not np.isfinite(float(your_value)) or not np.isfinite(float(paper_value)):
        return ''
    return float(your_value) - float(paper_value)


def write_table5_side_by_side(root: Path, our_table5):
    csv_path = root / 'paper_table5_bert_hatexplain_side_by_side.csv'
    fieldnames = ['token_method']
    for metric in TABLE5_METRIC_ORDER:
        fieldnames.extend([f'{metric}_paper', f'{metric}_yours', f'{metric}_delta'])

    rows = []
    for token_method in ['Attn', 'LIME']:
        row = {'token_method': token_method}
        for metric in TABLE5_METRIC_ORDER:
            paper_value = PAPER_TABLE5_BERT_HATEXPLAIN[token_method][metric]
            your_value = our_table5[token_method][metric]
            row[f'{metric}_paper'] = paper_value
            row[f'{metric}_yours'] = your_value
            row[f'{metric}_delta'] = maybe_delta(your_value, paper_value)
        rows.append(row)

    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / 'paper_table5_bert_hatexplain_side_by_side.md'
    with md_path.open('w') as f:
        f.write('# Table 5 Side-By-Side (BERT-HateXplain)\n\n')
        f.write('Delta is computed as Your Value - Paper Value.\n\n')
        for token_method in ['Attn', 'LIME']:
            f.write(f'## {token_method}\n\n')
            f.write('| Metric | Paper | Yours | Delta |\n')
            f.write('|---|---:|---:|---:|\n')
            for metric in TABLE5_METRIC_ORDER:
                paper_value = PAPER_TABLE5_BERT_HATEXPLAIN[token_method][metric]
                your_value = our_table5[token_method][metric]
                delta_value = maybe_delta(your_value, paper_value)
                f.write(
                    f"| {TABLE5_METRIC_LABELS[metric]} | {to_pretty(paper_value)} | {to_pretty(your_value)} | {to_pretty(delta_value)} |\n"
                )
            f.write('\n')

    return csv_path, md_path


def main():
    root = Path(__file__).resolve().parent.parent

    rational_score_path = root / 'model_explain_output_rational_full_0.001.json'
    lime_score_path = root / 'model_explain_output_lime_full_0.001.json'
    rational_expl_path = root / 'explanations_dicts' / 'bestModel_bert_base_uncased_Attn_train_TRUE_0.001_explanation_top5.json'
    lime_expl_path = root / 'explanations_dicts' / 'bestModel_bert_base_uncased_Attn_train_TRUE_explanation_with_lime_10_0.001.json'
    bias_pred_path = root / 'explanations_dicts' / 'bestModel_bert_base_uncased_Attn_train_TRUE_bias.json'

    rational = load_json(rational_score_path)
    lime = load_json(lime_score_path)

    class_order = [str(x) for x in np.load(str(root / 'Data' / 'classes.npy'), allow_pickle=True).tolist()]
    rationale_auroc = compute_multiclass_auroc_from_explanations(rational_expl_path, class_order)
    lime_auroc = compute_multiclass_auroc_from_explanations(lime_expl_path, class_order)

    params = {
        'num_classes': 2,
        'data_file': str(root / 'Data' / 'dataset.json'),
        'class_names': str(root / 'Data' / 'classes_two.npy'),
    }
    data_all_labelled = get_annotated_data(params)

    target_information, all_communities_selected = generate_target_information(data_all_labelled)
    community_count_dict = Counter(all_communities_selected)
    community_count_dict.pop('None', None)
    community_count_dict.pop('Other', None)
    list_selected_community = [community for community, _ in community_count_dict.most_common(10)]

    final_target_information = {}
    for post_id in target_information:
        temp = list(set(target_information[post_id]) & set(list_selected_community))
        if len(temp) == 0:
            final_target_information[post_id] = None
        else:
            final_target_information[post_id] = temp

    data_all_labelled = data_all_labelled.copy()
    data_all_labelled['final_target_category'] = data_all_labelled['post_id'].map(final_target_information)

    with (root / 'Data' / 'post_id_divisions.json').open() as fp:
        post_id_dict = json.load(fp)
    data_all_labelled_bias = data_all_labelled[data_all_labelled['post_id'].isin(post_id_dict['test'])]

    total_data = {}
    with bias_pred_path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total_data[data['annotation_id']] = data

    final_bias_dictionary = defaultdict(dict)
    method_list = ['subgroup', 'bpsn', 'bnsp']
    for each_method in method_list:
        for each_community in list_selected_community:
            community_data = bias_evaluation_metric(data_all_labelled_bias, each_method, each_community)
            truth_values = []
            prediction_values = []

            for pid in community_data['positiveID']:
                if pid in total_data:
                    truth_values.append(1.0)
                    prediction_values.append(
                        convert_to_score(total_data[pid]['classification'], total_data[pid]['classification_scores'])
                    )

            for pid in community_data['negativeID']:
                if pid in total_data:
                    truth_values.append(0.0)
                    prediction_values.append(
                        convert_to_score(total_data[pid]['classification'], total_data[pid]['classification_scores'])
                    )

            if len(truth_values) > 1 and len(set(truth_values)) > 1:
                roc_output_value = roc_auc_score(truth_values, prediction_values)
                final_bias_dictionary[each_method][each_community] = float(roc_output_value)

    power_value = -5
    bias_gmb = {}
    for each_method in method_list:
        vals = list(final_bias_dictionary[each_method].values())
        if vals:
            bias_gmb[each_method] = float(np.power(np.sum(np.power(vals, power_value)) / len(vals), 1 / power_value))
        else:
            bias_gmb[each_method] = float('nan')

    records = []
    records.extend(extract_explainability_metrics(rational, 'Rationale', str(rational_score_path.relative_to(root))))
    records.extend(extract_explainability_metrics(lime, 'LIME', str(lime_score_path.relative_to(root))))
    records.append(
        {
            'section': 'Explainability',
            'method': 'Rationale',
            'metric': 'classification_auroc',
            'your_value': rationale_auroc,
            'paper_value': '',
            'delta_to_paper': '',
            'source_file': str(rational_expl_path.relative_to(root)),
        }
    )
    records.append(
        {
            'section': 'Explainability',
            'method': 'LIME',
            'metric': 'classification_auroc',
            'your_value': lime_auroc,
            'paper_value': '',
            'delta_to_paper': '',
            'source_file': str(lime_expl_path.relative_to(root)),
        }
    )
    for method in method_list:
        records.append(
            {
                'section': 'Bias',
                'method': 'BERT-HateXplain',
                'metric': f'{method}_gmb_p-5',
                'your_value': bias_gmb[method],
                'paper_value': '',
                'delta_to_paper': '',
                'source_file': str(bias_pred_path.relative_to(root)),
            }
        )

    paper_value_map = {
        ('Rationale', 'classification_accuracy'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['acc'],
        ('Rationale', 'classification_macro_f1'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['macro_f1'],
        ('Rationale', 'classification_auroc'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['auroc'],
        ('Rationale', 'iou_macro_f1@0.5'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['iou_f1'],
        ('Rationale', 'token_instance_macro_f1'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['plaus_token_f1'],
        ('Rationale', 'token_auprc'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['auprc'],
        ('Rationale', 'comprehensiveness'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['comp'],
        ('Rationale', 'sufficiency'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['suff'],
        ('LIME', 'classification_accuracy'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['acc'],
        ('LIME', 'classification_macro_f1'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['macro_f1'],
        ('LIME', 'classification_auroc'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['auroc'],
        ('LIME', 'iou_macro_f1@0.5'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['iou_f1'],
        ('LIME', 'token_instance_macro_f1'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['plaus_token_f1'],
        ('LIME', 'token_auprc'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['auprc'],
        ('LIME', 'comprehensiveness'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['comp'],
        ('LIME', 'sufficiency'): PAPER_TABLE5_BERT_HATEXPLAIN['LIME']['suff'],
        ('BERT-HateXplain', 'subgroup_gmb_p-5'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['gmb_sub'],
        ('BERT-HateXplain', 'bpsn_gmb_p-5'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['gmb_bpsn'],
        ('BERT-HateXplain', 'bnsp_gmb_p-5'): PAPER_TABLE5_BERT_HATEXPLAIN['Attn']['gmb_bnsp'],
    }

    for record in records:
        key = (record['method'], record['metric'])
        if key in paper_value_map:
            paper_value = paper_value_map[key]
            record['paper_value'] = paper_value
            record['delta_to_paper'] = maybe_delta(record['your_value'], paper_value)

    records_sorted = sorted(records, key=lambda x: (x['section'], x['method'], x['metric']))

    long_csv = root / 'paper_comparison_table_latest.csv'
    with long_csv.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['section', 'method', 'metric', 'your_value', 'paper_value', 'delta_to_paper', 'source_file'],
        )
        writer.writeheader()
        writer.writerows(records_sorted)

    wide = {
        'model': 'BERT-HateXplain',
        'rationale_accuracy': rational['classification_scores']['accuracy'],
        'rationale_macro_f1': rational['classification_scores']['prf']['macro avg']['f1-score'],
        'rationale_auroc': rationale_auroc,
        'rationale_weighted_f1': rational['classification_scores']['prf']['weighted avg']['f1-score'],
        'rationale_comprehensiveness': rational['classification_scores']['comprehensiveness'],
        'rationale_sufficiency': rational['classification_scores']['sufficiency'],
        'rationale_iou_macro_f1': rational['iou_scores'][0]['macro']['f1'],
        'rationale_token_macro_f1': rational['token_prf']['instance_macro']['f1'],
        'rationale_token_auprc': rational['token_soft_metrics']['auprc'],
        'lime_accuracy': lime['classification_scores']['accuracy'],
        'lime_macro_f1': lime['classification_scores']['prf']['macro avg']['f1-score'],
        'lime_auroc': lime_auroc,
        'lime_weighted_f1': lime['classification_scores']['prf']['weighted avg']['f1-score'],
        'lime_comprehensiveness': lime['classification_scores']['comprehensiveness'],
        'lime_sufficiency': lime['classification_scores']['sufficiency'],
        'lime_iou_macro_f1': lime['iou_scores'][0]['macro']['f1'],
        'lime_token_macro_f1': lime['token_prf']['instance_macro']['f1'],
        'lime_token_auprc': lime['token_soft_metrics']['auprc'],
        'bias_subgroup_gmb_p-5': bias_gmb['subgroup'],
        'bias_bpsn_gmb_p-5': bias_gmb['bpsn'],
        'bias_bnsp_gmb_p-5': bias_gmb['bnsp'],
    }

    wide_csv = root / 'paper_comparison_table_wide_latest.csv'
    with wide_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(wide.keys()))
        writer.writeheader()
        writer.writerow(wide)

    table5_our_values = {
        'Attn': {
            'acc': wide['rationale_accuracy'],
            'macro_f1': wide['rationale_macro_f1'],
            'auroc': wide['rationale_auroc'],
            'gmb_sub': wide['bias_subgroup_gmb_p-5'],
            'gmb_bpsn': wide['bias_bpsn_gmb_p-5'],
            'gmb_bnsp': wide['bias_bnsp_gmb_p-5'],
            'iou_f1': wide['rationale_iou_macro_f1'],
            'plaus_token_f1': wide['rationale_token_macro_f1'],
            'auprc': wide['rationale_token_auprc'],
            'comp': wide['rationale_comprehensiveness'],
            'suff': wide['rationale_sufficiency'],
        },
        'LIME': {
            'acc': wide['lime_accuracy'],
            'macro_f1': wide['lime_macro_f1'],
            'auroc': wide['lime_auroc'],
            'gmb_sub': wide['bias_subgroup_gmb_p-5'],
            'gmb_bpsn': wide['bias_bpsn_gmb_p-5'],
            'gmb_bnsp': wide['bias_bnsp_gmb_p-5'],
            'iou_f1': wide['lime_iou_macro_f1'],
            'plaus_token_f1': wide['lime_token_macro_f1'],
            'auprc': wide['lime_token_auprc'],
            'comp': wide['lime_comprehensiveness'],
            'suff': wide['lime_sufficiency'],
        },
    }
    table5_csv_path, table5_md_path = write_table5_side_by_side(root, table5_our_values)

    md_path = root / 'paper_comparison_table_latest.md'
    with md_path.open('w') as f:
        f.write('# Paper Comparison Table (Latest Run vs Table 5)\n\n')
        f.write('Delta is computed as Your Value - Paper Value.\n\n')
        for method_name in ['Rationale', 'LIME']:
            paper_key = 'Attn' if method_name == 'Rationale' else 'LIME'
            f.write(f'## {method_name}\n\n')
            f.write('| Metric | Paper | Your Value | Delta |\n')
            f.write('|---|---:|---:|---:|\n')

            if method_name == 'Rationale':
                values = [
                    ('Acc', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['acc'], wide['rationale_accuracy']),
                    ('Macro F1', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['macro_f1'], wide['rationale_macro_f1']),
                    ('AUROC', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['auroc'], wide['rationale_auroc']),
                    ('GMB-Sub.', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['gmb_sub'], wide['bias_subgroup_gmb_p-5']),
                    ('GMB-BPSN', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['gmb_bpsn'], wide['bias_bpsn_gmb_p-5']),
                    ('GMB-BNSP', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['gmb_bnsp'], wide['bias_bnsp_gmb_p-5']),
                    ('IOU F1', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['iou_f1'], wide['rationale_iou_macro_f1']),
                    (
                        'Plausibility Token F1',
                        PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['plaus_token_f1'],
                        wide['rationale_token_macro_f1'],
                    ),
                    ('AUPRC', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['auprc'], wide['rationale_token_auprc']),
                    ('Faithfulness Comp.', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['comp'], wide['rationale_comprehensiveness']),
                    ('Faithfulness Suff.', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['suff'], wide['rationale_sufficiency']),
                ]
            else:
                values = [
                    ('Acc', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['acc'], wide['lime_accuracy']),
                    ('Macro F1', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['macro_f1'], wide['lime_macro_f1']),
                    ('AUROC', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['auroc'], wide['lime_auroc']),
                    ('GMB-Sub.', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['gmb_sub'], wide['bias_subgroup_gmb_p-5']),
                    ('GMB-BPSN', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['gmb_bpsn'], wide['bias_bpsn_gmb_p-5']),
                    ('GMB-BNSP', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['gmb_bnsp'], wide['bias_bnsp_gmb_p-5']),
                    ('IOU F1', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['iou_f1'], wide['lime_iou_macro_f1']),
                    (
                        'Plausibility Token F1',
                        PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['plaus_token_f1'],
                        wide['lime_token_macro_f1'],
                    ),
                    ('AUPRC', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['auprc'], wide['lime_token_auprc']),
                    ('Faithfulness Comp.', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['comp'], wide['lime_comprehensiveness']),
                    ('Faithfulness Suff.', PAPER_TABLE5_BERT_HATEXPLAIN[paper_key]['suff'], wide['lime_sufficiency']),
                ]

            for metric_name, paper_value, your_value in values:
                delta_value = maybe_delta(your_value, paper_value)
                f.write(
                    f'| {metric_name} | {to_pretty(paper_value)} | {to_pretty(your_value)} | {to_pretty(delta_value)} |\n'
                )
            f.write('\n')

    print('created', long_csv)
    print('created', wide_csv)
    print('created', md_path)
    print('created', table5_csv_path)
    print('created', table5_md_path)


if __name__ == '__main__':
    main()
