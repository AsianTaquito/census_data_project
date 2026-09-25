"""Streamlit dashboard for the Adult income project.

    streamlit run app.py

Reads whatever `python main.py` last wrote into model_results/.
"""
import warnings
warnings.filterwarnings('ignore')

import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

import data
import models

st.set_page_config(page_title='Adult Income Dashboard', page_icon='chart',
                   layout='wide')

# Validated categorical order (worst adjacent CVD dE 9.1 light / 8.4 dark).
SERIES = {
    'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
              '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    'dark': ['#3987e5', '#d95926', '#199e70', '#c98500',
             '#d55181', '#008300', '#9085e9', '#e66767'],
}
RAMP = {
    'light': ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
    'dark': ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
}


def mode():
    try:
        return 'dark' if st.context.theme.type == 'dark' else 'light'
    except Exception:
        return 'light'


@st.cache_data(show_spinner='Loading census data...')
def get_dissection():
    return data.dissect_data(data.load_data(), verbose=False)


@st.cache_data(show_spinner='Loading results...')
def get_results():
    if not os.path.exists(models.RESULTS_CSV):
        return None, None
    return pd.read_csv(models.RESULTS_CSV), joblib.load(models.CURVES_FILE)


def rate_chart(tbl, base_rate, title, color):
    """Share above $50K per group, with the base rate as a reference rule."""
    d = tbl.reset_index()
    d.columns = ['Group', 'n', 'rate']
    order = list(d['Group'].astype(str))

    bars = alt.Chart(d).mark_bar(height=18, cornerRadiusEnd=4).encode(
        x=alt.X('rate:Q', title='Share earning >$50K',
                axis=alt.Axis(format='.0%')),
        y=alt.Y('Group:N', sort=order, title=None),
        color=alt.value(color),
        tooltip=[alt.Tooltip('Group:N'),
                 alt.Tooltip('rate:Q', title='Share >$50K', format='.1%'),
                 alt.Tooltip('n:Q', title='People', format=',')],
    )
    labels = bars.mark_text(align='left', dx=5, fontSize=11).encode(
        text=alt.Text('rate:Q', format='.0%'), color=alt.value('gray'))
    rule = alt.Chart(pd.DataFrame({'v': [base_rate]})).mark_rule(
        strokeDash=[4, 4], color='gray').encode(x='v:Q')

    return (bars + labels + rule).properties(title=title, height=max(180, 30 * len(d)))


def metric_matrix(results_df, ramp):
    """Models x metrics, shaded by rank within each metric."""
    m = results_df.set_index('Model')[models.METRIC_COLS]
    spread = (m.max() - m.min()).replace(0, np.nan)
    norm = ((m - m.min()) / spread).fillna(0.5)

    long = m.reset_index().melt('Model', var_name='Metric', value_name='Value')
    long['Rank'] = norm.reset_index().melt('Model', var_name='Metric',
                                           value_name='Rank')['Rank']

    base = alt.Chart(long).encode(
        x=alt.X('Metric:N', sort=models.METRIC_COLS, title=None,
                axis=alt.Axis(labelAngle=0, orient='bottom')),
        y=alt.Y('Model:N', sort=list(m.index), title=None),
    )
    cells = base.mark_rect(stroke='white', strokeWidth=2).encode(
        color=alt.Color('Rank:Q', legend=None,
                        scale=alt.Scale(range=ramp, domain=[0, 1])),
        tooltip=[alt.Tooltip('Model:N'), alt.Tooltip('Metric:N'),
                 alt.Tooltip('Value:Q', format='.4f')],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text('Value:Q', format='.3f'),
        color=alt.condition(alt.datum.Rank > 0.55,
                            alt.value('white'), alt.value('black')),
    )
    return (cells + text).properties(height=45 * len(m))


def roc_chart(results_df, curves, colors):
    rows = []
    for name in results_df['Model']:
        score = curves['scores'].get(name)
        if score is None:
            continue
        fpr, tpr, _ = roc_curve(curves['y_test'], score)
        auc = results_df.loc[results_df['Model'] == name, 'ROC-AUC'].iat[0]
        keep = np.linspace(0, len(fpr) - 1, 300).astype(int)
        rows.append(pd.DataFrame({'FPR': fpr[keep], 'TPR': tpr[keep],
                                  'Model': f'{name}  ({auc:.3f})'}))
    if not rows:
        return None

    d = pd.concat(rows)
    curve = alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X('FPR:Q', title='False positive rate'),
        y=alt.Y('TPR:Q', title='True positive rate'),
        color=alt.Color('Model:N', title=None,
                        scale=alt.Scale(range=colors[:d['Model'].nunique()]),
                        legend=alt.Legend(orient='bottom', columns=2, labelLimit=0)),
        tooltip=['Model:N', alt.Tooltip('FPR:Q', format='.3f'),
                 alt.Tooltip('TPR:Q', format='.3f')],
    )
    chance = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(
        strokeDash=[4, 4], color='gray', strokeWidth=1).encode(x='x:Q', y='y:Q')
    return (chance + curve).properties(height=420)


FP_LABEL = 'False positive (said >50K, was not)'
FN_LABEL = 'False negative (missed a >50K earner)'


def error_chart(results_df, colors):
    order = list(results_df.sort_values('F1-macro', ascending=False)['Model'])
    d = results_df.melt('Model', ['FP', 'FN'], var_name='Kind', value_name='Rows')
    d['Kind'] = d['Kind'].map({'FP': FP_LABEL, 'FN': FN_LABEL})
    return alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X('Rows:Q', title='Test-set rows'),
        y=alt.Y('Model:N', sort=order, title=None),
        yOffset=alt.YOffset('Kind:N', sort=[FP_LABEL, FN_LABEL]),
        # pin the colors, or Altair assigns them by alphabetical label
        color=alt.Color('Kind:N', title=None,
                        scale=alt.Scale(domain=[FP_LABEL, FN_LABEL],
                                        range=colors[:2]),
                        legend=alt.Legend(orient='bottom', columns=1, labelLimit=0)),
        tooltip=['Model:N', 'Kind:N', alt.Tooltip('Rows:Q', format=',')],
    ).properties(height=62 * results_df['Model'].nunique())


def cost_chart(results_df, color):
    d = results_df.copy()
    d['Train Time (s)'] = d['Train Time (s)'].clip(lower=0.01)
    # models with equal scores sit on the same point, so alternate the labels
    d['slot'] = d['F1-macro'].rank(method='first').astype(int) % 2

    lo, hi = d['Train Time (s)'].min(), d['Train Time (s)'].max()
    points = alt.Chart(d).mark_circle(size=140, opacity=1).encode(
        # headroom on the right so the point labels are not clipped
        x=alt.X('Train Time (s):Q', title='Training time (s, log scale)',
                scale=alt.Scale(type='log', domain=[lo * 0.6, hi * 6])),
        y=alt.Y('F1-macro:Q', scale=alt.Scale(zero=False)),
        color=alt.value(color),
        tooltip=['Model:N', alt.Tooltip('F1-macro:Q', format='.3f'),
                 alt.Tooltip('Train Time (s):Q', format='.2f')],
    )
    labels = alt.layer(*[
        points.transform_filter(alt.datum.slot == slot)
              .mark_text(align='left', dx=10, dy=dy, fontSize=11)
              .encode(text='Model:N', color=alt.value('gray'))
        for slot, dy in ((0, -9), (1, 10))
    ])
    return (points + labels).properties(height=400)


def no_models():
    """Why the model views are empty: missing results, or nothing selected."""
    if not os.path.exists(models.RESULTS_CSV):
        st.warning(f'No results yet - `{models.RESULTS_CSV}` is missing.')
        st.code('python main.py --no-nn     # quick, two models\n'
                'python main.py             # all seven', language='bash')
    else:
        st.info('Pick at least one model in the sidebar.')


def main():
    theme = mode()
    colors, ramp = SERIES[theme], RAMP[theme]

    st.title('Adult Income Dashboard')
    st.caption('UCI Census data, dissected by the features that move income, '
               'then held up against seven classifiers on the same 20% test split.')

    dissection = get_dissection()
    base_rate = dissection['base_rate']
    results_df, curves = get_results()

    all_models = list(results_df['Model']) if results_df is not None else []
    with st.sidebar:
        st.header('Filters')
        chosen = st.multiselect('Models', all_models, default=all_models,
                                disabled=not all_models)
        st.caption('Applies to the model charts and the table. '
                   'The data views never change with the selection.')
    # None means the model views have nothing to draw; the data views do not care
    shown = (results_df[results_df['Model'].isin(chosen)].reset_index(drop=True)
             if chosen else None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Census records', f"{dissection['n_rows']:,}")
    c2.metric('Earn over $50K', f'{base_rate:.1%}',
              help=f'Always guessing <=50K scores {1 - base_rate:.1%}.')
    if shown is None:
        c3.metric('Best F1-macro', '-')
        c4.metric('Best ROC-AUC', '-')
    else:
        best_f1 = shown.loc[shown['F1-macro'].idxmax()]
        best_auc = shown.loc[shown['ROC-AUC'].idxmax()]
        c3.metric('Best F1-macro', f"{best_f1['F1-macro']:.3f}", best_f1['Model'],
                  delta_color='off')
        c4.metric('Best ROC-AUC', f"{best_auc['ROC-AUC']:.3f}", best_auc['Model'],
                  delta_color='off')

    tab_data, tab_models, tab_table = st.tabs(
        ['The data', 'The models', 'Full results'])

    with tab_data:
        st.subheader('Who earns more than $50K')
        st.write(f'Dashed line is the overall base rate of **{base_rate:.1%}**. '
                 'Groups under 50 rows are dropped.')
        titles = list(dissection['tables'])
        pick = st.radio('Break down by', titles, horizontal=True,
                        label_visibility='collapsed')
        st.altair_chart(
            rate_chart(dissection['tables'][pick], base_rate, pick, colors[0]),
            width='stretch')
        st.caption(
            f"Capital gains are the loudest single flag: "
            f"{dissection['gain_share']:.1%} of rows report one, and they are "
            f"{dissection['gain_lift']:+.1%} more likely to clear $50K than rows without.")

    if shown is None:
        with tab_models:
            no_models()
        with tab_table:
            no_models()
        return

    with tab_models:
        st.subheader('Every metric, every model')
        st.write(f'Accuracy alone is misleading at a {base_rate:.1%} positive rate. '
                 'Balanced accuracy, ROC-AUC, PR-AUC and the raw error counts are '
                 'the ones worth reading. Shading ranks each column separately.')
        st.altair_chart(metric_matrix(shown, ramp), width='stretch')

        left, right = st.columns(2)
        with left:
            st.subheader('ROC curves')
            chart = roc_chart(shown, curves, colors)
            if chart is None:
                st.info('No saved scores for the selected models.')
            else:
                st.altair_chart(chart, width='stretch')
        with right:
            st.subheader('Error profile')
            st.altair_chart(error_chart(shown, colors), width='stretch')

        st.subheader('Is the compute buying anything?')
        st.altair_chart(cost_chart(shown, colors[0]), width='stretch')

        twins = shown[shown.duplicated(models.METRIC_COLS, keep=False)]['Model'].tolist()
        if twins:
            st.info(f"{' and '.join(twins)} scored identically on every metric - "
                    'they converged to the same fitted model, so that '
                    'configuration difference bought nothing.')

    with tab_table:
        st.subheader('Full results')
        st.write('Sorted by F1-macro. Click any header to re-sort.')
        st.dataframe(
            shown.sort_values('F1-macro', ascending=False),
            width='stretch', hide_index=True,
            column_config={c: st.column_config.NumberColumn(c, format='%.3f')
                           for c in models.METRIC_COLS},
        )
        st.download_button('Download CSV', shown.to_csv(index=False),
                           'model_results.csv', 'text/csv')


if __name__ == '__main__':
    if not st.runtime.exists():
        raise SystemExit(
            'This is a Streamlit app - "python app.py" only imports it.\n'
            'Run it with:\n'
            '    streamlit run app.py\n'
            'or, if the streamlit command is not on your PATH:\n'
            '    python -m streamlit run app.py'
        )
    main()
