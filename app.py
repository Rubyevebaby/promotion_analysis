import html
import re
from datetime import date, timedelta
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="CRM팀 기획전 성과 분석",
    layout="wide",
    page_icon="Symbol.png",
)

REQUIRED_COLUMNS = [
    "병원 ID",
    "병원 이름",
    "대행사 ID",
    "대행사 이름",
    "이벤트 ID (식별자)",
    "이벤트 이름",
    "이벤트 가격 (text)",
    "카테고리 (최상위)",
    "카테고리 (대)",
    "카테고리 (중)",
    "카테고리 (소)",
    "대상일",
    "조회 수",
    "상담신청 수",
]

COLUMN_ALIASES = {
    "이벤트 ID (식별자)": ["이벤트 ID (식별자)", "이벤트 ID"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    missing = []
    for required in REQUIRED_COLUMNS:
        candidates = COLUMN_ALIASES.get(required, [required])
        matched = next((c for c in candidates if c in df.columns), None)
        if matched:
            rename_map[matched] = required
        else:
            missing.append(required)
    if missing:
        raise ValueError(f"다음 컬럼이 누락되었습니다: {', '.join(missing)}")
    return df.rename(columns=rename_map)


@st.cache_data(show_spinner=True)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="utf-8-sig")
    df = normalize_columns(df)

    df["대상일"] = pd.to_datetime(df["대상일"], errors="coerce")
    if df["대상일"].isna().any():
        raise ValueError("대상일 컬럼에 변환할 수 없는 값이 있습니다. YYYY-MM-DD 형식을 확인하세요.")

    for metric_col in ["조회 수", "상담신청 수"]:
        df[metric_col] = (
            pd.to_numeric(df[metric_col], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    df["이벤트 ID (식별자)"] = df["이벤트 ID (식별자)"].astype(str)
    return df


def get_event_options(df: pd.DataFrame):
    options = (
        df[["이벤트 ID (식별자)", "이벤트 이름"]]
        .drop_duplicates()
        .sort_values("이벤트 이름")
    )
    return list(options.itertuples(index=False, name=None))


def parse_event_ids_input(raw_text: str, valid_ids: set[str]):
    if not raw_text:
        return [], []
    tokens = [
        token.strip()
        for token in re.split(r"[,\n]+", raw_text)
        if token.strip()
    ]
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    invalid = [token for token in seen if token not in valid_ids]
    valid = [token for token in seen if token in valid_ids]
    return valid, invalid


def render_selected_events(selected_ids: list[str], event_lookup: dict[str, str]):
    if not selected_ids:
        st.info("선택된 이벤트가 없습니다.")
        return
    data = [
        {"이벤트 ID": event_id, "이벤트 이름": event_lookup.get(event_id, "알 수 없음")}
        for event_id in selected_ids
    ]
    df = pd.DataFrame(data)
    with st.expander("선택된 이벤트 목록", expanded=False):
        st.dataframe(df, hide_index=True, use_container_width=True, height=240)


def get_date_range_input(df: pd.DataFrame):
    min_date = df["대상일"].min().date()
    max_date = df["대상일"].max().date()
    default_range = (min_date, max_date)
    selected = st.sidebar.date_input(
        "분석 기간 (진행 기간)",
        value=default_range,
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected, date):
        return selected, selected
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        return selected[0], selected[1]
    st.sidebar.error("기간의 시작일과 종료일을 모두 선택해주세요.")
    st.stop()


def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    mask = (df["대상일"] >= start) & (df["대상일"] <= end)
    return df.loc[mask].copy()


def build_event_summary(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    event_lookup: dict[str, str],
) -> pd.DataFrame:
    metrics = ["조회 수", "상담신청 수"]
    summary = (
        current_df.groupby("이벤트 ID (식별자)")[metrics]
        .sum()
        .reset_index()
        .rename(
            columns={
                metric: f"{metric} (진행 기간)" for metric in metrics
            }
        )
    )
    previous = (
        previous_df.groupby("이벤트 ID (식별자)")[metrics]
        .sum()
        .reset_index()
        .rename(
            columns={
                metric: f"{metric} (이전 기간)" for metric in metrics
            }
        )
    )
    summary = summary.merge(previous, on="이벤트 ID (식별자)", how="outer").fillna(0)
    summary["이벤트 이름"] = summary["이벤트 ID (식별자)"].map(event_lookup)

    for metric in metrics:
        current_col = f"{metric} (진행 기간)"
        previous_col = f"{metric} (이전 기간)"
        diff_col = f"{metric} Diff"
        summary[diff_col] = summary[current_col] - summary[previous_col]
        rate_col = f"{metric} 증감률"
        summary[rate_col] = np.where(
            summary[previous_col] > 0,
            summary[diff_col] / summary[previous_col] * 100,
            np.nan,
        )

    columns_order = [
        "이벤트 이름",
        "이벤트 ID (식별자)",
        "조회 수 (진행 기간)",
        "조회 수 (이전 기간)",
        "조회 수 Diff",
        "조회 수 증감률",
        "상담신청 수 (진행 기간)",
        "상담신청 수 (이전 기간)",
        "상담신청 수 Diff",
        "상담신청 수 증감률",
    ]
    existing_columns = [col for col in columns_order if col in summary.columns]
    return summary[existing_columns].sort_values(
        "상담신청 수 (진행 기간)", ascending=False
    )


def generate_event_insights(summary_df: pd.DataFrame, top_n: int = 3) -> list[dict]:
    if summary_df.empty:
        return [
            {
                "title": "데이터 부족",
                "badge": "Info",
                "items": ["선택된 이벤트에 대한 비교 가능한 데이터가 없습니다."],
            }
        ]

    insights: list[dict] = []
    metrics = ["조회 수", "상담신청 수"]
    for metric in metrics:
        diff_col = f"{metric} Diff"
        if diff_col not in summary_df:
            continue
        positive = summary_df[summary_df[diff_col] > 0]

        if not positive.empty:
            top_positive = positive.sort_values(diff_col, ascending=False).head(top_n)
            items = [
                {
                    "label": row.get("이벤트 이름") or row["이벤트 ID (식별자)"],
                    "value": f"+{int(row[diff_col]):,}건",
                }
                for _, row in top_positive.iterrows()
            ]
            insights.append(
                {
                    "title": f"{metric} 상승 TOP {len(items)}",
                    "badge": metric,
                    "items": items,
                }
            )


    if not insights:
        insights.append(
            {
                "title": "변화 없음",
                "badge": "Info",
                "items": ["두 기간 사이에서 뚜렷한 증감을 보이는 이벤트가 없습니다."],
            }
        )
    return insights


def format_event_summary_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()
    number_cols = [
        col
        for col in display_df.columns
        if (
            ("조회 수" in col or "상담신청 수" in col)
            and "증감률" not in col
            and "전환율" not in col
        )
    ]
    rate_cols = [col for col in display_df.columns if "증감률" in col]

    for col in number_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "-"
        )
    for col in rate_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
        )
    return display_df


def render_insight_cards(insights: list[dict]):
    if not insights:
        st.info("표시할 인사이트가 없습니다.")
        return
    columns_count = min(3, len(insights))
    cols = st.columns(columns_count)
    card_template = dedent(
        """
        <div style="
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #eef2ff, #fef3c7);
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        ">
            <div style="font-size:12px;font-weight:600;color:#6366f1;">{badge}</div>
            <div style="font-size:16px;font-weight:700;color:#111827;margin-top:4px;">{title}</div>
            <div style="margin-top:10px;">{items}</div>
        </div>
        """
    ).strip()
    item_template = dedent(
        """
        <div style="
            display:flex;
            justify-content:space-between;
            font-size:14px;
            color:#374151;
            padding:4px 0;
            border-bottom:1px dashed #e5e7eb;
        ">
            <span style="flex:1; margin-right:8px;">{label}</span>
            <span style="font-weight:600;color:#111827;">{value}</span>
        </div>
        """
    ).strip()
    for idx, insight in enumerate(insights):
        column = cols[idx % columns_count]
        with column:
            items = insight.get("items") or []
            rendered_items = []
            for item in items:
                if isinstance(item, dict):
                    label = item.get("label", "")
                    value = item.get("value", "")
                elif isinstance(item, str):
                    label, value = item, ""
                else:
                    label, value = str(item), ""
                label_safe = html.escape(str(label))
                value_safe = html.escape(str(value))
                rendered_items.append(
                    item_template.format(label=label_safe, value=value_safe)
                )
            badge_safe = html.escape(str(insight.get("badge", "")))
            title_safe = html.escape(str(insight.get("title", "")))
            st.markdown(
                card_template.format(
                    badge=badge_safe,
                    title=title_safe,
                    items="".join(rendered_items) or "<div>데이터 없음</div>",
                ),
                unsafe_allow_html=True,
            )


def build_timeseries_with_dates(df: pd.DataFrame, label: str):
    if df.empty:
        return pd.DataFrame(columns=["Day", "Date", "상담신청 수", "기간"])
    sorted_df = df.sort_values("대상일")
    day_offsets = (
        sorted_df["대상일"].values - sorted_df["대상일"].min().to_datetime64()
    ) / np.timedelta64(1, "D")
    sorted_df["Day"] = day_offsets.astype(int) + 1
    sorted_df["Date"] = sorted_df["대상일"].dt.date
    ts = (
        sorted_df.groupby(["Day", "Date"])["상담신청 수"]
        .sum()
        .reset_index()
    )
    ts["기간"] = label
    return ts


def render_metrics(current_df: pd.DataFrame, previous_df: pd.DataFrame):
    metrics = {
        "조회 수": "조회 수",
        "상담신청 수": "상담신청 수",
    }
    cols = st.columns(len(metrics))
    for idx, (label, column) in enumerate(metrics.items()):
        current_val = int(current_df[column].sum())
        previous_val = int(previous_df[column].sum()) if not previous_df.empty else None
        delta = None
        if previous_val is not None:
            delta_numeric = current_val - previous_val
            if previous_val == 0:
                delta = f"{delta_numeric:+,} (이전 기간 0)"
            else:
                delta_percentage = (delta_numeric / previous_val) * 100
                delta = f"{delta_numeric:+,} ({delta_percentage:+.1f}%)"
        cols[idx].metric(
            label,
            f"{current_val:,}",
            delta=delta or "비교 데이터 없음",
        )


def render_chart(current_df: pd.DataFrame):
    chart_df = build_timeseries_with_dates(current_df, "진행 기간")
    if chart_df.empty:
        st.info("차트를 표시할 데이터가 없습니다.")
        return
    chart_df["DayLabel"] = chart_df.apply(
        lambda row: f"Day {int(row['Day'])} ({row['Date']})", axis=1
    )
    fig = px.line(
        chart_df,
        x="DayLabel",
        y="상담신청 수",
        markers=True,
        labels={"DayLabel": "경과 일수 (날짜)", "상담신청 수": "상담신청 수"},
    )
    fig.update_layout(height=400, hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


st.title("💜 CRM팀 기획전 성과 분석")
st.markdown(
    """
퀵사이트 대시보드([링크](https://ap-northeast-2.quicksight.aws.amazon.com/sn/account/babitalk-data-quicksight/dashboards/74afc507-059e-421c-910d-303f57ae1900/sheets/74afc507-059e-421c-910d-303f57ae1900_26f4f316-a3a6-4f09-9797-708c736937d5))에서 CSV를 내려받아 이 페이지에 업로드한 뒤, 사이드바에서 분석할 이벤트 ID와 분석 기간(기획전 진행기간)을 입력해주세요. 퀵사이트 대시보드에서 [대상일 - descending]을 선택해서 내려받는 것을 추천합니다. (약 최근 6개월 커버 가능) 이 페이지에 문제가 생기면 CRM팀 **@김예슬** 에게 문의해주세요.
"""
)

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])
if uploaded_file is None:
    st.info("분석을 시작하려면 CSV 파일을 업로드해주세요.")
    st.stop()

try:
    df = load_data(uploaded_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

st.sidebar.header("분석 설정")
event_options = get_event_options(df)
if not event_options:
    st.error("이벤트 정보가 포함된 데이터가 없습니다.")
    st.stop()

event_lookup = {event_id: event_name for event_id, event_name in event_options}
available_ids = list(event_lookup.keys())

with st.sidebar.expander("이벤트 ID 목록 보기"):
    st.dataframe(
        pd.DataFrame(event_options, columns=["이벤트 ID", "이벤트 이름"]),
        use_container_width=True,
    )

default_event_input = available_ids[0] if available_ids else ""
event_input = st.sidebar.text_area(
    "분석할 이벤트 ID (줄바꿈 또는 쉼표로 구분)",
    value=default_event_input,
    height=120,
    placeholder="예: 53004\\n47917",
)
selected_event_ids, invalid_event_ids = parse_event_ids_input(
    event_input,
    set(available_ids),
)

if invalid_event_ids:
    st.sidebar.error(
        f"다음 이벤트 ID는 데이터에 없습니다: {', '.join(invalid_event_ids)}"
    )
    st.stop()

if not selected_event_ids:
    st.sidebar.error("최소 한 개의 이벤트 ID를 입력해주세요.")
    st.stop()

event_df = df[df["이벤트 ID (식별자)"].isin(selected_event_ids)].copy()

if event_df.empty:
    st.error("선택한 이벤트에 대한 데이터가 없습니다.")
    st.stop()

start_date, end_date = get_date_range_input(event_df)

if start_date > end_date:
    st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
    st.stop()

current_start = pd.Timestamp(start_date)
current_end = pd.Timestamp(end_date)
period_days = (current_end - current_start).days + 1
if period_days <= 0:
    st.sidebar.error("분석 기간은 최소 하루 이상이어야 합니다.")
    st.stop()

current_period_df = filter_period(event_df, current_start, current_end)
previous_end = current_start - timedelta(days=1)
previous_start = previous_end - timedelta(days=period_days - 1)
previous_period_df = filter_period(event_df, previous_start, previous_end)

st.subheader(f"선택된 이벤트 ({len(selected_event_ids)}개)")
render_selected_events(selected_event_ids, event_lookup)
st.caption(
    f"진행 기간: {current_start.date()} ~ {current_end.date()} | "
    f"이전 기간: {previous_start.date()} ~ {previous_end.date()} "
    f"(총 {period_days}일)"
)

if current_period_df.empty:
    st.warning("선택된 기간 내 데이터가 없습니다. 다른 기간을 선택해주세요.")
    st.stop()

render_metrics(current_period_df, previous_period_df)

event_summary_df = build_event_summary(
    current_period_df, previous_period_df, event_lookup
)
event_insights = generate_event_insights(event_summary_df)
event_summary_display = format_event_summary_display(event_summary_df)

tab_insight, tab_trend = st.tabs(
    ["💡 이벤트 인사이트", "📈 기간별 추이"]
)

with tab_insight:
    st.markdown("#### 이벤트 성과 하이라이트")
    render_insight_cards(event_insights)
    if not event_summary_df.empty:
        st.markdown("##### 이벤트별 상세 지표")
        st.dataframe(
            event_summary_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("이벤트별 세부 지표를 계산할 데이터가 충분하지 않습니다.")

with tab_trend:
    st.markdown("#### 기간별 상담신청 수 추이")
    render_chart(current_period_df)

with st.expander("원본 데이터 미리보기"):
    st.dataframe(event_df.sort_values("대상일"), use_container_width=True)
