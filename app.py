import html
import re
from datetime import date, timedelta
from textwrap import dedent
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="CRM팀 기획전 성과 분석",
    layout="wide",
    page_icon="Symbol.png",
)

st.markdown(
    """
<style>
div[data-testid="stDownloadButton"] > button {
  background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0.7rem 1.1rem !important;
  font-weight: 700 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}
div[data-testid="stDownloadButton"] > button:active {
  transform: translateY(0px);
}
</style>
""",
    unsafe_allow_html=True,
)

def safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [df for df in frames if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def _seek_start(file) -> None:
    try:
        file.seek(0)
    except Exception:
        pass


def _resolve_rename_and_usecols(
    header_cols: list[str],
    required_columns: list[str],
    aliases: dict[str, list[str]],
) -> tuple[dict[str, str], list[str]]:
    rename_map: dict[str, str] = {}
    usecols: list[str] = []
    for required in required_columns:
        candidates = aliases.get(required, [required])
        matched = next((c for c in candidates if c in header_cols), None)
        if matched is None:
            continue
        rename_map[matched] = required
        usecols.append(matched)
    return rename_map, usecols


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
    "병원 이름": ["병원 이름", "병원명"],
}

CPV_REQUIRED_COLUMNS = [
    "병원 ID",
    "병원 이름",
    "이벤트 ID (식별자)",
    "이벤트 이름",
    "대카테고리명",
    "중카테고리명",
    "소카테고리명",
    "이벤트 할인가",
    "대상일",
    "CPV 조회 수",
    "CPV 매출",
]

CPV_COLUMN_ALIASES = {
    "이벤트 ID (식별자)": ["이벤트 ID (식별자)", "이벤트 ID"],
    "병원 이름": ["병원 이름", "병원명"],
    "이벤트 이름": ["이벤트 이름", "이벤트명"],
    "대카테고리명": ["대카테고리명"],
    "중카테고리명": ["중카테고리명"],
    "소카테고리명": ["소카테고리명"],
    "이벤트 할인가": ["이벤트 할인가"],
    "CPV 조회 수": ["CPV 조회 수"],
    "CPV 매출": ["CPV 매출"],
}


def normalize_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    aliases: dict[str, list[str]],
) -> pd.DataFrame:
    rename_map = {}
    missing = []
    for required in required_columns:
        candidates = aliases.get(required, [required])
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
    df = pd.read_csv(file, encoding="utf-8-sig", low_memory=False)
    df = normalize_columns(df, REQUIRED_COLUMNS, COLUMN_ALIASES)

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


def _clean_text(value: str) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).replace('"', "").strip()


@st.cache_data(show_spinner=True)
def load_primary_data(file) -> pd.DataFrame:
    try:
        file.seek(0)
    except Exception:
        pass
    try:
        return load_data(file)
    except Exception:
        pass

    try:
        file.seek(0)
    except Exception:
        pass
    df = pd.read_csv(file, encoding="utf-8-sig", skiprows=3)
    df.columns = [_clean_text(c) for c in df.columns]

    def find_column(candidates: list[str]) -> Optional[str]:
        candidates_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            lc = cand.lower()
            if lc in candidates_lower:
                return candidates_lower[lc]
        return None

    event_id_col = find_column(["event_id", "이벤트 ID", "이벤트ID"])
    date_col = find_column(["Time", "time", "date", "day", "대상일", "일자", "날짜"])
    view_col = find_column(["pageview_event.detail--All Users"])
    apply_col = find_column(["apply_event--All Users"])
    event_name_col = find_column(["event_name", "이벤트명", "이벤트 이름"])

    missing = []
    if event_id_col is None:
        missing.append("event_id")
    if date_col is None:
        missing.append("date")
    if view_col is None:
        missing.append("pageview_event.detail--All Users")
    if apply_col is None:
        missing.append("apply_event--All Users")
    if missing:
        raise ValueError(
            "지원하지 않는 조회/상담 CSV 형식입니다. 누락 컬럼: " + ", ".join(missing)
        )

    out = pd.DataFrame(index=df.index, columns=REQUIRED_COLUMNS)
    out["이벤트 ID (식별자)"] = df[event_id_col].map(_clean_text).astype(str)
    if event_name_col is not None:
        out["이벤트 이름"] = df[event_name_col].map(_clean_text)
    out["대상일"] = pd.to_datetime(df[date_col].map(_clean_text), errors="coerce")
    if out["대상일"].isna().any():
        raise ValueError("조회/상담 CSV의 날짜 형식을 파싱할 수 없습니다.")
    out["조회 수"] = pd.to_numeric(df[view_col], errors="coerce").fillna(0).astype(int)
    out["상담신청 수"] = (
        pd.to_numeric(df[apply_col], errors="coerce").fillna(0).astype(int)
    )
    return out


@st.cache_data(show_spinner=False)
def load_primary_meta(file) -> pd.DataFrame:
    # Try existing format meta read (header row is the first row)
    _seek_start(file)
    try:
        header = pd.read_csv(file, encoding="utf-8-sig", nrows=0).columns.tolist()
        meta_required = ["이벤트 ID (식별자)", "이벤트 이름", "대상일", "병원 이름"]
        rename_map, usecols = _resolve_rename_and_usecols(
            header, meta_required, COLUMN_ALIASES
        )
        if "대상일" not in rename_map.values() or "이벤트 ID (식별자)" not in rename_map.values():
            raise ValueError("not_primary_format")
        _seek_start(file)
        df = pd.read_csv(
            file,
            encoding="utf-8-sig",
            usecols=usecols,
            low_memory=False,
        ).rename(columns=rename_map)
        df["대상일"] = pd.to_datetime(df["대상일"], errors="coerce")
        df["이벤트 ID (식별자)"] = df["이벤트 ID (식별자)"].astype(str)
        return df
    except Exception:
        pass

    # Amplitude format meta read (header row is the 4th row)
    _seek_start(file)
    df = pd.read_csv(file, encoding="utf-8-sig", skiprows=3, nrows=0)
    raw_header = df.columns.tolist()
    clean_to_raw: dict[str, str] = {}
    header: list[str] = []
    for raw in raw_header:
        clean = _clean_text(raw)
        header.append(clean)
        if clean and clean not in clean_to_raw:
            clean_to_raw[clean] = raw

    def find_column(candidates: list[str]) -> tuple[Optional[str], Optional[str]]:
        candidates_lower = {c.lower(): c for c in header if c}
        for cand in candidates:
            if cand in header:
                return cand, clean_to_raw.get(cand, cand)
            lc = cand.lower()
            if lc in candidates_lower:
                clean = candidates_lower[lc]
                return clean, clean_to_raw.get(clean, clean)
        return None, None

    event_id_col, event_id_raw = find_column(["event_id", "이벤트 ID", "이벤트ID"])
    date_col, date_raw = find_column(["Time", "time", "date", "day", "대상일", "일자", "날짜"])
    event_name_col, event_name_raw = find_column(["event_name", "이벤트명", "이벤트 이름"])
    if event_id_col is None or date_col is None or event_id_raw is None or date_raw is None:
        raise ValueError("지원하지 않는 조회/상담 CSV 형식입니다.")

    usecols = [c for c in [event_id_raw, date_raw, event_name_raw] if c is not None]
    _seek_start(file)
    raw = pd.read_csv(file, encoding="utf-8-sig", skiprows=3, usecols=usecols)
    raw.columns = [_clean_text(c) for c in raw.columns]

    meta = pd.DataFrame(index=raw.index, columns=["이벤트 ID (식별자)", "이벤트 이름", "대상일", "병원 이름"])
    meta["이벤트 ID (식별자)"] = raw[event_id_col].map(_clean_text).astype(str)
    if event_name_col is not None and event_name_col in raw.columns:
        meta["이벤트 이름"] = raw[event_name_col].map(_clean_text)
    meta["대상일"] = pd.to_datetime(raw[date_col].map(_clean_text), errors="coerce")
    return meta


@st.cache_data(show_spinner=False)
def load_cpv_meta(file) -> pd.DataFrame:
    _seek_start(file)
    header = pd.read_csv(file, encoding="utf-8-sig", nrows=0).columns.tolist()
    meta_required = [
        "이벤트 ID (식별자)",
        "이벤트 이름",
        "대상일",
        "병원 이름",
        "대카테고리명",
        "중카테고리명",
        "소카테고리명",
        "이벤트 할인가",
    ]
    rename_map, usecols = _resolve_rename_and_usecols(
        header, meta_required, CPV_COLUMN_ALIASES
    )
    if "대상일" not in rename_map.values() or "이벤트 ID (식별자)" not in rename_map.values():
        raise ValueError("지원하지 않는 CPV CSV 형식입니다.")
    _seek_start(file)
    df = pd.read_csv(
        file,
        encoding="utf-8-sig",
        usecols=usecols,
        low_memory=False,
    ).rename(columns=rename_map)
    df["대상일"] = pd.to_datetime(df["대상일"], errors="coerce")
    df["이벤트 ID (식별자)"] = df["이벤트 ID (식별자)"].astype(str)
    return df


@st.cache_data(show_spinner=True)
def load_cpv_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="utf-8-sig", low_memory=False)
    df = normalize_columns(df, CPV_REQUIRED_COLUMNS, CPV_COLUMN_ALIASES)
    df["대상일"] = pd.to_datetime(df["대상일"], errors="coerce")
    if df["대상일"].isna().any():
        raise ValueError("CPV 데이터의 대상일 컬럼에 변환할 수 없는 값이 있습니다.")
    for metric_col in ["CPV 조회 수", "CPV 매출", "이벤트 할인가"]:
        if metric_col not in df.columns:
            continue
        df[metric_col] = (
            pd.to_numeric(df[metric_col], errors="coerce").fillna(0).astype(int)
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
        st.dataframe(df, hide_index=True, width="stretch", height=240)


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
    cpv_current_df: pd.DataFrame,
    cpv_previous_df: pd.DataFrame,
    event_lookup: dict[str, str],
) -> pd.DataFrame:
    metrics = ["조회 수", "상담신청 수"]
    hospital_sources = [
        current_df[["이벤트 ID (식별자)", "병원 이름"]],
        previous_df[["이벤트 ID (식별자)", "병원 이름"]],
        cpv_current_df[["이벤트 ID (식별자)", "병원 이름"]],
        cpv_previous_df[["이벤트 ID (식별자)", "병원 이름"]],
    ]
    hospital_info = (
        pd.concat(hospital_sources, ignore_index=True)
        .dropna(subset=["이벤트 ID (식별자)"])
        .drop_duplicates(subset=["이벤트 ID (식별자)"])
    )
    id_sources = [
        current_df["이벤트 ID (식별자)"],
        previous_df["이벤트 ID (식별자)"],
        cpv_current_df["이벤트 ID (식별자)"],
        cpv_previous_df["이벤트 ID (식별자)"],
    ]
    combined_ids = (
        pd.concat(id_sources, ignore_index=True)
        .dropna()
        .astype(str)
        .unique()
    )
    summary = pd.DataFrame({"이벤트 ID (식별자)": combined_ids})
    if summary.empty:
        return summary

    current = (
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
    summary = summary.merge(current, on="이벤트 ID (식별자)", how="left")
    summary = summary.merge(previous, on="이벤트 ID (식별자)", how="left")
    summary["이벤트 이름"] = summary["이벤트 ID (식별자)"].map(event_lookup)
    summary = summary.merge(hospital_info, on="이벤트 ID (식별자)", how="left")

    cpv_metrics = ["CPV 조회 수", "CPV 매출"]
    cpv_attrs = [
        "병원 이름",
        "대카테고리명",
        "중카테고리명",
        "소카테고리명",
        "이벤트 할인가",
        "이벤트 이름",
    ]
    cpv_attr_df = (
        pd.concat([cpv_current_df[["이벤트 ID (식별자)"] + cpv_attrs], cpv_previous_df[["이벤트 ID (식별자)"] + cpv_attrs]], ignore_index=True)
        if (not cpv_current_df.empty or not cpv_previous_df.empty)
        else pd.DataFrame(columns=["이벤트 ID (식별자)"] + cpv_attrs)
    ).drop_duplicates(subset=["이벤트 ID (식별자)"])
    if not cpv_attr_df.empty:
        summary = summary.merge(
            cpv_attr_df,
            on="이벤트 ID (식별자)",
            how="left",
            suffixes=("", "_cpv"),
        )
        if "병원 이름_cpv" in summary.columns:
            summary["병원 이름"] = summary["병원 이름_cpv"].combine_first(summary["병원 이름"])
        if "이벤트 이름_cpv" in summary.columns:
            summary["이벤트 이름"] = summary["이벤트 이름_cpv"].combine_first(summary["이벤트 이름"])
        for extra in ["병원 이름_cpv", "이벤트 이름_cpv"]:
            if extra in summary.columns:
                summary = summary.drop(columns=[extra])
    cpv_current = (
        cpv_current_df.groupby("이벤트 ID (식별자)")[cpv_metrics]
        .sum()
        .reset_index()
        .rename(columns={metric: f"{metric} (진행 기간)" for metric in cpv_metrics})
    )
    cpv_previous = (
        cpv_previous_df.groupby("이벤트 ID (식별자)")[cpv_metrics]
        .sum()
        .reset_index()
        .rename(columns={metric: f"{metric} (이전 기간)" for metric in cpv_metrics})
    )
    summary = summary.merge(cpv_current, on="이벤트 ID (식별자)", how="left")
    summary = summary.merge(cpv_previous, on="이벤트 ID (식별자)", how="left")

    all_metrics = metrics + cpv_metrics
    for metric in all_metrics:
        for period in ["진행 기간", "이전 기간"]:
            col = f"{metric} ({period})"
            if col not in summary:
                summary[col] = 0
            else:
                summary[col] = summary[col].fillna(0)
        current_col = f"{metric} (진행 기간)"
        previous_col = f"{metric} (이전 기간)"
        diff_col = f"{metric} 증감량"
        summary[diff_col] = summary[current_col] - summary[previous_col]
        rate_col = f"{metric} 증감률"
        summary[rate_col] = np.where(
            summary[previous_col] > 0,
            summary[diff_col] / summary[previous_col] * 100,
            np.nan,
        )

    columns_order = [
        "이벤트 이름",
        "병원 이름",
        "대카테고리명",
        "중카테고리명",
        "소카테고리명",
        "이벤트 할인가",
        "이벤트 ID (식별자)",
        "조회 수 (진행 기간)",
        "조회 수 (이전 기간)",
        "조회 수 증감량",
        "조회 수 증감률",
        "상담신청 수 (진행 기간)",
        "상담신청 수 (이전 기간)",
        "상담신청 수 증감량",
        "상담신청 수 증감률",
        "CPV 매출 (진행 기간)",
        "CPV 매출 (이전 기간)",
        "CPV 매출 증감량",
        "CPV 매출 증감률",
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

    def format_label(row: pd.Series) -> str:
        event_name = row.get("이벤트 이름") or row.get("이벤트 ID (식별자)") or ""
        hospital_name = row.get("병원 이름")
        if pd.notna(hospital_name) and str(hospital_name).strip():
            return f"{event_name} ({hospital_name})"
        return str(event_name)

    def add_amount_card(metric: str):
        diff_col = f"{metric} 증감량"
        if diff_col not in summary_df:
            return
        positive = summary_df[summary_df[diff_col] > 0]
        if positive.empty:
            return
        top_positive = positive.sort_values(diff_col, ascending=False).head(top_n)
        unit = "원" if metric == "CPV 매출" else "건"
        items = [
            {
                "label": format_label(row),
                "value": f"+{int(row[diff_col]):,}{unit}",
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

    def add_rate_card(metric: str):
        diff_rate_col = f"{metric} 증감률"
        diff_amount_col = f"{metric} 증감량"
        if diff_rate_col not in summary_df:
            return
        positive_rate = summary_df[summary_df[diff_rate_col] > 0]
        if positive_rate.empty:
            return
        top_rate = positive_rate.sort_values(diff_rate_col, ascending=False).head(top_n)
        unit = "원" if metric == "CPV 매출" else "건"
        items = []
        for _, row in top_rate.iterrows():
            amount = row.get(diff_amount_col)
            amount_part = (
                f"+{int(amount):,}{unit}" if pd.notna(amount) and amount != 0 else "-"
            )
            items.append(
                {
                    "label": format_label(row),
                    "value": f"+{row[diff_rate_col]:.1f}% ({amount_part})",
                }
            )
        insights.append(
            {
                "title": f"{metric} 증감률 TOP {len(items)}",
                "badge": f"{metric} 증감률",
                "items": items,
            }
        )

    insights: list[dict] = []
    add_amount_card("상담신청 수")
    add_rate_card("상담신청 수")
    add_amount_card("조회 수")
    add_rate_card("조회 수")
    add_amount_card("CPV 매출")
    add_rate_card("CPV 매출")

    if not insights:
        insights.append(
            {
                "title": "변화 없음",
                "badge": "Info",
                "items": ["두 기간 사이에서 뚜렷한 증감을 보이는 이벤트가 없습니다."],
            }
        )

    total_events = int(len(summary_df))
    if total_events > 0:
        views_up = int((summary_df.get("조회 수 증감량", 0) > 0).sum())
        applies_up = int((summary_df.get("상담신청 수 증감량", 0) > 0).sum())
        views_pct = (views_up / total_events) * 100
        applies_pct = (applies_up / total_events) * 100
        insights.append(
            {
                "title": "기획전 내 이벤트 성장 현황",
                "badge": "요약",
                "items": [
                    {"label": "전체 이벤트 수", "value": f"{total_events:,}"},
                    {
                        "label": "조회수 증가 이벤트",
                        "value": f"{views_up:,} ({views_pct:.1f}%)",
                    },
                    {
                        "label": "상담신청 증가 이벤트",
                        "value": f"{applies_up:,} ({applies_pct:.1f}%)",
                    },
                ],
            }
        )
    return insights


def format_event_summary_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()
    number_cols = [
        col
        for col in display_df.columns
        if (
            (
                "조회 수" in col
                or "상담신청 수" in col
                or "CPV 매출" in col
                or col == "이벤트 할인가"
            )
            and "증감률" not in col
        )
    ]
    rate_cols = [col for col in display_df.columns if "증감률" in col]

    for col in number_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x != 0 else "-"
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
    columns_per_row = min(2, len(insights))
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
    for start in range(0, len(insights), columns_per_row):
        cols = st.columns(columns_per_row)
        row_insights = insights[start : start + columns_per_row]
        for idx, insight in enumerate(row_insights):
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
            with cols[idx]:
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


def render_metrics(metric_rows: list[dict]):
    if not metric_rows:
        return
    cols = st.columns(len(metric_rows))
    for idx, metric in enumerate(metric_rows):
        label = metric["label"]
        current_raw = metric.get("current")
        previous_raw = metric.get("previous")

        current_display = "-"
        delta = None

        if current_raw is not None:
            current_val = int(current_raw)
            current_display = f"{current_val:,}"

        if current_raw is not None and previous_raw is not None:
            previous_val = int(previous_raw)
            delta_numeric = current_val - previous_val
            if previous_val == 0:
                delta = f"{delta_numeric:+,} (이전 기간 0)"
            else:
                delta_percentage = (delta_numeric / previous_val) * 100
                delta = f"{delta_numeric:+,} ({delta_percentage:+.1f}%)"

        cols[idx].metric(
            label,
            current_display,
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
    st.plotly_chart(fig, width="stretch")


st.title("💜 CRM팀 기획전 성과 분석")
st.markdown(
    """
조회/상담 CSV는 앰플리튜드([링크](https://app.amplitude.com/analytics/babitalk/chart/g7sowyhf))에서, \nCPV CSV는 퀵사이트([링크](https://ap-northeast-2.quicksight.aws.amazon.com/sn/account/babitalk-data-quicksight/dashboards/74afc507-059e-421c-910d-303f57ae1900/sheets/74afc507-059e-421c-910d-303f57ae1900_01e8ddcf-7b44-4d69-8cd4-7eb22915f9ec))에서 다운로드해 업로드해주세요.\n파일을 업로드한 뒤 사이드바에서 분석할 이벤트 ID와 분석 기간(기획전 진행기간)을 선택하고, '분석 시작'을 눌러주세요.\n이 페이지에 문제가 생기면 CRM팀 **@김예슬** 에게 문의해주세요.🍀\n\n*퀵사이트에서 [🗓️대상일 - descending]을 선택해서 CSV를 다운받는 것을 추천합니다. 조회/상담 CSV와 CPV CSV 모두 비교하고자 하는 분석 기간을 포함하고 있어야합니다.
"""
)

uploaded_file = st.file_uploader(
    "조회/상담 CSV 파일을 업로드하세요.", type=["csv"], key="primary_csv"
)
cpv_uploaded_file = st.file_uploader(
    "CPV CSV 파일을 업로드하세요.", type=["csv"], key="cpv_csv"
)

if uploaded_file is None and cpv_uploaded_file is None:
    st.info("조회/상담 CSV 또는 CPV CSV 중 최소 하나를 업로드해주세요.")
    st.stop()

try:
    df_meta = (
        load_primary_meta(uploaded_file)
        if uploaded_file
        else pd.DataFrame(columns=["이벤트 ID (식별자)", "이벤트 이름", "대상일", "병원 이름"])
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"조회/상담 CSV를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

try:
    cpv_meta = (
        load_cpv_meta(cpv_uploaded_file)
        if cpv_uploaded_file
        else pd.DataFrame(
            columns=[
                "이벤트 ID (식별자)",
                "이벤트 이름",
                "대상일",
                "병원 이름",
                "대카테고리명",
                "중카테고리명",
                "소카테고리명",
                "이벤트 할인가",
            ]
        )
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"CPV CSV를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

combined_meta = safe_concat([df_meta, cpv_meta])
if combined_meta.empty:
    st.error("업로드한 파일에서 이벤트/날짜 정보를 찾지 못했습니다.")
    st.stop()

combined_meta = combined_meta.dropna(subset=["대상일"])
if combined_meta.empty:
    st.error("날짜(대상일) 데이터가 없어 분석을 진행할 수 없습니다.")
    st.stop()

event_options = get_event_options(combined_meta)
if not event_options:
    st.error("이벤트 정보가 포함된 데이터가 없습니다.")
    st.stop()

event_lookup = {event_id: event_name for event_id, event_name in event_options}
available_ids = list(event_lookup.keys())

min_date = combined_meta["대상일"].min().date()
max_date = combined_meta["대상일"].max().date()
st.sidebar.header("분석 설정")

default_event_input = available_ids[0] if available_ids else ""
if "analysis_params" not in st.session_state:
    st.session_state.analysis_params = None

with st.sidebar.form("analysis_form"):
    with st.expander("이벤트 ID 목록 보기"):
        st.dataframe(
            pd.DataFrame(event_options, columns=["이벤트 ID", "이벤트 이름"]).head(300),
            width="stretch",
        )
    event_input = st.text_area(
        "분석할 이벤트 ID (줄바꿈 또는 쉼표로 구분)",
        value=default_event_input,
        height=120,
        placeholder="예: 53004\\n47917",
    )
    date_range = st.date_input(
        "분석 기간 (진행 기간)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    submitted = st.form_submit_button("분석 시작", type="primary")

if submitted:
    start_date, end_date = (
        (date_range, date_range) if isinstance(date_range, date) else date_range
    )
    if start_date > end_date:
        st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
        st.stop()
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
    st.session_state.analysis_params = {
        "selected_event_ids": selected_event_ids,
        "start_date": start_date,
        "end_date": end_date,
    }

if st.session_state.analysis_params is None:
    st.info("사이드바에서 이벤트/기간을 선택한 뒤 '분석 시작'을 눌러주세요.")
    st.stop()

selected_event_ids = st.session_state.analysis_params["selected_event_ids"]
start_date = st.session_state.analysis_params["start_date"]
end_date = st.session_state.analysis_params["end_date"]

try:
    df = (
        load_primary_data(uploaded_file)
        if uploaded_file
        else pd.DataFrame(columns=REQUIRED_COLUMNS)
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"조회/상담 CSV를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

try:
    cpv_df = (
        load_cpv_data(cpv_uploaded_file)
        if cpv_uploaded_file
        else pd.DataFrame(columns=CPV_REQUIRED_COLUMNS)
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"CPV CSV를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

if uploaded_file is None:
    st.warning("조회/상담 CSV가 없으면 조회 수 및 상담신청 지표는 제공되지 않습니다.")
if cpv_uploaded_file is None:
    st.info("CPV CSV를 업로드하면 CPV 매출 분석을 함께 확인할 수 있습니다.")

event_df = df[df["이벤트 ID (식별자)"].isin(selected_event_ids)].copy()
cpv_event_df = cpv_df[cpv_df["이벤트 ID (식별자)"].isin(selected_event_ids)].copy()

if event_df.empty and cpv_event_df.empty:
    st.error("선택한 이벤트에 대한 데이터가 없습니다.")
    st.stop()

if event_df.empty:
    st.warning("선택한 이벤트에 대한 조회/상담 데이터가 없습니다.")
if cpv_event_df.empty:
    st.warning("선택한 이벤트에 대한 CPV 데이터가 없습니다.")

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
cpv_current_period_df = filter_period(cpv_event_df, current_start, current_end)
cpv_previous_period_df = filter_period(cpv_event_df, previous_start, previous_end)

st.subheader(f"선택된 이벤트 ({len(selected_event_ids)}개)")
render_selected_events(selected_event_ids, event_lookup)
st.caption(
    f"진행 기간: {current_start.date()} ~ {current_end.date()} | "
    f"이전 기간: {previous_start.date()} ~ {previous_end.date()} "
    f"(총 {period_days}일)"
)

if current_period_df.empty and cpv_current_period_df.empty:
    st.warning("선택된 기간 내 데이터가 없습니다. 다른 기간을 선택해주세요.")
    st.stop()


def _metric_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df or df[column].dropna().empty:
        return None
    return int(df[column].sum())


metric_rows = [
    {
        "label": "조회 수",
        "current": _metric_sum(current_period_df, "조회 수"),
        "previous": _metric_sum(previous_period_df, "조회 수")
        if not previous_period_df.empty
        else None,
    },
    {
        "label": "상담신청 수",
        "current": _metric_sum(current_period_df, "상담신청 수"),
        "previous": _metric_sum(previous_period_df, "상담신청 수")
        if not previous_period_df.empty
        else None,
    },
    {
        "label": "CPV 조회 수",
        "current": _metric_sum(cpv_current_period_df, "CPV 조회 수"),
        "previous": _metric_sum(cpv_previous_period_df, "CPV 조회 수")
        if not cpv_previous_period_df.empty
        else None,
    },
    {
        "label": "CPV 매출",
        "current": _metric_sum(cpv_current_period_df, "CPV 매출"),
        "previous": _metric_sum(cpv_previous_period_df, "CPV 매출")
        if not cpv_previous_period_df.empty
        else None,
    },
]

render_metrics(metric_rows)

event_summary_df = build_event_summary(
    current_period_df,
    previous_period_df,
    cpv_current_period_df,
    cpv_previous_period_df,
    event_lookup,
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
            width="stretch",
            hide_index=True,
        )
        csv_bytes = event_summary_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            "📥 이 테이블을 CSV로 내보내기 (click!)",
            data=csv_bytes,
            file_name="event_summary.csv",
            mime="text/csv",
            key="download_event_summary",
        )
    else:
        st.info("이벤트별 세부 지표를 계산할 데이터가 충분하지 않습니다.")

with tab_trend:
    st.markdown("#### 기간별 상담신청 수 추이")
    render_chart(current_period_df)

with st.expander("원본 데이터 미리보기"):
    preview_df = event_df.sort_values("대상일").head(500)
    st.dataframe(preview_df, width="stretch", hide_index=True)
