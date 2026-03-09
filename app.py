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
    "CPV 매출",
    "결제 수",
]

PRIMARY_MIN_COLUMNS = [
    "이벤트 ID (식별자)",
    "이벤트 이름",
    "병원 이름",
    "대상일",
    "조회 수",
    "상담신청 수",
    "CPV 매출",
    "결제 수",
    "대카테고리명",
    "중카테고리명",
    "소카테고리명",
]

COLUMN_ALIASES = {
    "이벤트 ID (식별자)": ["이벤트 ID (식별자)", "이벤트 ID"],
    "병원 이름": ["병원 이름", "병원명"],
}

PRIMARY_MIN_ALIASES = {
    "이벤트 ID (식별자)": ["이벤트 ID (식별자)", "이벤트 ID", "event_id", "eventId", "id"],
    "이벤트 이름": ["이벤트 이름", "이벤트명", "event_name", "name"],
    "병원 이름": ["병원 이름", "병원명", "hospital_name", "client_name"],
    "대상일": ["대상일", "일자", "날짜", "Time", "time", "date", "day"],
    "조회 수": ["조회 수", "pageview_event.detail--All Users", "조회수_전체"],
    "상담신청 수": ["상담신청 수", "apply_event--All Users", "상담수_전체"],
    "CPV 매출": ["CPV 매출", "CPV_전체"],
    "결제 수": ["결제 수", "결제수_전체"],
    "대카테고리명": ["대카테고리명", "category1"],
    "중카테고리명": ["중카테고리명", "category2"],
    "소카테고리명": ["소카테고리명", "category3"],
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


def _update_first_non_empty(mapping: dict[str, str], key: str, value: Optional[str]) -> None:
    if key is None:
        return
    key = str(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return
    cleaned = _clean_text(value)
    if not cleaned:
        return
    if key not in mapping or not mapping.get(key):
        mapping[key] = cleaned


def _update_first_non_empty_row(
    mapping: dict[str, dict[str, object]],
    key: str,
    row: dict[str, object],
) -> None:
    if key is None:
        return
    key = str(key)
    if key not in mapping:
        mapping[key] = {}
    target = mapping[key]
    for k, v in row.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, str):
            v_clean = _clean_text(v)
            if not v_clean:
                continue
            if k not in target or not target.get(k):
                target[k] = v_clean
        else:
            if k not in target or target.get(k) is None or (isinstance(target.get(k), float) and pd.isna(target.get(k))):
                target[k] = v


def _aggregate_metric_chunks(
    file,
    *,
    read_csv_kwargs: dict,
    rename_map: dict[str, str],
    group_cols: list[str],
    metric_cols: list[str],
    info_extract_cols: list[str],
    chunk_size: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    _seek_start(file)
    aggregated: Optional[pd.DataFrame] = None
    info_map: dict[str, dict[str, object]] = {}

    reader = pd.read_csv(file, chunksize=chunk_size, **read_csv_kwargs)
    for chunk in reader:
        if rename_map:
            chunk = chunk.rename(columns=rename_map)
        chunk.columns = [_clean_text(c) for c in chunk.columns]

        if "이벤트 ID (식별자)" not in chunk.columns or "대상일" not in chunk.columns:
            continue

        chunk["이벤트 ID (식별자)"] = chunk["이벤트 ID (식별자)"].map(_clean_text).astype(str)
        chunk["대상일"] = pd.to_datetime(chunk["대상일"].map(_clean_text), errors="coerce")
        chunk = chunk.dropna(subset=["대상일"])

        for metric in metric_cols:
            if metric not in chunk.columns:
                chunk[metric] = 0
            chunk[metric] = pd.to_numeric(chunk[metric], errors="coerce").fillna(0)

        available_info_cols = [c for c in info_extract_cols if c in chunk.columns]
        if available_info_cols:
            subset = chunk[["이벤트 ID (식별자)"] + available_info_cols].dropna(
                subset=["이벤트 ID (식별자)"]
            )
            for col in available_info_cols:
                subset[col] = subset[col].map(_clean_text).replace("", np.nan)
            subset = subset.dropna(subset=available_info_cols, how="all")
            if not subset.empty:
                first_rows = (
                    subset.groupby("이벤트 ID (식별자)", sort=False)[available_info_cols]
                    .first()
                )
                for event_id, row in first_rows.iterrows():
                    _update_first_non_empty_row(
                        info_map, str(event_id), row.to_dict()
                    )

        grouped = (
            chunk.groupby(group_cols, sort=False)[metric_cols]
            .sum()
        )
        aggregated = grouped if aggregated is None else aggregated.add(grouped, fill_value=0)

    if aggregated is None:
        empty = pd.DataFrame(columns=group_cols + metric_cols)
        return empty, info_map
    return aggregated.reset_index(), info_map


def _scan_meta_chunks(
    file,
    *,
    read_csv_kwargs: dict,
    rename_map: dict[str, str],
    event_id_col: str,
    date_col: str,
    name_col: Optional[str] = None,
    hospital_col: Optional[str] = None,
    chunk_size: int = 200_000,
) -> dict:
    _seek_start(file)
    min_ts: Optional[pd.Timestamp] = None
    max_ts: Optional[pd.Timestamp] = None
    event_ids: set[str] = set()
    event_name_map: dict[str, str] = {}
    hospital_name_map: dict[str, str] = {}

    reader = pd.read_csv(file, chunksize=chunk_size, **read_csv_kwargs)
    for chunk in reader:
        if rename_map:
            chunk = chunk.rename(columns=rename_map)
        chunk.columns = [_clean_text(c) for c in chunk.columns]
        if event_id_col not in chunk.columns or date_col not in chunk.columns:
            continue

        chunk[event_id_col] = chunk[event_id_col].map(_clean_text).astype(str)
        event_ids.update(chunk[event_id_col].dropna().astype(str).unique().tolist())
        dates = pd.to_datetime(chunk[date_col].map(_clean_text), errors="coerce")
        dates = dates.dropna()
        if not dates.empty:
            chunk_min = dates.min()
            chunk_max = dates.max()
            min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
            max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)

        if name_col and name_col in chunk.columns:
            pairs = chunk[[event_id_col, name_col]].copy()
            pairs[name_col] = pairs[name_col].map(_clean_text).replace("", np.nan)
            pairs = pairs.dropna(subset=[name_col]).drop_duplicates(subset=[event_id_col])
            for _, row in pairs.iterrows():
                _update_first_non_empty(event_name_map, str(row[event_id_col]), row[name_col])
        if hospital_col and hospital_col in chunk.columns:
            pairs = chunk[[event_id_col, hospital_col]].copy()
            pairs[hospital_col] = pairs[hospital_col].map(_clean_text).replace("", np.nan)
            pairs = pairs.dropna(subset=[hospital_col]).drop_duplicates(subset=[event_id_col])
            for _, row in pairs.iterrows():
                _update_first_non_empty(
                    hospital_name_map,
                    str(row[event_id_col]),
                    row[hospital_col],
                )

    return {
        "min_ts": min_ts,
        "max_ts": max_ts,
        "event_ids": event_ids,
        "event_name_map": event_name_map,
        "hospital_name_map": hospital_name_map,
    }


def load_primary_meta_info(file) -> dict:
    # Legacy format (header at first row)
    _seek_start(file)
    try:
        raw_header = pd.read_csv(file, encoding="utf-8-sig", nrows=0).columns.tolist()
        clean_to_raw: dict[str, str] = {}
        header: list[str] = []
        for raw in raw_header:
            clean = _clean_text(raw)
            header.append(clean)
            if clean and clean not in clean_to_raw:
                clean_to_raw[clean] = raw
        required = ["이벤트 ID (식별자)", "대상일", "이벤트 이름", "병원 이름"]
        rename_map_clean, usecols_clean = _resolve_rename_and_usecols(
            header,
            required,
            {**COLUMN_ALIASES, **PRIMARY_MIN_ALIASES},
        )
        rename_map = {
            clean_to_raw.get(clean, clean): required
            for clean, required in rename_map_clean.items()
        }
        usecols = [clean_to_raw.get(clean, clean) for clean in usecols_clean]
        must_have = {"이벤트 ID (식별자)", "대상일"}
        if must_have.issubset(set(rename_map.values())):
            return _scan_meta_chunks(
                file,
                read_csv_kwargs={
                    "encoding": "utf-8-sig",
                    "usecols": usecols,
                    "low_memory": False,
                    "dtype": {
                        col: str for col, req in rename_map.items() if req == "이벤트 ID (식별자)"
                    },
                },
                rename_map=rename_map,
                event_id_col="이벤트 ID (식별자)",
                date_col="대상일",
                name_col="이벤트 이름" if "이벤트 이름" in rename_map.values() else None,
                hospital_col="병원 이름" if "병원 이름" in rename_map.values() else None,
            )
    except Exception:
        pass

    # Amplitude format (header at 4th row)
    _seek_start(file)
    header_df = pd.read_csv(file, encoding="utf-8-sig", skiprows=3, nrows=0)
    raw_header = header_df.columns.tolist()
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
    date_col, date_raw = find_column(
        ["Time", "time", "date", "day", "대상일", "일자", "날짜"]
    )
    event_name_col, event_name_raw = find_column(["event_name", "이벤트명", "이벤트 이름"])
    hospital_col, hospital_raw = find_column(["hospital_name", "병원 이름", "병원명"])

    if event_id_col is None or date_col is None or event_id_raw is None or date_raw is None:
        raise ValueError("지원하지 않는 기획전 성과 CSV 형식입니다.")

    usecols = [event_id_raw, date_raw]
    if event_name_raw is not None:
        usecols.append(event_name_raw)
    if hospital_raw is not None:
        usecols.append(hospital_raw)

    rename_map = {
        event_id_raw: "이벤트 ID (식별자)",
        date_raw: "대상일",
        **({event_name_raw: "이벤트 이름"} if event_name_raw is not None else {}),
        **({hospital_raw: "병원 이름"} if hospital_raw is not None else {}),
    }
    return _scan_meta_chunks(
        file,
        read_csv_kwargs={
            "encoding": "utf-8-sig",
            "skiprows": 3,
            "usecols": usecols,
            "low_memory": False,
            "dtype": {event_id_raw: str},
        },
        rename_map=rename_map,
        event_id_col="이벤트 ID (식별자)",
        date_col="대상일",
        name_col="이벤트 이름" if event_name_raw is not None else None,
        hospital_col="병원 이름" if hospital_raw is not None else None,
    )


def _clean_text(value: str) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).replace('"', "").strip()


def load_primary_data(file) -> pd.DataFrame:
    # Legacy format (header at first row)
    _seek_start(file)
    try:
        raw_header = pd.read_csv(file, encoding="utf-8-sig", nrows=0).columns.tolist()
        clean_to_raw: dict[str, str] = {}
        header: list[str] = []
        for raw in raw_header:
            clean = _clean_text(raw)
            header.append(clean)
            if clean and clean not in clean_to_raw:
                clean_to_raw[clean] = raw
        rename_map_clean, usecols_clean = _resolve_rename_and_usecols(
            header,
            PRIMARY_MIN_COLUMNS,
            {**COLUMN_ALIASES, **PRIMARY_MIN_ALIASES},
        )
        rename_map = {
            clean_to_raw.get(clean, clean): required
            for clean, required in rename_map_clean.items()
        }
        usecols = [clean_to_raw.get(clean, clean) for clean in usecols_clean]
        must_have = {"이벤트 ID (식별자)", "대상일", "조회 수", "상담신청 수"}
        if must_have.issubset(set(rename_map.values())):
            df, info_map = _aggregate_metric_chunks(
                file,
                read_csv_kwargs={
                    "encoding": "utf-8-sig",
                    "usecols": usecols,
                    "low_memory": False,
                    "dtype": {col: str for col, req in rename_map.items() if req == "이벤트 ID (식별자)"},
                },
                rename_map=rename_map,
                group_cols=["이벤트 ID (식별자)", "대상일"],
                metric_cols=["조회 수", "상담신청 수", "CPV 매출", "결제 수"],
                info_extract_cols=[
                    "이벤트 이름",
                    "병원 이름",
                    "대카테고리명",
                    "중카테고리명",
                    "소카테고리명",
                ],
            )
            for metric in ["조회 수", "상담신청 수", "CPV 매출", "결제 수"]:
                if metric in df.columns:
                    df[metric] = df[metric].fillna(0).round().astype("int64")
            if info_map:
                info_df = pd.DataFrame.from_dict(info_map, orient="index").reset_index().rename(
                    columns={"index": "이벤트 ID (식별자)"}
                )
                df = df.merge(info_df, on="이벤트 ID (식별자)", how="left")
            for col in [
                "이벤트 이름",
                "병원 이름",
                "대카테고리명",
                "중카테고리명",
                "소카테고리명",
            ]:
                if col not in df.columns:
                    df[col] = np.nan
            return df
    except Exception:
        pass

    # Amplitude format (header at 4th row)
    _seek_start(file)
    header_df = pd.read_csv(file, encoding="utf-8-sig", skiprows=3, nrows=0)
    raw_header = header_df.columns.tolist()
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
    view_col, view_raw = find_column(["pageview_event.detail--All Users"])
    apply_col, apply_raw = find_column(["apply_event--All Users"])
    event_name_col, event_name_raw = find_column(["event_name", "이벤트명", "이벤트 이름"])
    hospital_col, hospital_raw = find_column(["hospital_name", "병원 이름", "병원명"])

    missing = []
    if event_id_col is None or event_id_raw is None:
        missing.append("event_id")
    if date_col is None or date_raw is None:
        missing.append("Time")
    if view_col is None or view_raw is None:
        missing.append("pageview_event.detail--All Users")
    if apply_col is None or apply_raw is None:
        missing.append("apply_event--All Users")
    if missing:
        raise ValueError(
            "지원하지 않는 기획전 성과 CSV 형식입니다. 누락 컬럼: " + ", ".join(missing)
        )

    usecols = [event_id_raw, date_raw, view_raw, apply_raw]
    if event_name_raw is not None:
        usecols.append(event_name_raw)
    if hospital_raw is not None:
        usecols.append(hospital_raw)

    df, info_map = _aggregate_metric_chunks(
        file,
        read_csv_kwargs={
            "encoding": "utf-8-sig",
            "skiprows": 3,
            "usecols": usecols,
            "low_memory": False,
            "dtype": {event_id_raw: str},
        },
        rename_map={
            event_id_raw: "이벤트 ID (식별자)",
            date_raw: "대상일",
            view_raw: "조회 수",
            apply_raw: "상담신청 수",
            **({event_name_raw: "이벤트 이름"} if event_name_raw is not None else {}),
            **({hospital_raw: "병원 이름"} if hospital_raw is not None else {}),
        },
        group_cols=["이벤트 ID (식별자)", "대상일"],
        metric_cols=["조회 수", "상담신청 수", "CPV 매출", "결제 수"],
        info_extract_cols=[
            "이벤트 이름",
            "병원 이름",
            "대카테고리명",
            "중카테고리명",
            "소카테고리명",
        ],
    )

    for metric in ["조회 수", "상담신청 수", "CPV 매출", "결제 수"]:
        if metric in df.columns:
            df[metric] = df[metric].fillna(0).round().astype("int64")
    if info_map:
        info_df = pd.DataFrame.from_dict(info_map, orient="index").reset_index().rename(
            columns={"index": "이벤트 ID (식별자)"}
        )
        df = df.merge(info_df, on="이벤트 ID (식별자)", how="left")
    for col in [
        "이벤트 이름",
        "병원 이름",
        "대카테고리명",
        "중카테고리명",
        "소카테고리명",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    return df


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
        raise ValueError("지원하지 않는 기획전 성과 CSV 형식입니다.")

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
    event_lookup: dict[str, str],
) -> pd.DataFrame:
    metrics = ["조회 수", "상담신청 수", "CPV 매출", "결제 수"]
    attr_cols = [
        "병원 이름",
        "대카테고리명",
        "중카테고리명",
        "소카테고리명",
    ]
    attr_sources = [
        current_df[["이벤트 ID (식별자)"] + attr_cols],
        previous_df[["이벤트 ID (식별자)"] + attr_cols],
    ]
    attr_info = (
        pd.concat(attr_sources, ignore_index=True)
        .dropna(subset=["이벤트 ID (식별자)"])
        .drop_duplicates(subset=["이벤트 ID (식별자)"])
    )
    id_sources = [
        current_df["이벤트 ID (식별자)"],
        previous_df["이벤트 ID (식별자)"],
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
    summary = summary.merge(attr_info, on="이벤트 ID (식별자)", how="left")

    for metric in metrics:
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

    # 결제 전환율(결제 수 / 조회 수) 지표
    summary["결제 전환율 (진행 기간)"] = np.where(
        summary["조회 수 (진행 기간)"] > 0,
        summary["결제 수 (진행 기간)"] / summary["조회 수 (진행 기간)"] * 100,
        np.nan,
    )
    summary["결제 전환율 (이전 기간)"] = np.where(
        summary["조회 수 (이전 기간)"] > 0,
        summary["결제 수 (이전 기간)"] / summary["조회 수 (이전 기간)"] * 100,
        np.nan,
    )
    summary["결제 전환율 증감"] = (
        summary["결제 전환율 (진행 기간)"] - summary["결제 전환율 (이전 기간)"]
    )

    columns_order = [
        "이벤트 이름",
        "병원 이름",
        "대카테고리명",
        "중카테고리명",
        "소카테고리명",
        "이벤트 ID (식별자)",
        "조회 수 (진행 기간)",
        "조회 수 (이전 기간)",
        "조회 수 증감량",
        "조회 수 증감률",
        "상담신청 수 (진행 기간)",
        "상담신청 수 (이전 기간)",
        "상담신청 수 증감량",
        "상담신청 수 증감률",
        "결제 수 (진행 기간)",
        "결제 수 (이전 기간)",
        "결제 수 증감량",
        "결제 수 증감률",
        "결제 전환율 (진행 기간)",
        "결제 전환율 (이전 기간)",
        "결제 전환율 증감",
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
    add_amount_card("결제 수")
    add_rate_card("결제 수")
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
        purchases_up = int((summary_df.get("결제 수 증감량", 0) > 0).sum())
        views_pct = (views_up / total_events) * 100
        applies_pct = (applies_up / total_events) * 100
        purchases_pct = (purchases_up / total_events) * 100
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
                    {
                        "label": "결제 증가 이벤트",
                        "value": f"{purchases_up:,} ({purchases_pct:.1f}%)",
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
                or "결제 수" in col
            )
            and "증감률" not in col
        )
    ]
    rate_cols = [
        col
        for col in display_df.columns
        if ("증감률" in col) or ("전환율 (진행 기간)" in col) or ("전환율 (이전 기간)" in col)
    ]
    rate_diff_cols = [col for col in display_df.columns if "전환율 증감" in col]

    def _format_int_cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "-"
        if isinstance(value, str) and value.strip() in {"", "-"}:
            return "-"
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric) or numeric == 0:
            return "-"
        return f"{int(round(float(numeric))):,}"

    def _format_rate_cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "-"
        if isinstance(value, str) and value.strip() in {"", "-"}:
            return "-"
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):+.1f}%"

    for col in number_cols:
        display_df[col] = display_df[col].apply(_format_int_cell)
    for col in rate_cols:
        display_df[col] = display_df[col].apply(_format_rate_cell)
    for col in rate_diff_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{float(pd.to_numeric(x, errors='coerce')):+.2f}%p"
            if pd.notna(pd.to_numeric(x, errors="coerce"))
            else "-"
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
기획전 성과 CSV는 퀵사이트 기획전 성과 대시보드에서 raw 파일을 다운로드해 업로드해주세요. 파일을 업로드하면 raw 파일에 포함된 eid가 자동으로 선택됩니다. 기간을 지정한 뒤 '분석 시작'을 눌러주세요. 이 페이지에 문제가 생기면 CRM팀 **@김예슬** 에게 문의해주세요.🍀
"""
)

uploaded_file = st.file_uploader(
    "기획전 성과 CSV 파일을 업로드하세요.", type=["csv"], key="primary_csv"
)

if "analysis_params" not in st.session_state:
    st.session_state.analysis_params = None

file_signature = (uploaded_file.name, uploaded_file.size) if uploaded_file else None
if st.session_state.get("file_signature") != file_signature:
    st.session_state["file_signature"] = file_signature
    st.session_state.analysis_params = None
    st.session_state.pop("primary_meta_sig", None)
    st.session_state.pop("primary_meta_cache", None)

primary_meta = None
primary_error = None

if uploaded_file is not None:
    try:
        primary_sig = (uploaded_file.name, uploaded_file.size)
        cached_primary = st.session_state.get("primary_meta_cache")
        if st.session_state.get("primary_meta_sig") == primary_sig and cached_primary is not None:
            primary_meta = cached_primary
        else:
            with st.spinner("기획전 성과 CSV에서 이벤트/기간 정보를 추출하는 중..."):
                primary_meta = load_primary_meta_info(uploaded_file)
            st.session_state["primary_meta_sig"] = primary_sig
            st.session_state["primary_meta_cache"] = primary_meta
    except Exception as exc:  # noqa: BLE001
        primary_error = f"기획전 성과 CSV 메타 정보를 불러오는 중 오류가 발생했습니다: {exc}"
        st.session_state.pop("primary_meta_sig", None)
        st.session_state.pop("primary_meta_cache", None)

if primary_error:
    st.error(primary_error)

event_ids: set[str] = set()
event_name_map: dict[str, str] = {}
min_ts_candidates: list[pd.Timestamp] = []
max_ts_candidates: list[pd.Timestamp] = []

if primary_meta:
    event_ids |= set(primary_meta.get("event_ids") or set())
    for eid, name in (primary_meta.get("event_name_map") or {}).items():
        event_name_map.setdefault(eid, name)
    if primary_meta.get("min_ts") is not None:
        min_ts_candidates.append(primary_meta["min_ts"])
    if primary_meta.get("max_ts") is not None:
        max_ts_candidates.append(primary_meta["max_ts"])

event_lookup = {eid: (event_name_map.get(eid) or eid) for eid in event_ids}
event_options = sorted(event_lookup.items(), key=lambda x: (x[1] or "", x[0]))
available_ids = list(event_lookup.keys())

today = date.today()
if min_ts_candidates and max_ts_candidates:
    min_date = min(min_ts_candidates).date()
    max_date = max(max_ts_candidates).date()
    can_configure = bool(event_options)
else:
    min_date = today
    max_date = today
    can_configure = False

st.sidebar.header("분석 설정")
if uploaded_file is None:
    st.sidebar.info("먼저 기획전 성과 CSV를 업로드해주세요.")
elif not can_configure:
    st.sidebar.warning("업로드한 파일에서 이벤트/기간 정보를 찾지 못했습니다.")

with st.sidebar.form("analysis_form"):
    with st.expander("이벤트 ID 목록 보기", expanded=False):
        if event_options:
            st.dataframe(
                pd.DataFrame(event_options, columns=["이벤트 ID", "이벤트 이름"]).head(300),
                width="stretch",
            )
        else:
            st.caption("업로드 후 표시됩니다.")
    st.caption("이벤트 ID는 업로드한 CSV의 `id` 컬럼 전체를 자동으로 분석합니다.")
    date_range = st.date_input(
        "분석 기간 (진행 기간)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        disabled=not can_configure,
    )
    submitted = st.form_submit_button("분석 시작", type="primary", disabled=not can_configure)

if submitted and can_configure:
    start_date, end_date = (
        (date_range, date_range) if isinstance(date_range, date) else date_range
    )
    if start_date > end_date:
        st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
        st.stop()
    selected_event_ids = available_ids
    if not selected_event_ids:
        st.sidebar.error("CSV에서 이벤트 ID를 찾지 못했습니다.")
        st.stop()
    st.session_state.analysis_params = {
        "selected_event_ids": selected_event_ids,
        "start_date": start_date,
        "end_date": end_date,
    }

if st.session_state.analysis_params is None:
    if uploaded_file is None:
        st.info("기획전 성과 CSV를 업로드해주세요.")
    else:
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
    st.error(f"기획전 성과 CSV를 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

if uploaded_file is None:
    st.warning("CSV가 업로드되지 않아 분석을 진행할 수 없습니다.")
    st.stop()

event_df = df[df["이벤트 ID (식별자)"].isin(selected_event_ids)].copy()

if event_df.empty:
    st.error("선택한 이벤트에 대한 데이터가 없습니다.")
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
        "label": "결제 수",
        "current": _metric_sum(current_period_df, "결제 수"),
        "previous": _metric_sum(previous_period_df, "결제 수")
        if not previous_period_df.empty
        else None,
    },
    {
        "label": "CPV 매출",
        "current": _metric_sum(current_period_df, "CPV 매출"),
        "previous": _metric_sum(previous_period_df, "CPV 매출")
        if not previous_period_df.empty
        else None,
    },
]

render_metrics(metric_rows)

def _conversion_rate(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    views = _metric_sum(df, "조회 수")
    purchases = _metric_sum(df, "결제 수")
    if views is None or purchases is None or views == 0:
        return None
    return (purchases / views) * 100


current_conversion = _conversion_rate(current_period_df)
previous_conversion = _conversion_rate(previous_period_df)
conv_col = st.columns(1)[0]
conv_delta = None
if current_conversion is not None and previous_conversion is not None:
    conv_delta = f"{(current_conversion - previous_conversion):+.2f}%p"
conv_col.metric(
    "결제 전환율 (결제 수 / 조회 수)",
    f"{current_conversion:.2f}%" if current_conversion is not None else "-",
    delta=conv_delta or "비교 데이터 없음",
)

event_summary_df = build_event_summary(
    current_period_df,
    previous_period_df,
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
