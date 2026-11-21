import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="극지식물 개체별 생육 차이 분석", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner="CSV를 읽고 있어요...")
def read_csv_cached(file):
    return read_csv(file)


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


def mean_or_none(df, col):
    if df is None or col not in df.columns:
        return None
    v = df[col].dropna()
    if len(v) == 0:
        return None
    return round(float(v.mean()), 2)


def std_or_none(df, col):
    if df is None or col not in df.columns:
        return None
    v = df[col].dropna()
    if len(v) == 0:
        return None
    return round(float(v.std(ddof=0)), 2)


def cv_or_none(df, col):
    m = mean_or_none(df, col)
    s = std_or_none(df, col)
    if m is None or s is None or m == 0:
        return None
    return round(float(s / m * 100), 1)


def individual_bars(g, metrics, height=700):
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(metrics.values()))
    for i, (label, col) in enumerate(metrics.items(), start=1):
        if col not in g.columns:
            continue
        vals = g[col].dropna()
        if len(vals) == 0:
            continue
        colors = ["#90A4AE"] * len(g)
        if col == "생중량":
            top5 = set(g.nlargest(5, "생중량")["개체번호"])
            bottom5 = set(g.nsmallest(5, "생중량")["개체번호"])
            colors = ["#FF8A65" if (k in top5 or k in bottom5) else "#90A4AE" for k in g["개체번호"]]
        fig.add_trace(go.Bar(x=g["개체번호"], y=g[col], marker_color=colors), row=(i - 1) // 2, col=(i - 1) % 2 + 1)
    fig.update_layout(height=height, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    return fig


env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

env_df = None
growth_df = None
school_name = "우리 학교"

if env_file is not None:
    env_df = read_csv_cached(env_file)
    env_df = std_cols(env_df)
    school_name = env_file.name.rsplit(".", 1)[0]
if growth_file is not None:
    growth_df = read_csv_cached(growth_file)
    growth_df = std_cols(growth_df)
    if school_name == "우리 학교":
        school_name = growth_file.name.rsplit(".", 1)[0]

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
        st.caption(f"학교: {school_name}")

        st.subheader("🔬 연구 배경 및 목적")
        st.write(
            "극지 환경의 식물은 온도·습도 같은 환경 조건에 따라 생장이 달라질 수 있어요. "
            "이 대시보드는 우리 학교 극지식물의 개체별 생육 차이를 비교하고, "
            "환경 데이터와 생육 결과의 관계를 살펴보기 위해 만들었어요."
        )
        st.write(
            "**용어 안내** — 평균: 값을 모두 더해 개체 수로 나눈 값, 표준편차: 값들이 평균에서 얼마나 퍼져 있는지, "
            "CV(변동계수): 표준편차를 평균으로 나눈 값(%), 상관계수: 두 변수가 함께 변하는지 나타내는 값(-1~1)."
        )

        st.subheader("📋 데이터 요약")
        rows = []
        if env_df is not None and len(env_df) > 0:
            period = "측정 기간 없음"
            if "time" in env_df.columns:
                tvals = env_df["time"].dropna()
                if len(tvals) > 0:
                    period = f"{tvals.min():%Y-%m-%d %H:%M} ~ {tvals.max():%Y-%m-%d %H:%M}"
            rows.append(
                {
                    "데이터": "환경 데이터",
                    "측정 수": len(env_df),
                    "측정 기간": period,
                    "측정 항목": ", ".join(
                        [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"] if c in env_df.columns]
                    ),
                }
            )
        if growth_df is not None and len(growth_df) > 0:
            n_plants = growth_df["개체번호"].nunique() if "개체번호" in growth_df.columns else len(growth_df)
            rows.append(
                {
                    "데이터": "생육 결과",
                    "개체 수": n_plants,
                    "측정 기간": "측정 기간 없음",
                    "측정 항목": ", ".join(
                        [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")

        st.subheader("📊 주요 지표")
        avg_temp = mean_or_none(env_df, "temperature")
        avg_hum = mean_or_none(env_df, "humidity")
        avg_gw = mean_or_none(growth_df, "생중량")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_plants = growth_df["개체번호"].nunique() if growth_df is not None and "개체번호" in growth_df.columns else 0
            st.metric("총 개체수", n_plants if n_plants > 0 else "—")
        with c2:
            st.metric("평균 온도", f"{avg_temp:.1f} °C" if avg_temp is not None else "—")
        with c3:
            st.metric("평균 습도", f"{avg_hum:.1f} %" if avg_hum is not None else "—")
        with c4:
            st.metric("평균 생중량", f"{avg_gw:.2f} g" if avg_gw is not None else "—")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        st.title("🌡️ 환경 데이터")
        if env_df is None or len(env_df) == 0:
            st.info("환경 데이터가 아직 업로드되지 않았어요. 사이드바에서 환경 CSV를 올려주세요.")
        else:
            st.caption(f"학교: {school_name}")

            st.subheader("📈 환경 요약 (평균·최소·최대)")
            num_cols = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"] if c in env_df.columns]
            if num_cols:
                summary = env_df[num_cols].describe().loc[["mean", "min", "max"]].T
                summary.columns = ["평균", "최소", "최대"]
                st.dataframe(summary, width="stretch")
            else:
                st.info("온도·습도 같은 환경 측정 항목이 없어요.")

            st.subheader("📅 날짜별 시계열 (일별 평균)")
            if "time" in env_df.columns:
                df_t = env_df.dropna(subset=["time"]).copy()
                df_t["date"] = df_t["time"].dt.date
                agg = {}
                if "temperature" in df_t.columns:
                    agg["temperature"] = "mean"
                if "humidity" in df_t.columns:
                    agg["humidity"] = "mean"
                if "ph" in df_t.columns:
                    agg["ph"] = "mean"
                if "ec" in df_t.columns:
                    agg["ec"] = "mean"
                if agg:
                    daily = df_t.groupby("date", as_index=False).agg(agg)
                    fig = make_subplots(
                        rows=2,
                        cols=2,
                        subplot_titles=("온도 변화", "습도 변화", "pH 변화", "EC 변화"),
                    )
                    if "temperature" in daily.columns:
                        fig.add_trace(go.Scatter(x=daily["date"], y=daily["temperature"], mode="lines+markers", name="온도"), row=1, col=1)
                    if "humidity" in daily.columns:
                        fig.add_trace(go.Scatter(x=daily["date"], y=daily["humidity"], mode="lines+markers", name="습도"), row=1, col=2)
                    if "ph" in daily.columns:
                        fig.add_trace(go.Scatter(x=daily["date"], y=daily["ph"], mode="lines+markers", name="pH"), row=2, col=1)
                    if "ec" in daily.columns:
                        fig.add_trace(go.Scatter(x=daily["date"], y=daily["ec"], mode="lines+markers", name="EC"), row=2, col=2)
                    missing = []
                    if "ph" not in env_df.columns:
                        missing.append("pH")
                    if "ec" not in env_df.columns:
                        missing.append("EC")
                    if missing:
                        st.info(f"⚠️ 이 학교에는 {', '.join(missing)} 데이터가 없어서 시계열에서 제외했어요.")
                    fig.update_layout(height=700, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
                    st.plotly_chart(fig, width="stretch")

            with st.expander("환경 데이터 원본 테이블"):
                st.dataframe(env_df, width="stretch")
            csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("환경 데이터 다운로드", csv_env, file_name=f"{school_name}_환경데이터.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        st.title("📊 생육 결과")
        if growth_df is None or len(growth_df) == 0:
            st.info("생육 데이터가 아직 업로드되지 않았어요. 사이드바에서 생육 CSV를 올려주세요.")
        else:
            st.caption(f"학교: {school_name}")

            st.subheader("🥇 핵심 결과 카드")
            g = growth_df.dropna(subset=["개체번호", "생중량"]).copy()
            if len(g) > 0:
                g["개체번호"] = g["개체번호"].astype(str)
                avg_gw = g["생중량"].mean()
                max_idx = g["생중량"].idxmax()
                min_idx = g["생중량"].idxmin()
                st.metric("평균 생중량", f"{avg_gw:.2f} g")
                st.metric("최대 개체", g.loc[max_idx, "개체번호"])
                st.metric("최소 개체", g.loc[min_idx, "개체번호"])

                sd_gw = std_or_none(g, "생중량")
                cv_gw = cv_or_none(g, "생중량")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "표준편차 (g)": [f"{sd_gw:.2f}" if sd_gw is not None else "—"],
                            "CV (%)": [f"{cv_gw:.1f}" if cv_gw is not None else "—"],
                        }
                    ),
                    width="stretch",
                )

            st.subheader("📊 개체별 생육 비교 (막대 그래프)")
            if len(g) > 0:
                bar_metrics = {
                    "평균 생중량 (g)": "생중량",
                    "평균 잎 수 (장)": "잎 수",
                    "평균 지상부 길이 (mm)": "지상부 길이",
                    "평균 지하부 길이 (mm)": "지하부 길이",
                }
                st.plotly_chart(individual_bars(g, bar_metrics), width="stretch")

            st.subheader("📦 생중량 분포")
            if "생중량" in g.columns and len(g) > 0:
                fig = px.box(g, y="생중량", points="outliers")
                fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
                st.plotly_chart(fig, width="stretch")

            st.subheader("🔗 잎 수·지상부 길이 vs 생중량 (상관관계)")
            for label, xcol in [("잎 수 vs 생중량", "잎 수"), ("지상부 길이 vs 생중량", "지상부 길이")]:
                if xcol not in g.columns or "생중량" not in g.columns:
                    st.info(f"⚠️ {label} 분석을 위한 데이터가 부족해요.")
                    continue
                gc = g.dropna(subset=[xcol, "생중량"])
                if len(gc) < 2:
                    st.info(f"⚠️ {label}: 값이 2개 미만이라 추세선을 그릴 수 없어요.")
                    continue
                fig = px.scatter(gc, x=xcol, y="생중량")
                coef = np.polyfit(gc[xcol], gc["생중량"], 1)
                xs = np.linspace(gc[xcol].min(), gc[xcol].max(), 100)
                fig.add_trace(go.Scatter(x=xs, y=np.poly1d(coef)(xs), mode="lines", name="추세선", line=dict(color="red", dash="dash")))
                fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
                st.plotly_chart(fig, width="stretch")

            st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")
            if "생중량" in g.columns and len(g) >= 4:
                q = max(1, len(g) // 4)
                top = g.nlargest(q, "생중량")
                bottom = g.nsmallest(q, "생중량")
                rows = [
                    {
                        "구분": "상위 25%",
                        "개체 수": len(top),
                        "평균 잎 수 (장)": float(top["잎 수"].mean()) if "잎 수" in top.columns else None,
                        "평균 지상부 길이 (mm)": float(top["지상부 길이"].mean()) if "지상부 길이" in top.columns else None,
                        "평균 지하부 길이 (mm)": float(top["지하부 길이"].mean()) if "지하부 길이" in top.columns else None,
                        "평균 생중량 (g)": float(top["생중량"].mean()),
                    },
                    {
                        "구분": "하위 25%",
                        "개체 수": len(bottom),
                        "평균 잎 수 (장)": float(bottom["잎 수"].mean()) if "잎 수" in bottom.columns else None,
                        "평균 지상부 길이 (mm)": float(bottom["지상부 길이"].mean()) if "지상부 길이" in bottom.columns else None,
                        "평균 지하부 길이 (mm)": float(bottom["지하부 길이"].mean()) if "지하부 길이" in bottom.columns else None,
                        "평균 생중량 (g)": float(bottom["생중량"].mean()),
                    },
                ]
                comp = pd.DataFrame(rows)
                st.dataframe(comp, width="stretch")
                fig = go.Figure(
                    go.Bar(
                        x=comp["구분"],
                        y=comp["평균 생중량 (g)"],
                        marker_color=["#FF8A65", "#90A4AE"],
                    )
                )
                fig.update_layout(
                    title="평균 생중량: 상위 25% vs 하위 25%",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                )
                st.plotly_chart(fig, width="stretch")

            with st.expander("생육 데이터 원본 테이블"):
                st.dataframe(growth_df, width="stretch")
            csv_gw = growth_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("생육 데이터 다운로드", csv_gw, file_name=f"{school_name}_생육결과데이터.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
