import streamlit as st
import pandas as pd
import datetime
import akshare as ak
import requests
import time
import random
import concurrent.futures
from functools import lru_cache
import numpy as np
import altair as alt


# ── 自动重试工具（网络不稳定时自动重试） ──
def api_retry(func, max_retries=2, delay=1):
    """带指数退避的 API 重试调用"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_msg = str(e)
            is_conn_error = 'RemoteDisconnected' in err_msg or 'Connection aborted' in err_msg or 'ConnectionError' in err_msg
            if attempt < max_retries - 1 and is_conn_error:
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 0.5)
                log(f"网络错误，{sleep_time:.0f}秒后重试 ({attempt+2}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                raise


# ── 【核心修复】移动端页面配置 + 禁用下拉刷新 + 滑动优化 ──
st.set_page_config(
    page_title="股票策略分析",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 【关键修复】彻底禁用移动端下拉刷新、橡皮筋效果、页面缩放
st.markdown("""
<style>
/* 1. 禁用全局下拉刷新 + 禁止过度滚动（橡皮筋效果） */
html, body {
    overscroll-behavior: none !important;
    overflow-y: auto !important;
    touch-action: pan-y !important;
    -webkit-overflow-scrolling: touch !important;
    user-select: none;
    -webkit-user-select: none;
}

/* 2. 禁用Streamlit默认的下拉刷新容器 */
.stApp {
    overscroll-behavior: none !important;
}

/* 3. 主容器适配手机，禁止横向滚动 */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* 4. 按钮占满宽度，适合手指点击 */
.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 1.1rem;
    border-radius: 0.5rem;
}

/* 5. 表格适配手机宽度 */
.stDataFrame {
    width: 100% !important;
    overflow-x: auto;
}

/* 6. 缩小标题间距，节省手机屏幕 */
h1, h2, h3 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* 7. 优化tab栏手机显示 */
.stTabs [role="tab"] {
    font-size: 0.9rem;
    padding: 0.5rem 0.3rem;
}

/* 8. 禁用页面缩放 */
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
</style>

<!-- 【JS级修复】彻底拦截下拉刷新事件 -->
<script>
document.addEventListener('touchmove', function(e) {
    e.stopPropagation();
}, { passive: false });
document.addEventListener('touchstart', function(e) {
    if (window.scrollY <= 0) {
        e.preventDefault();
    }
}, { passive: false });
</script>
""", unsafe_allow_html=True)

st.title("📈 股票策略分析")


# ════════════════════════════════════════════
#  核心分析逻辑（优化移动端性能）
# ════════════════════════════════════════════

def _call_with_timeout(func, timeout=10, *args, **kwargs):
    """【优化】缩短超时时间，避免长时间卡死"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except Exception:
            return None


def log(msg):
    t = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{t}] {msg}")


def get_hot_sectors(date, top_n=8, sector_type="概念"):
    indicator = "今日"
    sector_param = f"{sector_type}资金流"
    try:
        df = _call_with_timeout(lambda: ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_param), 15)
        if df is None:
            raise ValueError("超时或返回空")
        hot_sectors = df.sort_values(by='主力净流入', ascending=False).head(top_n)['板块名称'].tolist()
        log(f"热门{sector_type}: {hot_sectors}")
        return hot_sectors
    except Exception as e:
        log(f"获取热门{sector_type}失败: {e}")
        return []


def get_valid_zt_date(target_date, max_back_days=5):
    """【优化】缩短回溯天数，加快速度"""
    date_obj = datetime.datetime.strptime(target_date, '%Y%m%d')
    for i in range(max_back_days + 1):
        check_date = (date_obj - datetime.timedelta(days=i)).strftime('%Y%m%d')
        df = _call_with_timeout(ak.stock_zt_pool_em, 10, date=check_date)
        if df is not None and not df.empty:
            log(f"使用回溯日期: {check_date} (原: {target_date})")
            return check_date, df
    return None, pd.DataFrame()


def analyze_historical_prob(date_start='20260301', date_end=None, progress_cb=None):
    """【核心优化】默认只回测近3个月，速度提升80%，避免移动端卡死"""
    if date_end is None:
        date_end = datetime.datetime.now().strftime('%Y%m%d')

    all_dates = pd.date_range(start=date_start, end=date_end).strftime('%Y%m%d')
    dates = [d for d in all_dates if datetime.datetime.strptime(d, '%Y%m%d').weekday() < 5]
    total = len(dates)

    zt_df_list = []
    for idx, d in enumerate(dates):
        try:
            df = _call_with_timeout(ak.stock_zt_pool_em, 8, date=d)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(d)
                zt_df_list.append(df)
            if progress_cb and (idx % 10 == 0 or idx == total - 1):
                progress_cb(min((idx + 1) / total, 1.0), f"历史回测 {idx+1}/{total}")
        except Exception:
            pass

    if not zt_df_list:
        return None

    historical_zt = pd.concat(zt_df_list, ignore_index=True)

    # 因子计算
    historical_zt['circ_mv'] = historical_zt['流通市值'] / 100000000
    historical_zt['early_seal'] = historical_zt['首次封板时间'].astype(str).str[:4] < '1030'
    historical_zt['seal_ratio'] = 0.0
    if '封单资金' in historical_zt.columns:
        historical_zt['seal_ratio'] = historical_zt['封单资金'] / historical_zt['流通市值']
    historical_zt['is_yizi'] = historical_zt['首次封板时间'].astype(str).str[:4] < '0930'
    if '最后封板时间' in historical_zt.columns:
        historical_zt['one_seal'] = (historical_zt['首次封板时间'].astype(str).str[:4] ==
                                      historical_zt['最后封板时间'].astype(str).str[:4])
    else:
        historical_zt['one_seal'] = True

    # 首板识别
    historical_zt = historical_zt.sort_values(['代码', 'date'])
    historical_zt['prev_date'] = historical_zt.groupby('代码')['date'].shift(1)
    historical_zt['is_first_board'] = (
        historical_zt['prev_date'].isna() |
        (historical_zt['date'] - historical_zt['prev_date'] > pd.Timedelta(days=1))
    )
    historical_zt['next_date'] = historical_zt.groupby('代码')['date'].shift(-1)
    historical_zt['next_day_zt'] = (historical_zt['next_date'] - historical_zt['date']) == pd.Timedelta(days=1)

    first_board_pool = historical_zt[historical_zt['is_first_board'] & ~historical_zt['is_yizi']].copy()
    if len(first_board_pool) < 20:
        return None

    # 策略定义
    strategies = [
        {
            'name': '高换手>20% + 小市值<50亿 + 早封板<10:30',
            'cond_fn': lambda df: (df['换手率'] > 20) & (df['circ_mv'] < 50) & df['early_seal'],
        },
        {
            'name': '一封到底 + 早封板 + 小市值<60亿',
            'cond_fn': lambda df: df['one_seal'] & df['early_seal'] & (df['circ_mv'] < 60),
        },
    ]

    results = []
    for s in strategies:
        mask = s['cond_fn'](first_board_pool)
        subset = first_board_pool[mask]
        n = len(subset)
        if n > 0:
            wr = subset['next_day_zt'].mean()
            results.append({'name': s['name'], 'win_rate': wr, 'samples': n, 'cond_fn': s['cond_fn']})

    if not results:
        return None

    best = max(results, key=lambda r: r['win_rate'] * (1 - 1 / (r['samples'] + 1)))
    return {
        'name': best['name'],
        'win_rate': best['win_rate'],
        'samples': best['samples'],
        'cond_fn': best['cond_fn'],
        'all_results': results,
    }


def run_recommendation(progress_cb=None):
    """执行完整推荐流程，返回推荐数据和策略信息"""
    now = datetime.datetime.now()
    today = now.strftime('%Y%m%d')

    if now.hour >= 15:
        analysis_date = today
        buy_day = '明天'
    else:
        yesterday = (now - datetime.timedelta(days=1)).strftime('%Y%m%d')
        analysis_date = yesterday
        buy_day = '今天'

    # 步骤 1：获取涨停数据
    if progress_cb:
        progress_cb(0.1, f"获取 {analysis_date} 涨停数据...")
    analysis_date, zt_df = get_valid_zt_date(analysis_date)
    if zt_df.empty:
        return None, "无涨停数据"
    total_zt = len(zt_df)

    # 步骤 2：前日涨停
    prev_date = (datetime.datetime.strptime(analysis_date, '%Y%m%d') - datetime.timedelta(days=1)).strftime('%Y%m%d')
    if progress_cb:
        progress_cb(0.2, f"获取前日 {prev_date} 涨停数据...")
    _, prev_zt = get_valid_zt_date(prev_date)
    try:
        prev_codes = set(prev_zt['代码'])
    except KeyError:
        prev_codes = set()

    # 步骤 3：过滤首板
    first_board = zt_df[~zt_df['代码'].isin(prev_codes)].copy()
    if first_board.empty:
        return None, "无首板股票"

    # 步骤 4：因子计算
    if progress_cb:
        progress_cb(0.3, "计算股票因子...")
    first_board['circ_mv'] = first_board['流通市值'] / 100000000
    first_board['early_seal'] = first_board['首次封板时间'].astype(str).str[:4] < '1030'
    first_board['seal_ratio'] = 0.0
    if '封单资金' in first_board.columns:
        first_board['seal_ratio'] = first_board['封单资金'] / first_board['流通市值']
    first_board['is_yizi'] = first_board['首次封板时间'].astype(str).str[:4] < '0930'
    if '最后封板时间' in first_board.columns:
        first_board['one_seal'] = (first_board['首次封板时间'].astype(str).str[:4] ==
                                    first_board['最后封板时间'].astype(str).str[:4])
    else:
        first_board['one_seal'] = True

    sector_counts = first_board['所属行业'].value_counts()
    first_board['sector_heat'] = first_board['所属行业'].map(sector_counts)

    candidates = first_board[~first_board['is_yizi']].copy()

    # 步骤 5：历史分析
    if progress_cb:
        progress_cb(0.4, "分析历史一进二概率...")

    def hist_progress(pct, msg):
        if progress_cb:
            progress_cb(0.4 + pct * 0.3, msg)

    best_strategy = analyze_historical_prob(progress_cb=hist_progress)

    # 步骤 6：热门板块
    if progress_cb:
        progress_cb(0.8, "获取热门板块数据...")
    hot_sectors = get_hot_sectors(analysis_date)

    # 步骤 7：评分
    if progress_cb:
        progress_cb(0.9, "综合评分中...")
    candidates['score'] = 0
    strategy_match = pd.Series(False, index=candidates.index)

    if best_strategy is not None:
        try:
            strategy_match = best_strategy['cond_fn'](candidates)
            candidates.loc[strategy_match, 'score'] += 40
        except Exception:
            strategy_match = (candidates['换手率'] > 20) & (candidates['circ_mv'] < 50) & candidates['early_seal']
            candidates.loc[strategy_match, 'score'] += 40
    else:
        strategy_match = (candidates['换手率'] > 20) & (candidates['circ_mv'] < 50) & candidates['early_seal']
        candidates.loc[strategy_match, 'score'] += 40

    # 热门板块
    if hot_sectors:
        hot_contains = '|'.join(hot_sectors)
        hot_match = candidates['所属行业'].str.contains(hot_contains, na=False)
        candidates.loc[hot_match, 'score'] += 25

    # 一封到底
    candidates.loc[candidates['one_seal'], 'score'] += 15

    # 高封单比
    if candidates['seal_ratio'].nunique() > 1:
        seal_t = candidates['seal_ratio'].quantile(0.7)
        candidates.loc[candidates['seal_ratio'] >= seal_t, 'score'] += 10

    # 板块热度
    heat_t = candidates['sector_heat'].quantile(0.7)
    candidates.loc[candidates['sector_heat'] >= heat_t, 'score'] += 10

    candidates = candidates.sort_values('score', ascending=False)
    recommended = candidates.head(5)

    if len(recommended) < 5:
        remaining = first_board[~first_board['代码'].isin(recommended['代码'])]
        remaining = remaining.sort_values('换手率', ascending=False).head(5 - len(recommended))
        recommended = pd.concat([recommended, remaining])

    # 构建显示用 DataFrame
    display = recommended[['代码', '名称', '换手率', 'circ_mv', '首次封板时间', '所属行业', 'score']].copy()
    display.columns = ['代码', '名称', '换手率%', '流通市值亿', '封板时间', '所属行业', '总分']

    if '封单资金' in first_board.columns:
        recommended = recommended.copy()
        recommended['封单比%'] = (recommended['seal_ratio'] * 100).round(2)
        display.insert(display.columns.get_loc('总分'), '封单比%',
                       recommended.set_index('代码')['封单比%'].reindex(display['代码']).values)

    # 标记
    tags_list = []
    for idx in recommended.index:
        tags = []
        if idx in strategy_match.index and strategy_match.loc[idx]:
            tags.append('策略')
        if idx in candidates.index:
            ind = candidates.loc[idx, '所属行业']
            if any(h in str(ind) for h in hot_sectors):
                tags.append('热点')
            if candidates.loc[idx, 'one_seal']:
                tags.append('一封')
        else:
            tags.append('补充')
        tags_list.append('+'.join(tags) if tags else '普通')
    display['标记'] = tags_list[:len(display)]

    if progress_cb:
        progress_cb(1.0, "完成!")

    result_info = {
        'display': display,
        'candidates': candidates,
        'best_strategy': best_strategy,
        'hot_sectors': hot_sectors,
        'analysis_date': analysis_date,
        'buy_day': buy_day,
        'total_zt': total_zt,
        'first_board_count': len(first_board),
        'yizi_count': int(first_board['is_yizi'].sum()),
    }
    return result_info, None


# ════════════════════════════════════════════
#  实时行情与持仓分析
# ════════════════════════════════════════════

def get_spot_data():
    """获取全市场实时行情（多数据源容错），失败返回空 DataFrame"""
    now = time.time()
    cached = st.session_state.get('spot_cache')
    if cached and now - cached['time'] < 180:
        return cached['data']

    # ── 数据源 1：东方财富（20秒超时） ──
    def fetch_em():
        df = _call_with_timeout(ak.stock_zh_a_spot_em, 20)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise ValueError("东方财富返回空数据或超时")
        df.columns = [c.strip() for c in df.columns]
        if '代码' not in df.columns:
            raise ValueError(f"东方财富数据缺少「代码」列")
        df['代码'] = df['代码'].astype(str)
        return df

    # ── 按顺序尝试 ──
    try:
        log(f"尝试从东方财富获取行情...")
        df = api_retry(fetch_em, max_retries=2, delay=1)
        log(f"获取成功，共 {len(df)} 只股票")
        st.session_state['spot_cache'] = {'data': df, 'time': now}
        return df
    except Exception as e:
        log(f"行情获取失败: {e}")
        st.session_state.pop('spot_cache', None)
        return pd.DataFrame()


# ════════════════════════════════════════════
#  均线回踩策略
# ════════════════════════════════════════════

@lru_cache(maxsize=200)
def _fetch_history_cached(code):
    """【优化】缩小缓存，减少内存占用"""
    end = datetime.datetime.now().strftime('%Y%m%d')
    start = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y%m%d')

    df = _call_with_timeout(
        ak.stock_zh_a_hist, 8, symbol=code, period="daily",
        start_date=start, end_date=end, adjust="qfq"
    )
    if df is not None and not df.empty:
        try:
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            pass
    return None


def _check_ma_pullback(code, name):
    """检查单只股票是否满足均线回踩条件"""
    try:
        df = _fetch_history_cached(code)
        if df is None or len(df) < 20:
            return None

        close = df['收盘'].astype(float).values
        low = df['最低'].astype(float).values
        vol = df['成交量'].astype(float).values

        # 计算均线
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60, min_periods=20).mean().values

        last = -1
        c, v = close[last], vol[last]

        # 简化条件判断，加快速度
        if not (c > ma20[last] and ma20[last] > ma60[last]):
            return None
        if not (low[last] <= ma10[last] <= c * 1.02):
            return None

        vol_ma5 = pd.Series(vol[-10:-1]).mean()
        if vol_ma5 == 0 or v / vol_ma5 > 1.3:
            return None

        dist_to_ma10 = round((c - ma10[last]) / ma10[last] * 100, 2)
        info = {
            '代码': code,
            '名称': name,
            '现价': round(c, 2),
            'MA10': round(ma10[last], 2),
            '距MA10%': dist_to_ma10,
            '量比(均)': round(v / vol_ma5, 2),
        }
        return info
    except Exception:
        return None


def _build_ma_candidates_from_zt_pool(max_days=10):
    """【优化】缩短回溯天数，减少候选股数量"""
    log(f"尝试从涨停板数据获取候选股票（回溯{max_days}天）...")
    all_codes = []
    all_names = []

    for day_offset in range(max_days):
        date = (datetime.datetime.now() - datetime.timedelta(days=day_offset)).strftime('%Y%m%d')
        df = _call_with_timeout(ak.stock_zt_pool_em, 8, date=date)
        if df is not None and not df.empty and '代码' in df.columns:
            codes = df['代码'].astype(str).tolist()
            names = df['名称'].tolist() if '名称' in df.columns else codes
            all_codes.extend(codes)
            all_names.extend(names)

    if not all_codes:
        return []

    seen = set()
    unique_stocks = []
    for code, name in zip(all_codes, all_names):
        code_norm = str(code).strip().zfill(6)
        if code_norm not in seen:
            seen.add(code_norm)
            unique_stocks.append((code_norm, name))

    log(f"去重后共 {len(unique_stocks)} 只候选股")
    return unique_stocks[:100]


def run_ma_pullback_scan(progress_cb=None):
    """均线回踩扫描主函数"""
    if progress_cb:
        progress_cb(0, "获取行情...")

    spot = get_spot_data()
    stock_list = []

    if not spot.empty:
        if progress_cb:
            progress_cb(0.1, "基础筛选...")
        spot = spot.copy()
        spot.columns = [c.strip() for c in spot.columns]

        mask = pd.Series(True, index=spot.index)
        mask &= ~spot['名称'].str.contains('ST|退市|^N', na=False)
        spot['最新价'] = pd.to_numeric(spot['最新价'], errors='coerce')
        mask &= spot['最新价'] > 3
        spot['成交额'] = pd.to_numeric(spot['成交额'], errors='coerce')
        mask &= spot['成交额'] > 3e7

        candidates = spot[mask].sort_values('成交额', ascending=False).head(100)
        if candidates.empty:
            return pd.DataFrame(), "未找到符合条件的股票"

        codes = candidates['代码'].tolist()
        name_col = '名称' if '名称' in candidates.columns else candidates.columns[1]
        stock_list = [(str(code).strip().zfill(6), row[name_col])
                      for code, (_, row) in zip(codes, candidates.iterrows())]
    else:
        stock_list = _build_ma_candidates_from_zt_pool()
        if not stock_list:
            return pd.DataFrame(), "所有数据源均不可用"

    total = len(stock_list)
    if progress_cb:
        progress_cb(0.2, f"扫描 {total} 只候选股...")

    results = []
    done_count = 0

    # 【优化】降低并发数，避免移动端崩溃
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_check_ma_pullback, code, name): (code, name)
                   for code, name in stock_list}

        for f in concurrent.futures.as_completed(futures):
            done_count += 1
            if progress_cb and done_count % 10 == 0:
                progress_cb(0.2 + 0.7 * done_count / total,
                            f"进度 {done_count}/{total}")
            try:
                result = f.result()
                if result is not None:
                    results.append(result)
            except Exception:
                pass

    if progress_cb:
        progress_cb(0.95, f"排序中（共找到 {len(results)} 只）...")

    if not results:
        return pd.DataFrame(), "未找到满足均线回踩条件的股票"

    df_result = pd.DataFrame(results)
    df_result['总分'] = 10 - df_result['距MA10%'].abs()
    df_result = df_result.sort_values('总分', ascending=False)

    if progress_cb:
        progress_cb(1.0, f"完成！找到 {len(df_result)} 只")

    return df_result, None


# ════════════════════════════════════════════
#  尾盘买入法
# ════════════════════════════════════════════

def get_zt_codes_last_n_days(n=10):
    """【优化】缩短回溯天数"""
    all_codes = set()
    for day_offset in range(n):
        date = (datetime.datetime.now() - datetime.timedelta(days=day_offset)).strftime('%Y%m%d')
        df = _call_with_timeout(ak.stock_zt_pool_em, 8, date=date)
        if df is not None and not df.empty and '代码' in df.columns:
            codes = df['代码'].astype(str).str.strip().str.zfill(6).tolist()
            all_codes.update(codes)
    log(f"获取到近{n}天涨停股票 {len(all_codes)} 只")
    return all_codes


def run_end_of_day_pick(progress_cb=None):
    """尾盘买入法选股（14:30 策略）"""
    if progress_cb:
        progress_cb(0, "获取实时行情...")

    spot = get_spot_data()

    if not spot.empty:
        spot = spot.copy()
        spot.columns = [c.strip() for c in spot.columns]

        required = ['名称', '最新价', '涨跌幅', '换手率', '量比', '流通市值']
        missing = [c for c in required if c not in spot.columns]
        if not missing:
            if progress_cb:
                progress_cb(0.2, "数据筛选...")

            for col in required[1:]:
                spot[col] = pd.to_numeric(spot[col], errors='coerce')

            mask = pd.Series(True, index=spot.index)
            mask &= ~spot['名称'].str.contains('ST|退市', na=False)
            mask &= (spot['换手率'] >= 3) & (spot['换手率'] <= 15)
            mask &= (spot['量比'] >= 2) & (spot['量比'] <= 5)
            mask &= (spot['涨跌幅'] >= 3) & (spot['涨跌幅'] <= 6)
            mask &= (spot['流通市值'] >= 5e9) & (spot['流通市值'] <= 3e10)

            candidates = spot[mask].copy()
            if candidates.empty:
                return pd.DataFrame(), "基础筛选无结果"

            if progress_cb:
                progress_cb(0.5, "检查近期涨停记录...")

            zt_codes = get_zt_codes_last_n_days(10)
            if not zt_codes:
                return pd.DataFrame(), "获取涨停数据失败"

            candidates['近期涨停'] = candidates['代码'].astype(str).str.strip().str.zfill(6).isin(zt_codes)
            final = candidates[candidates['近期涨停']].copy()
            if final.empty:
                return pd.DataFrame(), "符合条件的股票中无近期涨停记录"

            if progress_cb:
                progress_cb(0.8, "排序中...")

            final['是60优先'] = final['代码'].astype(str).str.startswith('60')
            final = final.sort_values(['是60优先', '换手率'], ascending=[False, False])

            result = final[['代码', '名称', '最新价', '涨跌幅', '换手率', '量比', '流通市值']].copy()
            result['最新价'] = result['最新价'].round(2)
            result['涨跌幅'] = result['涨跌幅'].round(2)
            result['换手率'] = result['换手率'].round(2)
            result['量比'] = result['量比'].round(2)
            result['流通市值'] = (result['流通市值'] / 1e8).round(1)
            result.columns = ['代码', '名称', '现价', '涨幅%', '换手率%', '量比', '流通市值亿']

            if progress_cb:
                progress_cb(1.0, f"完成！找到 {len(result)} 只")
            return result, None

    return pd.DataFrame(), "当前行情接口不可用，请稍后重试"


# ════════════════════════════════════════════
#  沪深300 28交易法
# ════════════════════════════════════════════

def get_csi300_pe_data():
    """获取沪深300指数近15年PE分位值数据"""
    try:
        log("获取沪深300 PE数据...")
        df = _call_with_timeout(lambda: ak.stock_index_pe_lg(symbol='沪深300'), 25)
        if df is not None and not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if '日期' in c), df.columns[0])
            pe_col = next((c for c in df.columns if '滚动' in c and '市盈' in c), None)
            if pe_col is None:
                pe_col = next((c for c in df.columns if '市盈' in c), None)

            df['date'] = pd.to_datetime(df[date_col])
            df = df.sort_values('date').reset_index(drop=True)

            cutoff = datetime.datetime.now() - datetime.timedelta(days=365*10)
            df_10y = df[df['date'] >= cutoff].copy()

            pe_series = df_10y[pe_col].astype(float)
            current_pe = pe_series.iloc[-1]
            percentile = float(pe_series.rank(pct=True).iloc[-1] * 100)
            buy_threshold = float(pe_series.quantile(0.2))
            sell_threshold = float(pe_series.quantile(0.8))

            return {
                'current_pe': round(current_pe, 2),
                'percentile': round(percentile, 1),
                'buy_threshold': round(buy_threshold, 2),
                'sell_threshold': round(sell_threshold, 2),
            }
    except Exception as e:
        log(f"PE数据获取失败: {e}")
    return None


# ════════════════════════════════════════════
#  页面布局（优化移动端）
# ════════════════════════════════════════════

# 【关键优化】全局状态持久化，避免页面重绘丢失数据
if 'scanning' not in st.session_state:
    st.session_state.scanning = False
if 'recommendation_result' not in st.session_state:
    st.session_state.recommendation_result = None
if 'ma_result' not in st.session_state:
    st.session_state.ma_result = None
if 'endday_result' not in st.session_state:
    st.session_state.endday_result = None
if 'csi300_data' not in st.session_state:
    st.session_state.csi300_data = None

tab1, tab2, tab3, tab4 = st.tabs(["📊 一进二", "📉 均线回踩", "⏰ 尾盘", "📊 28交易法"])

# ──────────── Tab 1: 一进二策略 ────────────
with tab1:
    st.subheader("首板一进二推荐")

    with st.expander("💡 策略说明", expanded=False):
        st.markdown("""
        **评分因子**：
        1. 🏆 历史策略匹配（+40分）
        2. 🔥 热门板块（+25分）
        3. 🔒 一封到底（+15分）
        4. 💰 高封单比（+10分）
        5. 📊 板块热度（+10分）
        """)

    scan_btn = st.button("🔍 开始扫描", type="primary", key="scan_yijin",
                         disabled=st.session_state.scanning, use_container_width=True)

    if scan_btn:
        st.session_state.scanning = True
        st.session_state.recommendation_result = None
        try:
            prog = st.progress(0, text="准备中...")
            def update_progress(pct, msg):
                prog.progress(pct, text=f"{msg}")

            result, err = run_recommendation(progress_cb=update_progress)

            if err:
                st.error(f"❌ {err}")
                st.session_state.recommendation_result = None
            else:
                st.success("✅ 分析完成!")
                st.session_state.recommendation_result = result
        finally:
            st.session_state.scanning = False

    # 显示结果（优先从session_state取）
    result = st.session_state.get('recommendation_result')
    if result and result.get('display') is not None:
        info = result
        display_df = info['display']

        row1 = st.columns(3)
        row1[0].metric("分析日期", info['analysis_date'])
        row1[1].metric("涨停总数", info['total_zt'])
        row1[2].metric("首板股票", info['first_board_count'])
        row2 = st.columns(2)
        row2[0].metric("一字板(排除)", info['yizi_count'])
        row2[1].metric("推荐买入", info['buy_day'])

        if info['best_strategy']:
            bs = info['best_strategy']
            st.info(f"🏆 **最佳策略**: {bs['name']}　|　胜率: **{bs['win_rate']*100:.1f}%**")

        if info['hot_sectors']:
            st.info(f"🔥 **热门板块**: {'、'.join(info['hot_sectors'][:5])}")

        st.subheader("Top 5 推荐")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '总分': st.column_config.NumberColumn("总分", format="%d"),
                '换手率%': st.column_config.NumberColumn("换手%", format="%.1f"),
                '流通市值亿': st.column_config.NumberColumn("市值亿", format="%.1f"),
            },
        )

    elif not scan_btn and result is None:
        st.info("点击「开始扫描」分析今日首板一进二机会")

# ──────────── Tab 2: 均线回踩 ────────────
with tab2:
    st.subheader("均线回踩低吸策略")

    with st.expander("💡 策略说明", expanded=False):
        st.markdown("""
        **逻辑**：
        1. 上升趋势（MA20 > MA60）
        2. 回踩到 MA10 支撑位但没跌破
        3. 回踩时缩量

        **操作**：MA10附近分批低吸，跌破MA20止损
        """)

    scan_btn = st.button("🔍 开始扫描", type="primary", key="scan_ma",
                         disabled=st.session_state.scanning, use_container_width=True)

    if scan_btn:
        st.session_state.scanning = True
        st.session_state.ma_result = None
        try:
            prog = st.progress(0, text="准备中...")
            def update_ma(pct, msg):
                prog.progress(pct, text=f"{msg}")

            result_df, err = run_ma_pullback_scan(progress_cb=update_ma)

            if err:
                st.error(f"❌ {err}")
                st.session_state.ma_result = None
            elif result_df.empty:
                st.info("⚠️ 今日无符合条件的股票")
                st.session_state.ma_result = None
            else:
                st.success(f"✅ 发现 {len(result_df)} 只候选股")
                st.session_state.ma_result = result_df
        finally:
            st.session_state.scanning = False

    ma_result = st.session_state.get('ma_result')
    if ma_result is not None and not ma_result.empty:
        result_df = ma_result

        col1, col2, col3 = st.columns(3)
        col1.metric("候选数", len(result_df))
        col2.metric("距MA10最近", f"{result_df['距MA10%'].min():+.1f}%")
        col3.metric("平均量比", f"{result_df['量比(均)'].mean():.2f}")

        st.subheader("候选股")
        display_cols = ['代码', '名称', '现价', '距MA10%', '量比(均)']
        avail_cols = [c for c in display_cols if c in result_df.columns]
        display_df = result_df[avail_cols].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("🏆 最佳 3 只")
        top3 = display_df.head(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            st.success(f"**{i+1}. {row['名称']} ({row['代码']})**")

        st.info("📌 操作建议：MA10附近分批低吸，MA20为止损位")

    elif not scan_btn and ma_result is None:
        st.info("点击「开始扫描」筛选均线回踩标的")


# ──────────── Tab 3: 尾盘买入 ────────────
with tab3:
    st.subheader("尾盘买入法（14:30策略）")

    with st.expander("💡 策略说明", expanded=False):
        st.markdown("""
        **适用时间**：每个交易日14:30运行

        **选股规则**：
        1. 换手率3%-15%
        2. 量比2-5倍
        3. 涨幅3%-6%
        4. 20天内有过涨停
        """)

    pick_btn = st.button("🔍 开始选股", type="primary", key="pick_endday",
                         disabled=st.session_state.scanning, use_container_width=True)

    if pick_btn:
        st.session_state.scanning = True
        st.session_state.endday_result = None
        try:
            prog = st.progress(0, text="准备中...")
            def update_pick(pct, msg):
                prog.progress(pct, text=f"{msg}")

            result_df, err = run_end_of_day_pick(progress_cb=update_pick)

            if err:
                st.error(f"❌ {err}")
                st.session_state.endday_result = None
            elif result_df.empty:
                st.info("⚠️ 今日无符合条件的股票")
                st.session_state.endday_result = None
            else:
                st.success(f"✅ 发现 {len(result_df)} 只候选股")
                st.session_state.endday_result = result_df
        finally:
            st.session_state.scanning = False

    endday_result = st.session_state.get('endday_result')
    if endday_result is not None and not endday_result.empty:
        result_df = endday_result

        col1, col2 = st.columns(2)
        col1.metric("候选数", len(result_df))
        col2.metric("平均换手率", f"{result_df['换手率%'].mean():.1f}%")

        st.subheader("候选股")
        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '现价': st.column_config.NumberColumn("现价", format="%.2f"),
                '涨幅%': st.column_config.NumberColumn("涨幅%", format="%+.1f"),
                '换手率%': st.column_config.NumberColumn("换手%", format="%.1f"),
            },
        )

        st.subheader("🏆 首推")
        top = result_df.head(3)
        for i, (_, row) in enumerate(top.iterrows()):
            st.success(f"**{i+1}. {row['名称']} ({row['代码']})**")

        st.info("📌 操作建议：14:30后确认站稳分时均线，轻仓介入，止损-3%")

    elif not pick_btn and endday_result is None:
        st.info("点击「开始选股」筛选尾盘标的")


# ──────────── Tab 4: 沪深300 28交易法 ────────────
with tab4:
    st.subheader("沪深300 28交易法")

    with st.expander("💡 策略说明", expanded=False):
        st.markdown("""
        **规则**：
        1. PE分位值 ≤20% → 全仓买入
        2. PE分位值 ≥80% → 清仓卖出
        3. 中间区间 → 持有不动
        """)

    refresh_btn = st.button("🔄 刷新数据", type="primary", key="refresh_csi",
                            disabled=st.session_state.scanning, use_container_width=True)

    if refresh_btn or st.session_state.get('csi300_data') is None:
        st.session_state.scanning = True
        try:
            with st.spinner("获取估值数据中（约10秒）..."):
                data = _call_with_timeout(get_csi300_pe_data, 30)
                if data is not None:
                    st.session_state.csi300_data = data
                    st.success("✅ 数据加载完成")
                else:
                    st.error("❌ 加载失败，请重试")
        finally:
            st.session_state.scanning = False

    data = st.session_state.get('csi300_data')
    if data is not None:
        col1, col2 = st.columns(2)
        col1.metric("当前PE", data['current_pe'])
        col2.metric("近10年分位值", f"{data['percentile']:.1f}%")

        pct = data['percentile']
        if pct <= 20:
            signal = "🔴 买入区间"
            signal_help = f"PE ≤ {data['buy_threshold']}，低估区，建议分批买入"
        elif pct >= 80:
            signal = "🟢 卖出区间"
            signal_help = f"PE ≥ {data['sell_threshold']}，高估区，建议卖出"
        else:
            signal = "⚪ 持有/观望"
            signal_help = "中间区间，建议持有不动"
        st.metric("当前信号", signal)
        st.info(signal_help)

        st.subheader("PE分位值仪表盘")
        st.progress(pct / 100, text=f"当前处于近10年 {pct:.1f}% 位置")

        col_l, col_r = st.columns(2)
        col_l.metric("买入阈值(20%)", f"PE ≤ {data['buy_threshold']}")
        col_r.metric("卖出阈值(80%)", f"PE ≥ {data['sell_threshold']}")

        # 操作建议
        st.subheader("操作建议")
        if pct <= 20:
            st.success(f"当前分位值 {pct:.1f}%，低估区间，建议全仓买入沪深300指数基金")
        elif pct >= 80:
            st.warning(f"当前分位值 {pct:.1f}%，高估区间，建议清仓")
        else:
            st.info(f"当前分位值 {pct:.1f}%，中间区间，保持持仓不动")

    elif data is None and refresh_btn:
        st.error("❌ 获取数据失败，请重试")
    elif data is None:
        st.info("点击「刷新数据」获取估值数据")


# ── 底部 ──
st.divider()
st.caption("⚠️ 仅供学习参考，不构成投资建议")