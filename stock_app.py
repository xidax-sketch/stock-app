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
def api_retry(func, max_retries=3, delay=2):
    """带指数退避的 API 重试调用"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_msg = str(e)
            is_conn_error = 'RemoteDisconnected' in err_msg or 'Connection aborted' in err_msg or 'ConnectionError' in err_msg
            if attempt < max_retries - 1 and is_conn_error:
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
                log(f"网络错误，{sleep_time:.0f}秒后重试 ({attempt+2}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                raise


# ── 页面配置 ──
st.set_page_config(page_title="市场分析", page_icon="📈", layout="wide")
st.title("📈 市场分析")


# ════════════════════════════════════════════
#  核心分析逻辑（从 测试py 迁移）
# ════════════════════════════════════════════

def _call_with_timeout(func, timeout=15, *args, **kwargs):
    """带超时的函数调用，超时或异常返回 None（不抛异常）"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except Exception:
            return None


def log(msg):
    t = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{t}] {msg}")


def get_hot_sectors(date, top_n=10, sector_type="概念"):
    indicator = "今日"
    sector_param = f"{sector_type}资金流"
    try:
        df = _call_with_timeout(lambda: ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_param), 20)
        if df is None:
            raise ValueError("超时或返回空")
        hot_sectors = df.sort_values(by='主力净流入', ascending=False).head(top_n)['板块名称'].tolist()
        log(f"热门{sector_type}: {hot_sectors}")
        return hot_sectors
    except Exception as e:
        log(f"获取热门{sector_type}失败: {e}")
        return []


def get_valid_zt_date(target_date, max_back_days=7):
    date_obj = datetime.datetime.strptime(target_date, '%Y%m%d')
    for i in range(max_back_days + 1):
        check_date = (date_obj - datetime.timedelta(days=i)).strftime('%Y%m%d')
        df = _call_with_timeout(ak.stock_zt_pool_em, 15, date=check_date)
        if df is not None and not df.empty:
            log(f"使用回溯日期: {check_date} (原: {target_date})")
            return check_date, df
    return None, pd.DataFrame()


def analyze_historical_prob(date_start='20240101', date_end=None, progress_cb=None):
    """分析历史首板，返回最佳策略信息"""
    if date_end is None:
        date_end = datetime.datetime.now().strftime('%Y%m%d')

    all_dates = pd.date_range(start=date_start, end=date_end).strftime('%Y%m%d')
    dates = [d for d in all_dates if datetime.datetime.strptime(d, '%Y%m%d').weekday() < 5]
    total = len(dates)

    zt_df_list = []
    success_count = 0
    for idx, d in enumerate(dates):
        try:
            df = _call_with_timeout(ak.stock_zt_pool_em, 15, date=d)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(d)
                zt_df_list.append(df)
                success_count += 1
            if progress_cb and (idx % 20 == 0 or idx == total - 1):
                progress_cb(min((idx + 1) / total, 1.0), f"获取历史数据 {idx+1}/{total}")
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
    if len(first_board_pool) < 50:
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
        {
            'name': '换手15-40% + 自然换手板 + 早封 + 市值<80亿',
            'cond_fn': lambda df: (df['换手率'] > 15) & (df['换手率'] < 40) & df['one_seal'] & (df['circ_mv'] < 80) & df['early_seal'],
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
        progress_cb(0.05, f"获取 {analysis_date} 涨停数据...")
    analysis_date, zt_df = get_valid_zt_date(analysis_date)
    if zt_df.empty:
        return None, "无涨停数据"
    total_zt = len(zt_df)

    # 步骤 2：前日涨停
    prev_date = (datetime.datetime.strptime(analysis_date, '%Y%m%d') - datetime.timedelta(days=1)).strftime('%Y%m%d')
    if progress_cb:
        progress_cb(0.10, f"获取前日 {prev_date} 涨停数据...")
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
        progress_cb(0.15, "计算股票因子...")
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
        progress_cb(0.20, "分析历史首板一进二概率（较慢，请等待...）")

    def hist_progress(pct, msg):
        if progress_cb:
            progress_cb(0.20 + pct * 0.50, msg)

    best_strategy = analyze_historical_prob(progress_cb=hist_progress)

    # 步骤 6：热门板块
    if progress_cb:
        progress_cb(0.75, "获取热门板块数据...")
    hot_sectors = get_hot_sectors(analysis_date)

    # 步骤 7：评分
    if progress_cb:
        progress_cb(0.80, "综合评分中...")
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
    if cached and now - cached['time'] < 120:
        return cached['data']

    # ── 数据源 1：东方财富（30秒超时） ──
    def fetch_em():
        df = _call_with_timeout(ak.stock_zh_a_spot_em, 30)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise ValueError("东方财富返回空数据或超时")
        df.columns = [c.strip() for c in df.columns]
        if '代码' not in df.columns:
            raise ValueError(f"东方财富数据缺少「代码」列，列名={list(df.columns)}")
        df['代码'] = df['代码'].astype(str)
        return df

    # ── 数据源 2：新浪（备用，30秒超时） ──
    def fetch_sina():
        df = _call_with_timeout(ak.stock_zh_a_spot, 30)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise ValueError("新浪返回空数据或超时")
        df.columns = [c.strip() for c in df.columns]
        # 新浪返回英文列名，统一成中文
        col_map = {'symbol': '代码', 'name': '名称', 'trade': '最新价',
                   'pricechange': '涨跌额', 'changepercent': '涨跌幅',
                   'buy': '买入', 'sell': '卖出', 'settlement': '昨收',
                   'open': '今开', 'high': '最高', 'low': '最低',
                   'volume': '成交量', 'amount': '成交额', 'turnoverratio': '换手率'}
        for eng, ch in col_map.items():
            if eng in df.columns and ch not in df.columns:
                df = df.rename(columns={eng: ch})
        if '代码' not in df.columns:
            raise ValueError(f"新浪数据缺少「代码」列，列名={list(df.columns)}")
        df['代码'] = df['代码'].astype(str)
        # 数值列转数字
        for col in ['最新价', '成交额', '换手率']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # ── 按顺序尝试 ──
    sources = [
        ("东方财富", fetch_em),
        ("新浪", fetch_sina),
    ]
    last_err = ""
    for name, fn in sources:
        try:
            log(f"尝试从 {name} 获取行情...")
            df = api_retry(fn, max_retries=2, delay=2)
            log(f"{name} 获取成功，共 {len(df)} 只股票")
            st.session_state['spot_cache'] = {'data': df, 'time': now}
            return df
        except Exception as e:
            last_err = f"{name}: {e}"
            log(f"{name} 获取失败: {e}")

    log(f"所有行情源均失败: {last_err}")
    st.session_state.pop('spot_cache', None)
    return pd.DataFrame()


# ════════════════════════════════════════════
#  均线回踩策略
# ════════════════════════════════════════════

@lru_cache(maxsize=500)
def _fetch_history_cached(code):
    """缓存单只股票历史数据（60日），支持多数据源 + 超时保护"""
    end = datetime.datetime.now().strftime('%Y%m%d')
    start = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y%m%d')

    # 数据源 1：东方财富（15秒超时）
    df = _call_with_timeout(
        ak.stock_zh_a_hist, 15, symbol=code, period="daily",
        start_date=start, end_date=end, adjust="qfq"
    )
    if df is not None and not df.empty:
        try:
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            pass

    # 数据源 2：腾讯（备用，15秒超时）
    try:
        prefix = 'sh' if str(code).startswith('6') else 'sz'
        df = _call_with_timeout(
            ak.stock_zh_a_daily, 15,
            symbol=f'{prefix}{code}', start_date=start, end_date=end, adjust='qfq'
        )
        if df is not None and not df.empty:
            df = df.rename(columns={
                'date': '日期', 'open': '开盘', 'close': '收盘', 'high': '最高',
                'low': '最低', 'volume': '成交量', 'amount': '成交额',
                'outstanding_share': '流通股', 'turnover': '换手率',
            })
            return df
    except Exception:
        pass

    return None


def _check_ma_pullback(code, name):
    """检查单只股票是否满足均线回踩条件"""
    try:
        df = _fetch_history_cached(code)
        if df is None or len(df) < 30:
            return None

        close = df['收盘'].astype(float).values
        low = df['最低'].astype(float).values
        vol = df['成交量'].astype(float).values
        dates = df['日期'].values

        # 计算均线
        ma5 = pd.Series(close).rolling(5).mean().values
        ma10 = pd.Series(close).rolling(10).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60, min_periods=20).mean().values

        last = -1
        c, v = close[last], vol[last]

        # ── 条件判断 ──
        # 1. 多头排列：收盘价 > MA20 > MA60
        if not (c > ma20[last] and ma20[last] > ma60[last]):
            return None

        # 2. 回踩 MA10：最低价 <= MA10 <= 收盘价（或略微跌破但收回来）
        if not (low[last] <= ma10[last] <= c * 1.02):
            return None

        # 3. 缩量：今日量 < 前5日均量
        vol_ma5 = pd.Series(vol[-10:-1]).mean()
        if vol_ma5 == 0:
            return None
        vol_ratio = v / vol_ma5
        if vol_ratio > 1.3:
            return None

        # 4. 近期有涨停或大阳线（有主力活跃痕迹）
        recent_high = max(close[-10:]) / close[-11] if len(close) > 10 else 1
        if recent_high < 1.05 and vol_ratio < 0.5:
            return None  # 太死寂的票不要

        # 5. MACD 未死叉（快速计算）
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9).mean().values
        if dif[last] < dea[last] and dif[last] < dif[last-1]:
            return None  # MACD 死叉中

        # 计算距 MA10 的距离（负值=跌破，正值=在之上）
        dist_to_ma10 = round((c - ma10[last]) / ma10[last] * 100, 2)
        # 计算距 MA20 的距离
        dist_to_ma20 = round((c - ma20[last]) / ma20[last] * 100, 2)

        info = {
            '代码': code,
            '名称': name,
            '现价': round(c, 2),
            'MA10': round(ma10[last], 2),
            'MA20': round(ma20[last], 2),
            'MA60': round(ma60[last], 2) if not pd.isna(ma60[last]) else 0,
            '距MA10%': dist_to_ma10,
            '距MA20%': dist_to_ma20,
            '量比(均)': round(vol_ratio, 2),
            '涨幅%': round((close[last] / close[-2] - 1) * 100, 2) if len(close) > 1 else 0,
        }
        return info
    except Exception:
        return None


def _build_ma_candidates_from_zt_pool(max_days=20):
    """使用涨停板数据作为候选股票源（行情接口不可用时的备选方案）"""
    log(f"尝试从涨停板数据获取候选股票（回溯{max_days}天）...")
    all_codes = []
    all_names = []

    for day_offset in range(max_days):
        date = (datetime.datetime.now() - datetime.timedelta(days=day_offset)).strftime('%Y%m%d')
        df = _call_with_timeout(ak.stock_zt_pool_em, 15, date=date)
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

    log(f"涨停板数据去重后共 {len(unique_stocks)} 只候选股")
    return unique_stocks


def _build_all_stock_candidates(max_count=800):
    """使用股票基本信息接口获取全市场股票（最终备选方案）"""
    try:
        log("尝试获取全市场股票列表...")
        df = _call_with_timeout(ak.stock_info_a_code_name, 20)
        if df is None or df.empty:
            return []
        # 兼容不同列名
        code_col = 'code' if 'code' in df.columns else '代码'
        name_col = 'name' if 'name' in df.columns else '名称'
        codes = df[code_col].astype(str).tolist()
        names = df[name_col].tolist()
        log(f"获取到全市场 {len(codes)} 只股票，取前 {max_count} 只")
        return [(str(c).strip().zfill(6), n) for c, n in zip(codes, names)][:max_count]
    except Exception as e:
        log(f"获取全市场股票列表失败: {e}")
        return []


def get_zt_codes_last_n_days(n=20):
    """获取最近 n 个交易日有过涨停的股票代码集合（带超时保护）"""
    all_codes = set()
    for day_offset in range(n):
        date = (datetime.datetime.now() - datetime.timedelta(days=day_offset)).strftime('%Y%m%d')
        df = _call_with_timeout(ak.stock_zt_pool_em, 15, date=date)
        if df is not None and not df.empty and '代码' in df.columns:
            codes = df['代码'].astype(str).str.strip().str.zfill(6).tolist()
            all_codes.update(codes)
    log(f"获取到近{n}天涨停股票 {len(all_codes)} 只")
    return all_codes


def run_ma_pullback_scan(progress_cb=None):
    """均线回踩扫描主函数"""
    if progress_cb:
        progress_cb(0, "获取全市场行情...")

    spot = get_spot_data()

    stock_list = []

    if not spot.empty:
        # ── 行情数据可用：正常筛选 ──
        if progress_cb:
            progress_cb(0.05, f"基础筛选（共 {len(spot)} 只）...")

        spot = spot.copy()
        spot.columns = [c.strip() for c in spot.columns]

        required_cols = ['名称', '最新价', '成交额']
        missing = [c for c in required_cols if c not in spot.columns]
        if missing:
            err_msg = f"行情数据缺少列: {missing}，实际列名: {list(spot.columns)}"
            log(err_msg)
            # 行情列缺失，降级到涨停板源
            stock_list = _build_ma_candidates_from_zt_pool()
            if not stock_list:
                stock_list = _build_all_stock_candidates()
            if not stock_list:
                return pd.DataFrame(), f"行情数据异常且所有备选源均无数据: {err_msg}"
        else:
            # 筛选条件
            mask = pd.Series(True, index=spot.index)
            mask &= ~spot['名称'].str.contains('ST|退市|^N', na=False)
            spot['最新价'] = pd.to_numeric(spot['最新价'], errors='coerce')
            mask &= spot['最新价'] > 3
            spot['成交额'] = pd.to_numeric(spot['成交额'], errors='coerce')
            mask &= spot['成交额'] > 3e7
            if '换手率' in spot.columns:
                spot['换手率'] = pd.to_numeric(spot['换手率'], errors='coerce')
                mask &= spot['换手率'] > 1
            else:
                log("换手率列不可用，跳过该筛选条件")

            candidates = spot[mask].sort_values('成交额', ascending=False).head(200)
            if candidates.empty:
                return pd.DataFrame(), "未找到符合条件的股票"

            codes = candidates['代码'].tolist()
            name_col = '名称' if '名称' in candidates.columns else candidates.columns[1]
            stock_list = [(str(code).strip().zfill(6), row[name_col])
                          for code, (_, row) in zip(codes, candidates.iterrows())]
    else:
        # ── 行情不可用：降级到涨停板源 ──
        if progress_cb:
            progress_cb(0.05, "行情接口不可用，改用涨停板数据源...")
        stock_list = _build_ma_candidates_from_zt_pool()
        if not stock_list:
            if progress_cb:
                progress_cb(0.05, "涨停板数据不足，改用全市场股票列表...")
            stock_list = _build_all_stock_candidates()
        if not stock_list:
            return pd.DataFrame(), "所有数据源均不可用（行情接口 + 涨停板 + 股票列表均失败）"

    total = len(stock_list)
    if progress_cb:
        progress_cb(0.1, f"扫描 {total} 只候选股的历史走势（并行获取，请等待）...")

    # ── 并行获取历史数据并检查 ──
    results = []
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_check_ma_pullback, code, name): (code, name)
                   for code, name in stock_list}

        for f in concurrent.futures.as_completed(futures):
            done_count += 1
            if progress_cb and done_count % 20 == 0:
                progress_cb(0.1 + 0.85 * done_count / total,
                            f"分析进度 {done_count}/{total}")

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

    # ── 排序 ──
    df_result = pd.DataFrame(results)
    df_result['距离得分'] = df_result['距MA10%'].abs().apply(lambda x: max(0, 10 - x * 2))
    df_result['量能得分'] = df_result['量比(均)'].apply(lambda x: max(0, (1.3 - x) * 10))
    df_result['总分'] = df_result['距离得分'] + df_result['量能得分']
    df_result = df_result.sort_values('总分', ascending=False)

    if progress_cb:
        progress_cb(1.0, f"完成！找到 {len(df_result)} 只")

    return df_result, None


def _fetch_sina_quote(code):
    """获取新浪个股实时行情（仅需一次小HTTP请求）"""
    try:
        prefix = 'sh' if code.startswith('6') else 'sz'
        url = f'http://hq.sinajs.cn/list={prefix}{code}'
        headers = {'Referer': 'http://finance.sina.com.cn'}
        r = requests.get(url, headers=headers, timeout=5)
        r.encoding = 'gbk'
        text = r.text
        if 'hq_str_' not in text:
            return None
        data = text.split('"')[1].split(',')
        if len(data) < 32:
            return None
        name = data[0]
        yclose = float(data[2])
        price = float(data[3])
        volume = float(data[8])   # 成交量（股）
        amount = float(data[9])   # 成交额（元）
        change_pct = round((price - yclose) / yclose * 100, 2)
        return {
            'name': name, 'price': price, 'change_pct': change_pct,
            'volume': volume, 'amount': amount,
        }
    except Exception:
        return None


def _fetch_stock_info(code):
    """获取个股流通股本和流通市值（含重试）"""
    # 尝试1：带超时的东方财富个股信息（10秒超时）
    df = _call_with_timeout(ak.stock_individual_info_em, 10, symbol=code)
    if df is not None and not df.empty:
        try:
            info = dict(zip(df['item'], df['value']))
            return {
                'circ_shares': float(info.get('流通股', 0)),
                'circ_mv': float(info.get('流通市值', 0)),
            }
        except Exception:
            pass

    # 尝试2：使用涨停板数据的流通市值（更稳定，15秒超时）
    today = datetime.datetime.now().strftime('%Y%m%d')
    df = _call_with_timeout(ak.stock_zt_pool_em, 15, date=today)
    if df is not None and not df.empty:
        try:
            zt_codes = df['代码'].astype(str).str.strip().str.zfill(6).tolist()
            if code in zt_codes and '流通市值' in df.columns:
                idx = zt_codes.index(code)
                circ_mv = float(df['流通市值'].iloc[idx])
                price_in_zt = float(df.iloc[idx, 4])
                circ_shares = circ_mv / price_in_zt if price_in_zt > 0 else 0
                return {'circ_shares': circ_shares, 'circ_mv': circ_mv}
        except Exception:
            pass
    return None


def _check_end_of_day_stock(code, name, zt_codes):
    """检查单只股票是否满足尾盘买入条件（逐只API调用，避免大请求）"""
    try:
        # 近期涨停检查
        if code not in zt_codes:
            return None

        # 基本信息（流通股本、流通市值）
        info = _fetch_stock_info(code)
        if info is None or info['circ_shares'] <= 0:
            return None

        circ_shares = info['circ_shares']
        circ_mv = info['circ_mv']

        # 流通市值 50-300亿
        if circ_mv < 5e9 or circ_mv > 3e10:
            return None

        # 实时行情（新浪）
        quote = _fetch_sina_quote(code)
        if quote is None:
            return None

        price = quote['price']
        change_pct = quote['change_pct']
        volume = quote['volume']
        amount = quote['amount']

        # 涨跌幅 3-6%
        if change_pct < 3 or change_pct > 6:
            return None

        # 换手率 3%-15%（成交量 / 流通股本 * 100）
        turnover_pct = volume / circ_shares * 100
        if turnover_pct < 3 or turnover_pct > 15:
            return None

        # 分时均线检查：最新价 > 均价（成交额/成交量）
        avg_price = amount / volume if volume > 0 else 0
        if price <= avg_price:
            return None

        return {
            '代码': code,
            '名称': quote['name'] if quote['name'] else name,
            '现价': round(price, 2),
            '涨幅%': change_pct,
            '换手率%': round(turnover_pct, 2),
            '量比': 0.0,
            '流通市值亿': round(circ_mv / 1e8, 1),
            '60优先': code.startswith('60'),
        }
    except Exception:
        return None


def run_end_of_day_pick(progress_cb=None):
    """尾盘买入法选股（14:30 策略）"""
    if progress_cb:
        progress_cb(0, "获取实时行情...")

    spot = get_spot_data()
    fast_path = False

    if not spot.empty:
        spot = spot.copy()
        spot.columns = [c.strip() for c in spot.columns]

        required = ['名称', '最新价', '涨跌幅', '换手率', '量比', '成交额', '成交量', '流通市值']
        missing = [c for c in required if c not in spot.columns]
        if not missing:
            fast_path = True
        else:
            log(f"行情列不全({missing})，降级到逐只查询")

    if fast_path:
        # ── 快速路径：全市场行情可用且列完整 ──
        if progress_cb:
            progress_cb(0.1, "数据清洗与筛选...")

        for col in required[1:]:
            spot[col] = pd.to_numeric(spot[col], errors='coerce')

        mask = pd.Series(True, index=spot.index)
        mask &= ~spot['名称'].str.contains('ST|退市', na=False)
        mask &= (spot['换手率'] >= 3) & (spot['换手率'] <= 15)
        mask &= (spot['量比'] >= 2) & (spot['量比'] <= 5)
        mask &= (spot['涨跌幅'] >= 3) & (spot['涨跌幅'] <= 6)
        mask &= (spot['流通市值'] >= 5e9) & (spot['流通市值'] <= 3e10)

        spot['均价'] = spot.apply(lambda r: r['成交额'] / r['成交量'] if r['成交量'] > 0 else 0, axis=1)
        mask &= spot['最新价'] > spot['均价']

        candidates = spot[mask].copy()
        if candidates.empty:
            return pd.DataFrame(), "基础筛选无结果（当前无股票同时满足换手率/量比/涨幅条件）"

        if progress_cb:
            progress_cb(0.4, f"基础筛选剩 {len(candidates)} 只，检查近期涨停记录...")

        zt_codes = get_zt_codes_last_n_days(20)
        if not zt_codes:
            return pd.DataFrame(), "获取涨停数据失败"

        candidates['近期涨停'] = candidates['代码'].astype(str).str.strip().str.zfill(6).isin(zt_codes)
        final = candidates[candidates['近期涨停']].copy()
        if final.empty:
            return pd.DataFrame(), "符合基础条件的股票中无近期涨停记录"

        if progress_cb:
            progress_cb(0.7, "排序生成结果...")

        final['是60优先'] = final['代码'].astype(str).str.startswith('60')
        final = final.sort_values(['是60优先', '换手率'], ascending=[False, False])

        result = final[['代码', '名称', '最新价', '涨跌幅', '换手率', '量比', '流通市值', '是60优先']].copy()
        result['最新价'] = result['最新价'].round(2)
        result['涨跌幅'] = result['涨跌幅'].round(2)
        result['换手率'] = result['换手率'].round(2)
        result['量比'] = result['量比'].round(2)
        result['流通市值'] = (result['流通市值'] / 1e8).round(1)
        result.columns = ['代码', '名称', '现价', '涨幅%', '换手率%', '量比', '流通市值亿', '60优先']

        if progress_cb:
            progress_cb(1.0, f"完成！找到 {len(result)} 只")
        return result, None

    # ═══════════════════════════════════════════════
    #  慢速路径：逐只个股查询
    # ═══════════════════════════════════════════════
    if progress_cb:
        progress_cb(0.05, "改用逐只个股查询（约1-2分钟）...")

    zt_codes = get_zt_codes_last_n_days(20)
    if not zt_codes:
        return pd.DataFrame(), "获取涨停数据失败"

    if progress_cb:
        progress_cb(0.1, f"近20天涨停股票 {len(zt_codes)} 只，准备查询...")

    all_stocks = _build_all_stock_candidates(max_count=1500)
    candidates = [(c, n) for c, n in all_stocks if c in zt_codes]
    if not candidates:
        candidates = [(c, '') for c in list(zt_codes)[:500]]

    if progress_cb:
        progress_cb(0.15, f"检查 {len(candidates)} 只候选股（需近期涨停+换手率+涨幅+市值+分时均线）...")

    results = []
    total = len(candidates)
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_check_end_of_day_stock, code, name, zt_codes): (code, name)
                   for code, name in candidates}

        for f in concurrent.futures.as_completed(futures):
            done += 1
            if progress_cb and done % 20 == 0:
                progress_cb(0.15 + 0.75 * done / total,
                            f"进度 {done}/{total}，已找到 {len(results)} 只")
            try:
                r = f.result()
                if r is not None:
                    results.append(r)
                    if progress_cb and len(results) % 5 == 1:
                        progress_cb(0.15 + 0.75 * done / total,
                                    f"找到 {len(results)} 只（进度 {done}/{total}）")
            except Exception:
                pass

    if progress_cb:
        progress_cb(0.9, f"排序中（共找到 {len(results)} 只）...")

    if not results:
        log(f"尾盘买入无结果: 涨停股{len(zt_codes)}只, 检查{total}只候选, 均未通过筛选")
        return pd.DataFrame(), f"未找到满足条件的股票（涨停股{len(zt_codes)}只，检查了{total}只，均未通过换手率/涨幅/市值/分时均线条件）"

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(['60优先', '换手率%'], ascending=[False, False])

    if progress_cb:
        progress_cb(1.0, f"完成！找到 {len(df_result)} 只（慢速查询）")

    return df_result, None


# ════════════════════════════════════════════
#  沪深300 28交易法
# ════════════════════════════════════════════

def get_csi300_pe_data():
    """获取沪深300指数近15年PE分位值数据

    数据来源：
    - 乐股(legulegu.com)：20年PE历史序列（用于分位值计算和走势图）
    - 中证指数有限公司(csindex.com.cn)：官方每日PE值（用于当前PE）

    两个来源的PE值存在约8%的差异（中证指数PE高于乐股），
    原因是财报数据更新节奏和计算方法不同。
    分位值基于乐股数据计算以保持历史一致性。
    """
    now = datetime.datetime.now()
    result = None

    # ── 1. 主数据源：乐股 PE 历史（20年） ──
    try:
        log("获取沪深300 PE数据(乐股)...")
        df = _call_with_timeout(lambda: ak.stock_index_pe_lg(symbol='沪深300'), 30)
        if df is not None and not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            log(f"PE数据列: {list(df.columns)}，共 {len(df)} 行")

            date_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), df.columns[0])
            idx_col = next((c for c in df.columns if '指数' in c or 'index' in c.lower()), None)
            pe_col = '滚动市盈率' if '滚动市盈率' in df.columns else None
            if pe_col is None:
                pe_col = next((c for c in df.columns if '滚动' in c and '市盈' in c), None)
            if pe_col is None:
                pe_col = next((c for c in df.columns if '市盈' in c), None)
            if pe_col is None:
                raise ValueError(f"未找到PE列: {list(df.columns)}")

            df['date'] = pd.to_datetime(df[date_col])
            df = df.sort_values('date').reset_index(drop=True)

            cutoff = now - datetime.timedelta(days=365*15)
            df_10y = df[df['date'] >= cutoff].copy()

            if len(df_10y) < 60:
                raise ValueError(f"数据仅 {len(df_10y)} 行")

            pe_series = df_10y[pe_col].astype(float)
            current_pe = pe_series.iloc[-1]
            percentile = float(pe_series.rank(pct=True).iloc[-1] * 100)

            buy_threshold = float(pe_series.quantile(0.2))
            sell_threshold = float(pe_series.quantile(0.8))

            df_full = df_10y.copy()
            df_full['zone'] = '持有'
            df_full.loc[pe_series <= buy_threshold, 'zone'] = '买入区'
            df_full.loc[pe_series >= sell_threshold, 'zone'] = '卖出区'
            df_full['prev_pe'] = pe_series.shift(1)
            df_full['buy_signal'] = (pe_series <= buy_threshold) & (df_full['prev_pe'] > buy_threshold)
            df_full['sell_signal'] = (pe_series >= sell_threshold) & (df_full['prev_pe'] < sell_threshold)

            price_col = idx_col or '指数'
            log(f"沪深300 28法: PE={current_pe}, 分位值={percentile:.1f}%, "
                f"买入<={buy_threshold:.1f}, 卖出>={sell_threshold:.1f}")

            result = {
                'df': df_full, 'current_pe': round(current_pe, 2),
                'percentile': round(percentile, 1),
                'pe_col': pe_col, 'price_col': price_col,
                'buy_threshold': round(buy_threshold, 2),
                'sell_threshold': round(sell_threshold, 2),
                'source': '乐股(legulegu.com)',
                'pe_series_full': pe_series,
            }
    except Exception as e:
        log(f"PE数据获取失败: {e}")

    # ── 2. 中证指数官方最新PE（补充参考） ──
    csi_pe = None
    csi_source = None
    try:
        log("获取中证指数官方PE...")
        df_csi = _call_with_timeout(lambda: ak.stock_zh_index_value_csindex(symbol='000300'), 15)
        if df_csi is not None and not df_csi.empty:
            df_csi.columns = [str(c).strip() for c in df_csi.columns]
            pe_col_csi = next((c for c in df_csi.columns if '市盈' in c), None)
            date_col_csi = next((c for c in df_csi.columns if '日期' in c or 'date' in c.lower()), df_csi.columns[0])
            if pe_col_csi and not df_csi.empty:
                csi_date = df_csi[date_col_csi].iloc[0]
                csi_pe = round(float(df_csi[pe_col_csi].iloc[0]), 2)
                csi_source = f"中证指数({csi_date})"
                log(f"中证指数官方PE={csi_pe} ({csi_date})")
    except Exception as e:
        log(f"中证指数PE获取失败: {e}")

    # 如果有主数据且有官方PE，合并信息
    if result is not None:
        if csi_pe:
            result['csi_pe'] = csi_pe
            result['csi_source'] = csi_source
        else:
            result['csi_pe'] = None
            result['csi_source'] = None
        return result

    # ── 3. 备用：仅用中证指数20条数据 ──
    if csi_pe:
        try:
            log("尝试获取价格数据绘制走势图...")
            df_price = _call_with_timeout(lambda: ak.stock_zh_index_daily(symbol='sh000300'), 15)
            if df_price is not None and not df_price.empty:
                df_price.columns = [str(c).strip() for c in df_price.columns]
                pr_date = next((c for c in df_price.columns if '日期' in c or 'date' in c.lower()), df_price.columns[0])
                price_col = next((c for c in df_price.columns if '收盘' in c or 'close' in c.lower()), df_price.columns[4])
                df_price['date'] = pd.to_datetime(df_price[pr_date])

                pe_col_csi = next((c for c in df_csi.columns if '市盈' in c), None)
                df_csi['date'] = pd.to_datetime(df_csi[date_col_csi])
                merged = pd.merge(df_csi[['date', pe_col_csi]], df_price[['date', price_col]], on='date', how='inner')
                merged = merged.sort_values('date').reset_index(drop=True)

                if len(merged) >= 5:
                    pe_csi = merged[pe_col_csi].astype(float)
                    return {
                        'df': merged, 'current_pe': round(pe_csi.iloc[-1], 2),
                        'percentile': round(float(pe_csi.rank(pct=True).iloc[-1] * 100), 1),
                        'pe_col': pe_col_csi, 'price_col': price_col,
                        'buy_threshold': round(pe_csi.quantile(0.2), 2),
                        'sell_threshold': round(pe_csi.quantile(0.8), 2),
                        'source': '中证指数(近期)',
                        'csi_pe': round(pe_csi.iloc[-1], 2),
                        'csi_source': '中证指数',
                    }
        except Exception as e:
            log(f"备用走势图失败: {e}")

    return None


# ════════════════════════════════════════════
#  页面布局
# ════════════════════════════════════════════

# ── 全局扫描状态（防止重复点击） ──
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

tab1, tab2, tab3, tab4 = st.tabs(["📊 一进二策略", "📉 均线回踩", "⏰ 尾盘买入", "📊 沪深300 28法"])

# ──────────── Tab 1: 一进二策略 ────────────

with tab1:
    st.subheader("今日首板一进二推荐")

    with st.expander("💡 策略说明"):
        st.markdown("""
        **逻辑**：从当日涨停板中筛选出**首板股票**（非一字板），综合评分选出最可能次日连板的标的。

        **评分因子**：
        1. 🏆 **历史策略匹配**（+40分） — 最优历史一进二策略命中
        2. 🔥 **热门板块**（+25分） — 属于当日资金流入热门板块
        3. 🔒 **一封到底**（+15分） — 封板后未炸板，封单坚决
        4. 💰 **高封单比**（+10分） — 封单资金 / 流通市值 排名前30%
        5. 📊 **板块热度**（+10分） — 所属行业板块涨停家数多

        **历史分析**：回测历史首板数据，自动选择胜率最高的策略组合
        """)

    scan_btn = st.button("🔍 开始扫描", type="primary", key="scan_yijin",
                         disabled=st.session_state.scanning, width='stretch')

    if scan_btn:
        st.session_state.scanning = True
        st.cache_data.clear()
        try:
            with st.status("分析中...", expanded=True) as status:
                prog = st.progress(0, text="0%")
                def update_progress(pct, msg):
                    status.update(label=f"**{msg}**", state="running")
                    prog.progress(pct, text=f"{int(pct*100)}%")

                result, err = run_recommendation(progress_cb=update_progress)

                if err:
                    status.update(label=f"❌ {err}", state="error")
                    st.warning(err)
                    st.session_state.recommendation_result = None
                else:
                    status.update(label="✅ 分析完成!", state="complete")
                    st.session_state.recommendation_result = result
        finally:
            st.session_state.scanning = False

    # 显示结果
    result = st.session_state.get('recommendation_result')
    if result and result.get('display') is not None:
        info = result
        display_df = info['display']

        meta_cols = st.columns(5)
        meta_cols[0].metric("分析日期", info['analysis_date'])
        meta_cols[1].metric("涨停总数", info['total_zt'])
        meta_cols[2].metric("首板股票", info['first_board_count'])
        meta_cols[3].metric("一字板(排除)", info['yizi_count'])
        meta_cols[4].metric("推荐买入", info['buy_day'])

        if info['best_strategy']:
            bs = info['best_strategy']
            st.info(f"🏆 **最佳策略**: {bs['name']}　|　"
                    f"历史一进二胜率: **{bs['win_rate']*100:.1f}%**　|　"
                    f"样本量: {bs['samples']} 次")

        if info['hot_sectors']:
            st.info(f"🔥 **热门板块**: {'、'.join(info['hot_sectors'][:8])}")

        st.subheader("Top 5 推荐")
        cols = [c for c in display_df.columns if c not in ['总分', '标记']]
        cols += ['总分', '标记']
        display_df = display_df[cols]

        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                '总分': st.column_config.NumberColumn("总分", format="%d"),
                '换手率%': st.column_config.NumberColumn("换手率%", format="%.1f"),
                '流通市值亿': st.column_config.NumberColumn("流通市值亿", format="%.1f"),
                '封单比%': st.column_config.NumberColumn("封单比%", format="%.2f"),
            },
        )

        st.subheader("推荐股票评分对比")
        chart_data = display_df.copy()
        chart_data['股票'] = chart_data['名称'] + '(' + chart_data['代码'] + ')'
        st.bar_chart(chart_data, x='股票', y='总分')

        with st.expander(f"查看全部 {len(info['candidates'])} 只候选首板评分"):
            all_display = info['candidates'][
                ['代码', '名称', '换手率', 'circ_mv', '首次封板时间', '所属行业', 'score', 'one_seal', 'sector_heat']
            ].copy()
            all_display.columns = ['代码', '名称', '换手率%', '流通市值亿', '封板时间', '所属行业', '总分', '一封到底', '板块热度']
            all_display = all_display.sort_values('总分', ascending=False)
            st.dataframe(all_display, width='stretch', hide_index=True)

    elif not scan_btn:
        st.info("点击「开始扫描」按钮，系统将分析今日首板一进二机会")

# ──────────── Tab 2: 均线回踩 ────────────

with tab2:
    st.subheader("📉 均线回踩低吸策略")

    with st.expander("💡 策略说明"):
        st.markdown("""
        **适用人群**：上班族，没时间盯盘

        **逻辑**：
        1. 股价在 **上升趋势** 中（MA20 > MA60，多头排列）
        2. 近期 **回踩到 MA10 支撑位** 但没跌破
        3. 回踩时 **缩量**（不是主力出货）
        4. MACD 未死叉（趋势未破坏）

        **操作建议**：
        - 在 MA10 附近分批低吸
        - 跌破 MA20 止损
        - 反弹到前高或放量滞涨时止盈
        """)

    scan_btn = st.button("🔍 开始扫描", type="primary", key="scan_ma",
                         disabled=st.session_state.scanning, width='stretch')

    if scan_btn:
        st.session_state.scanning = True
        try:
            with st.status("扫描均线回踩机会...", expanded=True) as status:
                prog = st.progress(0, text="0%")
                def update_ma(pct, msg):
                    status.update(label=f"**{msg}**", state="running")
                    prog.progress(pct, text=f"{int(pct*100)}%")

                result_df, err = run_ma_pullback_scan(progress_cb=update_ma)

                if err:
                    status.update(label=f"❌ {err}", state="error")
                    st.error(f"**错误详情**：{err}")
                    with st.expander("🔧 排查建议"):
                        st.markdown("""
                        1. **网络问题**：东方财富数据接口可能暂时不稳定，等待 1-2 分钟后重试
                        2. **接口变更**：如果持续报错，可能是 akshare 版本过旧，终端运行以下命令更新：
                           ```
                           pip install --upgrade akshare
                           ```
                        3. 详情请查看终端中的日志输出
                        """)
                    if st.button("🔄 重试", type="primary", key="retry_ma"):
                        st.cache_data.clear()
                        st.session_state.pop('spot_cache', None)
                        st.rerun()
                    st.session_state.ma_result = None
                elif result_df.empty:
                    status.update(label="⚠️ 今日未找到符合条件的股票", state="error")
                    st.info("今日没有符合均线回踩条件的股票，换个交易日再试")
                    st.session_state.ma_result = None
                else:
                    status.update(label=f"✅ 发现 {len(result_df)} 只均线回踩候选股", state="complete")
                    st.session_state.ma_result = result_df
        finally:
            st.session_state.scanning = False

    # 显示结果（优先使用 session_state 持久化）
    ma_result = st.session_state.get('ma_result')
    if ma_result is not None and not ma_result.empty:
        result_df = ma_result

        # 指标展示
        col1, col2, col3 = st.columns(3)
        col1.metric("候选股数", len(result_df))
        col2.metric("距MA10最近", f"{result_df['距MA10%'].min():+.2f}%")
        col3.metric("平均量比", f"{result_df['量比(均)'].mean():.2f}")

        st.subheader("均线回踩候选股")

        display_cols = ['代码', '名称', '现价', 'MA10', 'MA20',
                        '距MA10%', '距MA20%', '量比(均)', '涨幅%', '总分']

        avail_cols = [c for c in display_cols if c in result_df.columns]
        display_df = result_df[avail_cols].copy()

        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                '现价': st.column_config.NumberColumn("现价", format="%.2f"),
                'MA10': st.column_config.NumberColumn("MA10", format="%.2f"),
                'MA20': st.column_config.NumberColumn("MA20", format="%.2f"),
                '距MA10%': st.column_config.NumberColumn("距MA10%", format="%+.2f"),
                '距MA20%': st.column_config.NumberColumn("距MA20%", format="%+.2f"),
                '量比(均)': st.column_config.NumberColumn("量比(均值)", format="%.2f"),
                '涨幅%': st.column_config.NumberColumn("当日涨幅%", format="%+.2f"),
                '总分': st.column_config.NumberColumn("综合评分", format="%.1f"),
            },
        )

        # 最佳候选推荐
        st.subheader("🏆 最佳 3 只")
        top3 = display_df.head(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            tags = []
            if row.get('距MA10%', 0) <= 0:
                tags.append("已触及MA10支撑")
            elif row.get('距MA10%', 0) < 2:
                tags.append(f"接近MA10（{row['距MA10%']:+.1f}%）")
            if row.get('量比(均)', 2) < 1:
                tags.append("缩量回踩")
            tags.append(f"评分{row['总分']:.0f}")
            st.success(f"**{i+1}. {row['名称']} ({row['代码']})** — {' | '.join(tags)}")

        st.info("📌 **操作建议**：以上股票在 MA10 附近可分批低吸，设 MA20 为止损位")

    elif not scan_btn:
        st.info("点击「开始扫描」按钮，系统将从全市场筛选均线回踩到 MA10 支撑位的股票")


# ──────────── Tab 3: 尾盘买入 ────────────

with tab3:
    st.subheader("⏰ 尾盘买入法（14:30 策略）")

    with st.expander("💡 策略说明"):
        st.markdown("""
        **适用时间**：每个交易日 **14:30** 运行（尾盘确认信号）

        **选股规则**：
        1. 🔄 **换手率 3%-15%** — 交投活跃，有主力参与
        2. 📈 **分时均线在黄线以上** — 价格强于全天均价，非脉冲拉升
        3. 📊 **量比 2-5倍** — 温和放量，非异常巨量
        4. 💹 **涨幅 3%-6%** — 有拉升但未封板，留明日空间
        5. 💰 **流通市值 50亿-300亿** — 中盘股，弹性好
        6. 🏆 **20天内有过涨停** — 有主力活跃痕迹，股性好

        **60开头（上海主板）优先** — 上海主板股票加分

        **操作建议**：
        - 14:30 后选股，尾盘确认不回落可轻仓介入
        - 次日开盘不及预期（低开3%+）及时止损
        - 次日冲高不封板可止盈
        """)

    pick_btn = st.button("🔍 开始选股", type="primary", key="pick_endday",
                         disabled=st.session_state.scanning, width='stretch')

    if pick_btn:
        st.session_state.scanning = True
        try:
            with st.status("尾盘买入法选股中...", expanded=True) as status:
                prog = st.progress(0, text="0%")
                def update_pick(pct, msg):
                    status.update(label=f"**{msg}**", state="running")
                    prog.progress(pct, text=f"{int(pct*100)}%")

                result_df, err = run_end_of_day_pick(progress_cb=update_pick)

                if err:
                    status.update(label=f"❌ {err}", state="error")
                    st.error(f"**{err}**")
                    if st.button("🔄 重试", type="primary", key="retry_endday"):
                        st.cache_data.clear()
                        st.session_state.pop('spot_cache', None)
                        st.rerun()
                    st.session_state.endday_result = None
                elif result_df.empty:
                    status.update(label="⚠️ 今日无符合条件的股票", state="error")
                    st.info("没有同时满足换手率3%-15%、量比2-5、涨幅3-6%、近期涨停等条件的股票，换个交易日再试")
                    st.session_state.endday_result = None
                else:
                    status.update(label=f"✅ 发现 {len(result_df)} 只尾盘候选股", state="complete")
                    st.session_state.endday_result = result_df
        finally:
            st.session_state.scanning = False

    # 显示结果
    endday_result = st.session_state.get('endday_result')
    if endday_result is not None and not endday_result.empty:
        result_df = endday_result

        # 指标
        col1, col2, col3 = st.columns(3)
        col1.metric("候选股数", len(result_df))
        col2.metric("60开头占比", f"{result_df['60优先'].sum()}/{len(result_df)}")
        avg_turnover = result_df['换手率%'].mean()
        col3.metric("平均换手率", f"{avg_turnover:.1f}%")

        st.subheader("尾盘买入候选股")

        st.dataframe(
            result_df,
            width='stretch',
            hide_index=True,
            column_config={
                '现价': st.column_config.NumberColumn("现价", format="%.2f"),
                '涨幅%': st.column_config.NumberColumn("涨幅%", format="%+.2f"),
                '换手率%': st.column_config.NumberColumn("换手率%", format="%.1f"),
                '量比': st.column_config.NumberColumn("量比", format="%.2f"),
                '流通市值亿': st.column_config.NumberColumn("流通市值亿", format="%.1f"),
                '60优先': st.column_config.CheckboxColumn("60优先"),
            },
        )

        # Top 推荐
        st.subheader("🏆 首推")
        top = result_df.head(3)
        for i, (_, row) in enumerate(top.iterrows()):
            tags = [f"换手{row['换手率%']}%", f"量比{row['量比']}", f"涨幅{row['涨幅%']}%"]
            if row['60优先']:
                tags.append("沪市主板")
            st.success(f"**{i+1}. {row['名称']} ({row['代码']})** — {' | '.join(tags)}")

        st.info("📌 **操作建议**：14:30 后确认价格站稳分时均线，轻仓介入，止损设-3%")

    elif not pick_btn:
        st.info("点击「开始选股」按钮，系统将按尾盘买入法筛选符合条件的股票")


# ──────────── Tab 4: 沪深300 28交易法 ────────────

with tab4:
    st.subheader("📊 沪深300 28交易法")

    with st.expander("💡 策略说明"):
        st.markdown("""
        **核心指标**：沪深300指数近15年PE分位值

        PE分位值代表当前市盈率在近15年中所处的位置（0%~100%）。
        分位值 **≤20%** → 低估区，**≥80%** → 高估区。

        **操作规则**：
        1. 📉 **买入信号**：PE分位值 ≤ 20% → **全仓买入**沪深300指数基金
        2. 📈 **卖出信号**：PE分位值 ≥ 80% → **清仓卖出**，等待下一次买入

        **历史成绩**：过去15年仅触发3次买点、3次卖点，平均持仓27个月。
        极低交易频率，适合没时间盯盘的普通投资者。
        """)

    # 刷新按钮
    refresh_col1, refresh_col2 = st.columns([1, 5])
    with refresh_col1:
        refresh_btn = st.button("🔄 刷新数据", type="primary", key="refresh_csi",
                                disabled=st.session_state.scanning)

    with refresh_col2:
        if 'csi300_loaded' in st.session_state and st.session_state.csi300_loaded:
            st.caption(f"数据更新于: {st.session_state.get('csi300_time', '—')}")

    # 加载/刷新数据
    if refresh_btn or 'csi300_data' not in st.session_state:
        st.session_state.scanning = True
        try:
            with st.status("获取沪深300估值数据...", expanded=True) as status:
                prog = st.progress(0, text="加载中...")
                prog.progress(0.3, text="获取数据中（约10-20秒）...")
                data = _call_with_timeout(get_csi300_pe_data, 60)
                prog.progress(0.9, text="数据处理中...")

                if data is not None:
                    st.session_state.csi300_data = data
                    st.session_state.csi300_loaded = True
                    st.session_state.csi300_time = datetime.datetime.now().strftime('%H:%M:%S')
                    status.update(label="✅ 数据加载完成", state="complete")
                else:
                    st.session_state.csi300_data = None
                    status.update(label="❌ 加载失败", state="error")
                prog.progress(1.0, text="完成")
        finally:
            st.session_state.scanning = False

    # 显示数据
    data = st.session_state.get('csi300_data')
    if data is None and 'csi300_loaded' in st.session_state:
        st.error("❌ 获取沪深300数据失败，请检查网络后重试")
    elif data is not None:
        df = data['df']

        # ── 当前状态卡片 ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("当前PE", data['current_pe'])
        col2.metric("近15年PE分位值", f"{data['percentile']:.1f}%")

        # 如果有中证指数官方PE，额外显示
        csi_pe = data.get('csi_pe')
        if csi_pe:
            col1.metric("中证指数官方PE", csi_pe,
                        delta=round(csi_pe - data['current_pe'], 2))

        pct = data['percentile']
        if pct <= 20:
            signal = "🔴 买入区间"
            signal_help = f"PE ≤ {data['buy_threshold']}，处于近15年低估区，建议分批买入"
        elif pct >= 80:
            signal = "🟢 卖出区间"
            signal_help = f"PE ≥ {data['sell_threshold']}，处于近15年高估区，建议卖出"
        else:
            signal = "⚪ 持有/观望"
            signal_help = "PE处于中间区间，建议持有不动"
        col3.metric("当前信号", signal)
        col4.metric("数据源", data.get('source', '—'))

        st.info(signal_help)

        # 数据源差异说明
        if csi_pe and abs(csi_pe - data['current_pe']) > 0.5:
            st.caption(
                f"📌 **数据说明**：当前PE因数据源不同存在差异。"
                f"「中证指数」为官方数据({csi_pe})，"
                f"「{data.get('source','')}」({data['current_pe']})主要用于计算历史分位值。"
                f"主流平台（且慢、雪球等）及本图表使用前者，分值略有不同属正常现象。"
                f"分位值反映近15年估值相对位置，趋势判断不变。"
            )

        # ── PE分位值仪表盘 ──
        st.subheader("PE分位值仪表盘")
        st.progress(pct / 100, text=f"当前处于近15年 {pct:.1f}% 位置")

        col_l, col_m, col_r = st.columns(3)
        col_l.metric("买入阈值(20%)", f"PE ≤ {data['buy_threshold']}")
        csi_pe = data.get('csi_pe')
        if csi_pe:
            col_m.metric("当前PE", csi_pe,
                         delta=round(csi_pe - data['current_pe'], 2))
        else:
            col_m.metric("当前PE", data['current_pe'])
        col_r.metric("卖出阈值(80%)", f"PE ≥ {data['sell_threshold']}")

        # ── PE走势与信号图 ──
        st.subheader("PE分位值走势（近15年）")

        df_chart = df[['date', data['pe_col']]].copy()
        df_chart.columns = ['date', 'PE']
        df_chart['PE分位值'] = df_chart['PE'].rank(pct=True) * 100
        df_chart['买入区'] = df_chart['PE分位值'].apply(lambda x: x if x <= 20 else None)
        df_chart['卖出区'] = df_chart['PE分位值'].apply(lambda x: x if x >= 80 else None)

        # 图表：中文月份，买入区=红色，卖出区=绿色
        chart_data = df_chart[['date', 'PE分位值', '买入区', '卖出区']].copy()
        base = alt.Chart(chart_data).encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y年%m月', title=None, labelAngle=-45,
                     labelFontSize=11, grid=False)),
        )
        # 主曲线
        main_line = base.mark_line(
            strokeWidth=2, color='#3a6ea5', point=False,
        ).encode(
            y=alt.Y('PE分位值:Q', title='分位值(%)', scale=alt.Scale(domain=[0, 100])),
        )
        # 买入区(红色) - 分位值≤20%
        buy_area = base.mark_area(
            color='#d62728', opacity=0.15,
        ).encode(y='买入区:Q')
        buy_line = base.mark_line(
            strokeWidth=2, color='#d62728', strokeDash=[4, 3],
        ).encode(y='买入区:Q')
        # 卖出区(绿色) - 分位值≥80%
        sell_area = base.mark_area(
            color='#2ca02c', opacity=0.15,
        ).encode(y='卖出区:Q')
        sell_line = base.mark_line(
            strokeWidth=2, color='#2ca02c', strokeDash=[4, 3],
        ).encode(y='卖出区:Q')
        # 20%和80%参考线
        rule_layer = alt.Chart(pd.DataFrame({'y': [20, 80]})).mark_rule(
            stroke='#888', strokeWidth=0.8, strokeDash=[4, 4], opacity=0.4
        ).encode(y='y:Q')
        # 悬浮锚点（方便鼠标悬停）
        hover_pts = base.mark_circle(strokeWidth=14, stroke='transparent', fill='transparent').encode(
            y='PE分位值:Q',
            tooltip=[alt.Tooltip('date:T', format='%Y年%m月%d日'),
                     alt.Tooltip('PE分位值:Q', title='分位值', format='.1f')],
        )
        st.altair_chart(
            (buy_area + sell_area + main_line + buy_line + sell_line + rule_layer + hover_pts)
            .interactive(),
            use_container_width=True,
        )

        # ── 历史信号表 ──
        st.subheader("历史信号")

        signals = df[df['buy_signal'] | df['sell_signal']].copy()
        if not signals.empty:
            signal_rows = []
            for _, row in signals.iterrows():
                signal_type = "🔴 买入" if row['buy_signal'] else "🟢 卖出"
                pe_val = float(row[data['pe_col']])
                pct_val = float(df[data['pe_col']].rank(pct=True).loc[row.name] * 100)
                signal_rows.append({
                    '日期': row['date'].strftime('%Y-%m-%d'),
                    '信号': signal_type,
                    'PE': round(pe_val, 2),
                    '分位值': f"{pct_val:.1f}%",
                })
            st.dataframe(pd.DataFrame(signal_rows), hide_index=True, width='stretch')
        else:
            # 用区间边界作为近似信号
            zones = df[df['zone'] != df['zone'].shift(1)].copy()
            zone_signals = []
            for _, row in zones.iterrows():
                z = row['zone']
                if z == '买入区':
                    pct_val = float(df[data['pe_col']].rank(pct=True).loc[row.name] * 100)
                    zone_signals.append({
                        '日期': row['date'].strftime('%Y-%m'),
                        '信号': '🔴 买入',
                        'PE': round(float(row[data['pe_col']]), 2),
                        '分位值': f"{pct_val:.1f}%",
                    })
                elif z == '卖出区':
                    pct_val = float(df[data['pe_col']].rank(pct=True).loc[row.name] * 100)
                    zone_signals.append({
                        '日期': row['date'].strftime('%Y-%m'),
                        '信号': '🟢 卖出',
                        'PE': round(float(row[data['pe_col']]), 2),
                        '分位值': f"{pct_val:.1f}%",
                    })
            if zone_signals:
                st.dataframe(pd.DataFrame(zone_signals), hide_index=True, width='stretch')
            else:
                st.info("近15年未触发买入/卖出信号")

        # ── 操作建议 ──
        st.subheader("操作建议")
        if pct <= 20:
            st.success(f"当前PE分位值 {pct:.1f}%，处于低估区间。\n\n"
                       f"**建议**：全仓买入沪深300指数基金（如 110020 易方达沪深300ETF联接），"
                       f"持有至卖出信号出现。")
        elif pct >= 80:
            st.warning(f"当前PE分位值 {pct:.1f}%，处于高估区间。\n\n"
                       f"**建议**：清仓全部持仓，落袋为安，等待PE分位值回落至20%以下再买入。")
        else:
            st.info(f"当前PE分位值 {pct:.1f}%，处于中间区间。\n\n"
                    f"**建议**：保持现有持仓不动，既不加仓也不减仓。每月定投可继续，但无需额外操作。")

    elif 'csi300_loaded' not in st.session_state:
        st.info("点击「刷新数据」按钮，获取沪深300近15年PE分位值数据")


# ── 底部 ──
st.divider()
st.caption("⚠️ 本工具仅供学习参考，不构成投资建议。股市有风险，投资需谨慎。")
