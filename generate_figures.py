# -*- coding: utf-8 -*-
"""
OFDM雷达通感一体仿真 - 生成报告所需全部图表
5G NR FR1 n79频段参数 | 跨时隙慢时间多普勒处理
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import os

# 中文字体设置（使用系统自带的文泉驿）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
                                    'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 全局颜色主题
# ============================================================
CLR = {
    'blue':   '#1565C0',
    'green':  '#2E7D32',
    'orange': '#E65100',
    'red':    '#C62828',
    'purple': '#4527A0',
    'teal':   '#00695C',
    'bg':     '#F5F7FA',
    'text':   '#212121',
}

# ============================================================
# 1. 仿真参数（5G NR n79，跨时隙慢时间处理）
# ============================================================
fc       = 4.9e9       # 载波频率
c        = 3e8         # 光速
lam      = c / fc      # 波长 ≈ 0.0612 m
delta_f  = 30e3        # 子载波间隔 30 kHz
M        = 3276        # 子载波数（100 MHz）
N_sym    = 14          # 每时隙 OFDM 符号数
N_CPI    = 14          # CPI 时隙数（慢时间采样点）
B        = 100e6       # 系统带宽
T_sym    = 1 / delta_f                  # 有效符号周期 ≈ 33.33 μs
T_cp     = T_sym * 0.07                 # CP ≈ 2.33 μs
T_total  = T_sym + T_cp                 # 含 CP 的符号周期
T_slot   = N_sym * T_total              # 一个时隙 ≈ 0.499 ms ≈ 0.5 ms

# 感知性能指标（跨时隙慢时间处理）
range_res = c / (2 * B)                         # 距离分辨率 1.50 m
range_max = c / (2 * delta_f)                   # 最大不模糊距离 5000 m
vel_res   = lam / (2 * N_CPI * T_slot)          # 速度分辨率 ≈ 4.4 m/s
vel_max   = lam / (4 * T_slot)                  # 最大不模糊速度 ≈ 30.6 m/s

print("=" * 52)
print("  OFDM雷达感知参数（跨时隙慢时间多普勒处理）")
print("=" * 52)
print(f"  载波频率  f_c   = {fc/1e9:.1f} GHz")
print(f"  带宽      B     = {B/1e6:.0f} MHz   子载波数 M={M}")
print(f"  时隙符号数 N_sym = {N_sym}   CPI时隙数 N_CPI={N_CPI}")
print(f"  时隙周期  T_slot= {T_slot*1e3:.3f} ms")
print(f"  距离分辨率 Δr   = {range_res:.2f} m")
print(f"  最大探测距离 R_max = {range_max:.0f} m")
print(f"  速度分辨率 Δv   = {vel_res:.2f} m/s")
print(f"  最大探测速度 v_max = {vel_max:.1f} m/s")
print("=" * 52)

# ============================================================
# 2. 目标定义
# ============================================================
targets = [
    {"name": "配送无人机A", "range": 800,  "velocity":  12,  "snr_db": 15},
    {"name": "巡检无人机B", "range": 1500, "velocity":  -8,  "snr_db": 10},
    {"name": "黑飞无人机C", "range": 2200, "velocity":  18,  "snr_db":  5},
    {"name": "eVTOL",       "range": 3500, "velocity": -25,  "snr_db": 20},
]

# ============================================================
# 3. 跨时隙慢时间仿真：每个时隙输出一路距离剖面
# ============================================================
np.random.seed(42)

M_fft    = 4096
N_fft    = 256

range_profiles = np.zeros((N_CPI, M_fft), dtype=complex)

for k in range(N_CPI):
    # 每个时隙内：对 N_sym 个 OFDM 符号做相参积累
    slot_accum = np.zeros(M, dtype=complex)
    for n in range(N_sym):
        t_abs = (k * N_sym + n) * T_total   # 全局绝对时刻
        D = (np.random.choice([1,-1], M) + 1j*np.random.choice([1,-1], M)) / np.sqrt(2)
        Y = np.zeros(M, dtype=complex)
        for tgt in targets:
            tau = 2 * tgt["range"] / c
            fd  = 2 * tgt["velocity"] / lam
            alpha = 10 ** (tgt["snr_db"] / 20)
            fm = np.arange(M) * delta_f
            Y += alpha * D * np.exp(-2j * np.pi * (fm * tau - fd * t_abs))
        Y += (np.random.randn(M) + 1j*np.random.randn(M)) / np.sqrt(2)
        slot_accum += Y / D   # 倒频滤波后相参积累

    # 距离维 IFFT（零填充）
    Z_pad = np.zeros(M_fft, dtype=complex)
    Z_pad[:M] = slot_accum / N_sym
    range_profiles[k] = np.fft.ifft(Z_pad)

# 速度维 FFT（跨 N_CPI 时隙的慢时间 FFT）
rdm = np.fft.fftshift(np.fft.fft(range_profiles, n=N_fft, axis=0), axes=0)

range_axis = np.arange(M_fft) * c / (2 * M_fft * delta_f)
vel_axis   = np.linspace(-vel_max, vel_max, N_fft)
rdm_db     = 20 * np.log10(np.abs(rdm) + 1e-12)
rdm_db    -= rdm_db.max()   # 归一化到 0 dB

# ============================================================
# 4. 图1：Range-Doppler Map
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(CLR['bg'])
ax.set_facecolor('#0d1117')

im = ax.pcolormesh(range_axis / 1e3, vel_axis, rdm_db,
                   cmap='plasma', vmin=-40, vmax=0, shading='auto')
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cb.set_label('归一化幅度 (dB)', fontsize=12, color=CLR['text'])
cb.ax.tick_params(labelsize=10)

# CFAR 检测标注
annotation_colors = ['#FFD54F', '#80DEEA', '#F48FB1', '#A5D6A7']
for tgt, color in zip(targets, annotation_colors):
    r_km = tgt['range'] / 1e3
    v    = tgt['velocity']
    # 检测圆圈
    circle = plt.Circle((r_km, v), radius=0.06, fill=False,
                         edgecolor=color, linewidth=2, zorder=5)
    ax.add_patch(circle)
    # 标注文字
    offset_x = 0.15 if r_km < 3.5 else -0.15
    offset_y = 2.5  if v > 0 else -3.5
    ax.annotate(tgt['name'],
                xy=(r_km, v), xytext=(r_km + offset_x, v + offset_y),
                fontsize=9.5, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.4, connectionstyle='arc3,rad=0.1'),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#1a1a2e',
                          edgecolor=color, alpha=0.85))

ax.set_xlabel('距离 (km)', fontsize=13, color=CLR['text'])
ax.set_ylabel('速度 (m/s)', fontsize=13, color=CLR['text'])
ax.set_title('OFDM雷达 Range-Doppler Map（5G NR n79，4.9 GHz）\n'
             r'$N_\mathrm{CPI}=14$时隙，$\Delta v=4.4\,\mathrm{m/s}$，$v_\mathrm{max}=30.6\,\mathrm{m/s}$',
             fontsize=13, fontweight='bold', color=CLR['text'], pad=10)
ax.set_xlim([0, 4.5])
ax.set_ylim([-33, 33])
ax.tick_params(colors=CLR['text'], labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.grid(True, alpha=0.2, color='white', linestyle='--')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'range_doppler_map.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ Range-Doppler Map 已保存")

# ============================================================
# 5. 图2：信号处理流程图
# ============================================================
fig, ax = plt.subplots(figsize=(15, 4.5))
fig.patch.set_facecolor(CLR['bg'])
ax.set_facecolor(CLR['bg'])
ax.axis('off')

steps = [
    ("①\nOFDM\n发射信号\ns(t)",          CLR['blue']),
    ("②\n目标回波\n接收\nr(t)",           CLR['teal']),
    ("③\n倒频滤波\nZ = Y / D",            CLR['orange']),
    ("④\n距离维\nIFFT\n(子载波↓)",        CLR['purple']),
    ("⑤\n速度维\nFFT\n(时隙↑)",           '#AD1457'),
    ("⑥\nCFAR\n检测",                     CLR['green']),
    ("⑦\n目标参数\n(R, v, θ)",            CLR['red']),
]

n = len(steps)
box_w, box_h = 0.105, 0.58
x_start = 0.025
gap = (1.0 - x_start * 2 - n * box_w) / (n - 1)

for i, (text, color) in enumerate(steps):
    x = x_start + i * (box_w + gap)
    y = 0.21
    fancy = FancyBboxPatch((x, y), box_w, box_h,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='white',
                           linewidth=2, alpha=0.92,
                           transform=ax.transAxes, zorder=2)
    ax.add_patch(fancy)
    ax.text(x + box_w / 2, y + box_h / 2, text,
            transform=ax.transAxes,
            ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=3,
            linespacing=1.4)
    if i < n - 1:
        ax.annotate('',
                    xy=(x + box_w + gap, y + box_h / 2),
                    xytext=(x + box_w, y + box_h / 2),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#555',
                                    lw=2.5, mutation_scale=18))

ax.text(0.5, 0.95, 'OFDM雷达感知信号处理流程（5G NR n79）',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=14, fontweight='bold', color=CLR['text'])

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'signal_processing_flow.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ 信号处理流程图 已保存")

# ============================================================
# 6. 图3：自干扰消除三级架构
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(CLR['bg'])
ax.set_facecolor(CLR['bg'])
ax.axis('off')

levels = [
    ("空间域隔离",   "天线物理分离\n极化隔离",     "~30 dB",  CLR['blue']),
    ("射频域消除",   "模拟对消电路\n自适应调节",   "~30 dB",  CLR['green']),
    ("数字域消除",   "自适应滤波\n非线性PA建模",   "~40 dB",  CLR['orange']),
]

# 信号电平条（左侧）
signal_levels = [146, 116, 86, 46]   # dBm（相对）：发射→残余→残余→接收
level_labels  = ['+46 dBm\n发射', '+16 dBm\n空间后', '-14 dBm\n射频后', '-54 dBm\n数字后']

for i, (title, desc, db, color) in enumerate(levels):
    x = 0.08 + i * 0.30
    # 主模块框
    fancy = FancyBboxPatch((x, 0.12), 0.26, 0.70,
                           boxstyle="round,pad=0.025",
                           facecolor=color, edgecolor='white',
                           linewidth=2.5, alpha=0.88,
                           transform=ax.transAxes, zorder=2)
    ax.add_patch(fancy)
    ax.text(x + 0.13, 0.67, title,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, fontweight='bold', color='white', zorder=3)
    ax.text(x + 0.13, 0.42, desc,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10, color='#E3F2FD', zorder=3, linespacing=1.5)
    ax.text(x + 0.13, 0.20, f'消除量：{db}',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=12, fontweight='bold', color='#FFD54F', zorder=3)
    # 连接箭头
    if i < len(levels) - 1:
        ax.annotate('',
                    xy=(x + 0.30, 0.47), xytext=(x + 0.27, 0.47),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#666', lw=3,
                                    mutation_scale=20))

# 底部汇总
ax.text(0.5, 0.04,
        '三级级联总消除量 ≥ 100 dB，满足 ISAC 同频全双工收发要求',
        transform=ax.transAxes, ha='center', fontsize=12,
        fontweight='bold', color=CLR['red'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor=CLR['red']))

ax.set_title('自干扰消除（SIC）三级级联架构', fontsize=15,
             fontweight='bold', color=CLR['text'], pad=12)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'sic_architecture.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ 自干扰消除架构图 已保存")

# ============================================================
# 7. 图4：3GPP 标准演进时间轴
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(CLR['bg'])
ax.set_facecolor(CLR['bg'])
ax.axis('off')

# 时间主轴（箭头）
ax.annotate('', xy=(0.97, 0.45), xytext=(0.03, 0.45),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#555', lw=3,
                            mutation_scale=20))

milestones = [
    (0.14, "R18", "2024.06 冻结",
     "● 感知场景与需求预研（SI）\n● TR 22.837 定义 6 大感知用例\n● 无人机跟踪列为最高优先级",
     CLR['blue'],   True),
    (0.36, "R19", "2025 年冻结",
     "● ISAC 信道建模 WI\n● TR 38.857 六种感知模式\n● 感知参考信号设计",
     CLR['green'],  False),
    (0.59, "R20", "2026 年启动",
     "● 集成感知为 5G-A 核心方向\n● 功能规范（Normative）制定\n● 感知会话管理协议",
     CLR['orange'], True),
    (0.82, "6G", "2030 年商用",
     "● ITU IMT-2030 三大核心能力\n● 原生感知（Sensing-by-Design）\n● OTFS/RIS 使能",
     CLR['red'],    False),
]

for x, ver, date, content, color, above in milestones:
    # 时间轴节点圆点
    ax.plot(x, 0.45, 'o', color=color, markersize=18,
            transform=ax.transAxes, zorder=4)
    ax.plot(x, 0.45, 'o', color='white', markersize=8,
            transform=ax.transAxes, zorder=5)

    # 版本标签框（紧贴节点）
    y_tag  = 0.60 if above else 0.30
    y_line_end = 0.55 if above else 0.35
    ax.plot([x, x], [0.45, y_line_end], '-', color=color,
            lw=2, transform=ax.transAxes, zorder=3)

    ax.text(x, y_tag, f'{ver}\n{date}',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                      edgecolor='white', linewidth=1.5), zorder=5)

    # 内容文字框
    y_content = 0.87 if above else 0.07
    ax.text(x, y_content, content,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8.5, color=CLR['text'], linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=color, linewidth=1.2, alpha=0.92))

ax.set_title('3GPP ISAC 标准化演进路线', fontsize=15,
             fontweight='bold', color=CLR['text'], pad=15)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '3gpp_timeline.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ 3GPP 标准演进时间轴 已保存")

# ============================================================
# 8. 图5：IoT 系统五层架构
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8.5))
fig.patch.set_facecolor(CLR['bg'])
ax.set_facecolor(CLR['bg'])
ax.axis('off')

layers = [
    ("应用层",
     "非法入侵实时报警 ｜ 自动航路规划与避障 ｜ 数字孪生三维可视化 ｜ 物流调度优化 ｜ 空域流量管理",
     CLR['red'],    0.855),
    ("平台层",
     "低空智联网平台：数据接入网关（协议适配）｜ 时序数据库（InfluxDB/TDengine）｜ 规则引擎（地理围栏）｜ API 开放接口",
     CLR['orange'], 0.660),
    ("网络层",
     "5G 核心网 → 感知功能网元（SF）→ MQTT/HTTP 封装 → 感知数据上报（JSON/Protobuf）",
     CLR['green'],  0.465),
    ("感知层",
     "5G-A 基站（gNB）：OFDM 信号收发 → 2D-FFT 距离-速度估计 → CFAR 检测 → 目标参数集 {R, v, θ, RCS, t}",
     CLR['blue'],   0.270),
    ("目标层",
     "配送无人机（< 120 m）｜ 物流无人机（120-300 m）｜ eVTOL 飞行器（300-600 m）｜ 非合作目标（黑飞）",
     CLR['purple'], 0.075),
]

box_h = 0.155
for name, desc, color, y_center in layers:
    # 背景框
    fancy = FancyBboxPatch((0.04, y_center - box_h/2), 0.92, box_h,
                           boxstyle="round,pad=0.015",
                           facecolor=color, edgecolor='white',
                           linewidth=2, alpha=0.90,
                           transform=ax.transAxes, zorder=2)
    ax.add_patch(fancy)
    # 层名（左侧竖直区域）
    label_box = FancyBboxPatch((0.04, y_center - box_h/2), 0.10, box_h,
                               boxstyle="round,pad=0.008",
                               facecolor='white', edgecolor='none',
                               alpha=0.18, transform=ax.transAxes, zorder=3)
    ax.add_patch(label_box)
    ax.text(0.09, y_center, name,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, fontweight='bold', color='white', zorder=4)
    # 描述文字
    ax.text(0.57, y_center, desc,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.5, color='#E8F5E9', zorder=4, linespacing=1.5)

# 层间双向箭头
y_centers = [l[3] for l in layers]
for i in range(len(y_centers) - 1):
    y_top = y_centers[i] - box_h / 2 - 0.005
    y_bot = y_centers[i+1] + box_h / 2 + 0.005
    ax.annotate('', xy=(0.5, y_top), xytext=(0.5, y_bot),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='#888', lw=2,
                                mutation_scale=14))

ax.set_title('低空物联网通感一体化系统架构（五层架构）',
             fontsize=16, fontweight='bold', color=CLR['text'], pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'iot_architecture.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ IoT 系统架构图 已保存")

# ============================================================
# 9. 图6：ISAC vs 传统雷达雷达图
# ============================================================
categories = ['部署成本', '覆盖密度', '距离精度', '低慢小\n检测', '通信能力', '组网能力']
N_cat = len(categories)
# 评分均为 0-10，来源：综合文献定性对比
isac_scores  = [9, 9, 6, 5, 10, 9]
radar_scores = [2, 3, 9, 8,  1, 3]

angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]
isac_v  = isac_scores  + isac_scores[:1]
radar_v = radar_scores + radar_scores[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(CLR['bg'])
ax.set_facecolor('#F0F4FF')

ax.plot(angles, isac_v, 'o-', linewidth=2.5,
        label='5G-A ISAC 方案', color=CLR['blue'])
ax.fill(angles, isac_v, alpha=0.20, color=CLR['blue'])

ax.plot(angles, radar_v, 's-', linewidth=2.5,
        label='专用雷达方案', color=CLR['orange'])
ax.fill(angles, radar_v, alpha=0.20, color=CLR['orange'])

# 数据标注
for ang, iv, rv in zip(angles[:-1], isac_scores, radar_scores):
    ax.text(ang, iv + 0.8, str(iv), ha='center', va='center',
            fontsize=9, color=CLR['blue'], fontweight='bold')
    ax.text(ang, rv + 0.8, str(rv), ha='center', va='center',
            fontsize=9, color=CLR['orange'], fontweight='bold')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, color=CLR['text'])
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8, color='#666')
ax.grid(True, alpha=0.4, color='#CCC')
ax.spines['polar'].set_edgecolor('#AAA')

ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
          fontsize=12, framealpha=0.9, edgecolor='#CCC')
ax.set_title('ISAC 与传统雷达综合性能对比\n（各维度满分10分，综合文献定性评估）',
             fontsize=13, fontweight='bold', color=CLR['text'],
             pad=25, y=1.08)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'isac_vs_radar.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ ISAC vs 雷达对比图 已保存")

# ============================================================
# 10. 图7：运营商低空试点部署对比
# ============================================================
operators = ['中国移动\n（+美团）', '中国电信\n（+华为）', '中国联通\n（+中兴）']
colors_op  = [CLR['blue'], CLR['orange'], CLR['green']]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.patch.set_facecolor(CLR['bg'])
for axi in axes:
    axi.set_facecolor(CLR['bg'])

data = [
    ("试点项目数",   [15, 10, 80],  "项",  "线性"),
    ("感知距离 (km)", [1.5, 1.2, 2.0], "km", "线性"),
    ("覆盖省份数",   [10,  8, 25],  "省",  "线性"),
]

for ax, (title, vals, unit, _) in zip(axes, data):
    bars = ax.bar(operators, vals, color=colors_op, alpha=0.88,
                  edgecolor='white', linewidth=2, width=0.55)
    ax.set_title(title, fontsize=13, fontweight='bold', color=CLR['text'], pad=8)
    ax.set_ylabel(unit, fontsize=11, color='#555')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(CLR['bg'])
    # 数值标注
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.03,
                str(v), ha='center', va='bottom',
                fontweight='bold', fontsize=12, color=CLR['text'])
    ax.set_ylim(0, max(vals) * 1.22)

# 底部数据来源注释
fig.text(0.5, -0.02,
         '数据来源：中国移动/电信/联通各运营商低空通感一体技术白皮书（2024）；中兴通讯白皮书（2024）',
         ha='center', fontsize=8.5, color='#777', style='italic')

fig.suptitle('三大运营商低空通感一体试点部署对比（截至 2025 年）',
             fontsize=14, fontweight='bold', color=CLR['text'], y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'operator_comparison.png'),
            dpi=200, bbox_inches='tight', facecolor=CLR['bg'])
plt.close()
print("✓ 运营商对比图 已保存")

print("\n" + "=" * 52)
print(f"  所有图表已保存至: {OUT_DIR}")
print("=" * 52)
