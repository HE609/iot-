# -*- coding: utf-8 -*-
"""
OFDM雷达通感一体仿真 - 白底莫兰迪配色版
5G NR FR1 n79频段 | 跨时隙慢时间多普勒处理
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

# 莫兰迪配色
M_BLUE   = '#7B9EB2'
M_GREEN  = '#8BA78F'
M_ORANGE = '#C4956A'
M_RED    = '#B07A7A'
M_PURPLE = '#9B8EAD'
M_TEAL   = '#6D9A9B'
M_PINK   = '#C49BA5'
M_TEXT   = '#3C3C3C'

# ============ 仿真参数 ============
fc=4.9e9; c=3e8; lam=c/fc; delta_f=30e3; M=3276; N_sym=14; N_CPI=14; B=100e6
T_sym=1/delta_f; T_cp=T_sym*0.07; T_total=T_sym+T_cp; T_slot=N_sym*T_total
range_res=c/(2*B); range_max=c/(2*delta_f)
vel_res=lam/(2*N_CPI*T_slot); vel_max=lam/(4*T_slot)

print(f"Δr={range_res:.2f}m  R_max={range_max:.0f}m  Δv={vel_res:.2f}m/s  v_max={vel_max:.1f}m/s")

# ============ 目标与仿真 ============
targets = [
    {"name":"配送无人机A","range":800, "velocity":12, "snr_db":15},
    {"name":"巡检无人机B","range":1500,"velocity":-8, "snr_db":10},
    {"name":"黑飞无人机C","range":2200,"velocity":18, "snr_db":5},
    {"name":"eVTOL飞行器","range":3500,"velocity":-25,"snr_db":20},
]
np.random.seed(42)
M_fft=4096; N_fft=256
rp = np.zeros((N_CPI, M_fft), dtype=complex)
for k in range(N_CPI):
    sa = np.zeros(M, dtype=complex)
    for n in range(N_sym):
        t_abs=(k*N_sym+n)*T_total
        D=(np.random.choice([1,-1],M)+1j*np.random.choice([1,-1],M))/np.sqrt(2)
        Y=np.zeros(M,dtype=complex)
        for tgt in targets:
            tau=2*tgt["range"]/c; fd=2*tgt["velocity"]/lam; a=10**(tgt["snr_db"]/20)
            Y+=a*D*np.exp(-2j*np.pi*(np.arange(M)*delta_f*tau-fd*t_abs))
        Y+=(np.random.randn(M)+1j*np.random.randn(M))/np.sqrt(2)
        sa+=Y/D
    Zp=np.zeros(M_fft,dtype=complex); Zp[:M]=sa/N_sym; rp[k]=np.fft.ifft(Zp)
rdm=np.fft.fftshift(np.fft.fft(rp,n=N_fft,axis=0),axes=0)
ra=np.arange(M_fft)*c/(2*M_fft*delta_f); va=np.linspace(-vel_max,vel_max,N_fft)
rdm_db=20*np.log10(np.abs(rdm)+1e-12); rdm_db-=rdm_db.max()

# ============ 图1: Range-Doppler Map ============
fig,ax=plt.subplots(figsize=(11,6.5))
im=ax.pcolormesh(ra/1e3,va,rdm_db,cmap='viridis',vmin=-40,vmax=0,shading='auto')
cb=fig.colorbar(im,ax=ax,pad=0.02,fraction=0.04)
cb.set_label('归一化幅度 (dB)',fontsize=17)
cb.ax.tick_params(labelsize=14)

colors_t=[M_RED,M_BLUE,M_ORANGE,M_TEAL]
for tgt,clr in zip(targets,colors_t):
    rk=tgt['range']/1e3; v=tgt['velocity']
    circle=plt.Circle((rk,v),radius=0.06,fill=False,edgecolor=clr,linewidth=2.5,zorder=5)
    ax.add_patch(circle)
    ox=0.18 if rk<3 else -0.8; oy=3 if v>0 else -4
    ax.annotate(tgt['name'],xy=(rk,v),xytext=(rk+ox,v+oy),fontsize=16,color=clr,fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=clr,lw=1.8),
                bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=clr,alpha=0.9))

ax.set_xlabel('距离 (km)',fontsize=18,color=M_TEXT)
ax.set_ylabel('速度 (m/s)',fontsize=18,color=M_TEXT)
ax.set_title('OFDM雷达 Range-Doppler Map（5G NR n79, 4.9 GHz）\n'
             r'$N_{\mathrm{CPI}}=14$时隙，$\Delta v=4.4\,\mathrm{m/s}$，$v_{\mathrm{max}}=30.6\,\mathrm{m/s}$',
             fontsize=18,fontweight='bold',color=M_TEXT,pad=12)
ax.set_xlim([0,4.5]); ax.set_ylim([-33,33])
ax.tick_params(labelsize=14); ax.grid(True,alpha=0.3,linestyle='--')
plt.tight_layout()
fig.savefig(os.path.join(OUT,'range_doppler_map.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图1 RDM")

# ============ 图2: 信号处理流程 ============
fig,ax=plt.subplots(figsize=(15,4.5)); ax.axis('off')
steps=[("①\nOFDM\n发射信号\ns(t)",M_BLUE),("②\n目标回波\n接收\nr(t)",M_TEAL),
       ("③\n倒频滤波\nZ=Y/D",M_ORANGE),("④\n距离维\nIFFT\n(子载波↓)",M_PURPLE),
       ("⑤\n速度维\nFFT\n(时隙↑)",M_PINK),("⑥\nCFAR\n检测",M_GREEN),
       ("⑦\n目标参数\n(R, v, θ)",M_RED)]
n=len(steps); bw,bh=0.105,0.60; xs=0.025; gap=(1-xs*2-n*bw)/(n-1)
for i,(txt,clr) in enumerate(steps):
    x=xs+i*(bw+gap)
    fb=FancyBboxPatch((x,0.20),bw,bh,boxstyle="round,pad=0.02",facecolor=clr,
                      edgecolor='white',linewidth=2.5,alpha=0.88,transform=ax.transAxes,zorder=2)
    ax.add_patch(fb)
    ax.text(x+bw/2,0.20+bh/2,txt,transform=ax.transAxes,ha='center',va='center',
            fontsize=16,fontweight='bold',color='white',zorder=3,linespacing=1.3)
    if i<n-1:
        ax.annotate('',xy=(x+bw+gap,0.20+bh/2),xytext=(x+bw+0.005,0.20+bh/2),
                    xycoords='axes fraction',textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->',color='#888',lw=3,mutation_scale=20))
ax.text(0.5,0.95,'OFDM雷达感知信号处理流程（5G NR n79）',transform=ax.transAxes,
        ha='center',va='top',fontsize=20,fontweight='bold',color=M_TEXT)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'signal_processing_flow.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图2 流程")

# ============ 图3: SIC三级架构 ============
fig,ax=plt.subplots(figsize=(13,5.5)); ax.axis('off')
levels=[("空间域隔离","天线物理分离\n极化隔离","~30 dB",M_BLUE),
        ("射频域消除","模拟对消电路\n自适应调节","~30 dB",M_GREEN),
        ("数字域消除","自适应滤波\n非线性PA建模","~40 dB",M_ORANGE)]
for i,(t,d,db,clr) in enumerate(levels):
    x=0.08+i*0.30
    fb=FancyBboxPatch((x,0.15),0.26,0.68,boxstyle="round,pad=0.025",facecolor=clr,
                      edgecolor='white',linewidth=2.5,alpha=0.85,transform=ax.transAxes,zorder=2)
    ax.add_patch(fb)
    ax.text(x+0.13,0.67,t,transform=ax.transAxes,ha='center',va='center',
            fontsize=20,fontweight='bold',color='white',zorder=3)
    ax.text(x+0.13,0.44,d,transform=ax.transAxes,ha='center',va='center',
            fontsize=16,color='white',zorder=3,linespacing=1.5)
    ax.text(x+0.13,0.22,f'消除量：{db}',transform=ax.transAxes,ha='center',va='center',
            fontsize=17,fontweight='bold',color='#FFFDE7',zorder=3)
    if i<2:
        ax.annotate('',xy=(x+0.305,0.49),xytext=(x+0.265,0.49),
                    xycoords='axes fraction',textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->',color='#888',lw=3,mutation_scale=20))
ax.text(0.5,0.04,'三级级联总消除量 ≥ 100 dB，满足 ISAC 同频全双工收发要求',
        transform=ax.transAxes,ha='center',fontsize=16,fontweight='bold',color=M_RED,
        bbox=dict(boxstyle='round,pad=0.4',facecolor='#F5E6E6',edgecolor=M_RED))
ax.set_title('自干扰消除（SIC）三级级联架构',fontsize=20,fontweight='bold',color=M_TEXT,pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'sic_architecture.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图3 SIC")

# ============ 图4: 3GPP时间轴 ============
fig,ax=plt.subplots(figsize=(14,5.5)); ax.axis('off')
ax.annotate('',xy=(0.97,0.45),xytext=(0.03,0.45),xycoords='axes fraction',textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->',color='#999',lw=3,mutation_scale=20))
ms=[(0.14,"R18","2024.06 冻结","● 感知场景需求预研（SI）\n● TR 22.837 定义6大用例\n● 无人机跟踪最高优先级",M_BLUE,True),
    (0.36,"R19","2025年 冻结","● ISAC信道建模 WI\n● TR 38.857 六种感知模式\n● 感知参考信号设计",M_GREEN,False),
    (0.59,"R20","2026年 启动","● 集成感知为5G-A核心方向\n● 功能规范制定\n● 感知会话管理协议",M_ORANGE,True),
    (0.82,"6G","2030年 商用","● ITU IMT-2030核心能力\n● 原生感知Sensing-by-Design\n● OTFS/RIS使能",M_RED,False)]
for x,ver,date,content,clr,above in ms:
    ax.plot(x,0.45,'o',color=clr,markersize=20,transform=ax.transAxes,zorder=4)
    ax.plot(x,0.45,'o',color='white',markersize=9,transform=ax.transAxes,zorder=5)
    yt=0.62 if above else 0.28; yl=0.56 if above else 0.34
    ax.plot([x,x],[0.45,yl],'-',color=clr,lw=2.5,transform=ax.transAxes,zorder=3)
    ax.text(x,yt,f'{ver}\n{date}',transform=ax.transAxes,ha='center',va='center',
            fontsize=16,fontweight='bold',color='white',
            bbox=dict(boxstyle='round,pad=0.35',facecolor=clr,edgecolor='white',linewidth=1.5),zorder=5)
    yc=0.88 if above else 0.06
    ax.text(x,yc,content,transform=ax.transAxes,ha='center',va='center',
            fontsize=14,color=M_TEXT,linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',edgecolor=clr,linewidth=1.5,alpha=0.95))
ax.set_title('3GPP ISAC 标准化演进路线',fontsize=20,fontweight='bold',color=M_TEXT,pad=15)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'3gpp_timeline.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图4 3GPP")

# ============ 图5: IoT五层架构 ============
fig,ax=plt.subplots(figsize=(14,8.5)); ax.axis('off')
layers=[("应用层","非法入侵报警 ｜ 航路规划与避障 ｜ 数字孪生可视化 ｜ 物流调度 ｜ 空域管理",M_RED,0.855),
        ("平台层","低空智联网平台：数据接入网关 ｜ 时序数据库 ｜ 规则引擎（地理围栏）｜ API开放接口",M_ORANGE,0.660),
        ("网络层","5G核心网 → 感知功能网元（SF）→ MQTT/HTTP 封装 → JSON/Protobuf 上报",M_GREEN,0.465),
        ("感知层","5G-A基站（gNB）：OFDM收发 → 2D-FFT → CFAR检测 → {R, v, θ, RCS, t}",M_BLUE,0.270),
        ("目标层","配送无人机（<120m）｜ 物流无人机（120-300m）｜ eVTOL（300-600m）｜ 黑飞目标",M_PURPLE,0.075)]
bh=0.155
for nm,desc,clr,yc in layers:
    fb=FancyBboxPatch((0.04,yc-bh/2),0.92,bh,boxstyle="round,pad=0.015",facecolor=clr,
                      edgecolor='white',linewidth=2,alpha=0.82,transform=ax.transAxes,zorder=2)
    ax.add_patch(fb)
    ax.text(0.09,yc,nm,transform=ax.transAxes,ha='center',va='center',
            fontsize=20,fontweight='bold',color='white',zorder=4)
    ax.text(0.55,yc,desc,transform=ax.transAxes,ha='center',va='center',
            fontsize=15,color='white',zorder=4)
ycs=[l[3] for l in layers]
for i in range(len(ycs)-1):
    ax.annotate('',xy=(0.5,ycs[i]-bh/2-0.005),xytext=(0.5,ycs[i+1]+bh/2+0.005),
                xycoords='axes fraction',textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<->',color='#888',lw=2.5,mutation_scale=14))
ax.set_title('低空物联网通感一体化系统架构（五层）',fontsize=22,fontweight='bold',color=M_TEXT,pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'iot_architecture.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图5 IoT架构")

# ============ 图6: ISAC vs 雷达 ============
cats=['部署成本','覆盖密度','距离精度','低慢小\n检测','通信能力','组网能力']
nc=len(cats)
isac=[9,9,6,5,10,9]; radar=[2,3,9,8,1,3]
angles=np.linspace(0,2*np.pi,nc,endpoint=False).tolist(); angles+=angles[:1]
iv=isac+isac[:1]; rv=radar+radar[:1]
fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
ax.plot(angles,iv,'o-',lw=2.5,label='5G-A ISAC方案',color=M_BLUE,markersize=8)
ax.fill(angles,iv,alpha=0.15,color=M_BLUE)
ax.plot(angles,rv,'s-',lw=2.5,label='专用雷达方案',color=M_ORANGE,markersize=8)
ax.fill(angles,rv,alpha=0.15,color=M_ORANGE)
for ang,i2,r2 in zip(angles[:-1],isac,radar):
    ax.text(ang,i2+0.9,str(i2),ha='center',va='center',fontsize=15,color=M_BLUE,fontweight='bold')
    ax.text(ang,r2+0.9,str(r2),ha='center',va='center',fontsize=15,color=M_ORANGE,fontweight='bold')
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=17,color=M_TEXT)
ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10])
ax.set_yticklabels(['2','4','6','8','10'],fontsize=13,color='#888')
ax.grid(True,alpha=0.3,color='#CCC')
ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.12),fontsize=16,framealpha=0.9,edgecolor='#CCC')
ax.set_title('ISAC与传统雷达综合性能对比\n（各维度满分10分，综合文献定性评估）',
             fontsize=18,fontweight='bold',color=M_TEXT,pad=25,y=1.08)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'isac_vs_radar.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图6 雷达图")

# ============ 图7: 运营商对比 ============
ops=['中国移动\n（+美团）','中国电信\n（+华为）','中国联通\n（+中兴）']
cop=[M_BLUE,M_ORANGE,M_GREEN]
fig,axes=plt.subplots(1,3,figsize=(14,5.5))
data=[("试点项目数",[15,10,80],"项"),("感知距离 (km)",[1.5,1.2,2.0],"km"),("覆盖省份数",[10,8,25],"省")]
for ax,(title,vals,unit) in zip(axes,data):
    bars=ax.bar(ops,vals,color=cop,alpha=0.80,edgecolor='white',linewidth=2,width=0.55)
    ax.set_title(title,fontsize=18,fontweight='bold',color=M_TEXT,pad=8)
    ax.set_ylabel(unit,fontsize=15,color='#666')
    ax.tick_params(axis='x',labelsize=14); ax.tick_params(axis='y',labelsize=13)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*0.03,
                str(v),ha='center',va='bottom',fontweight='bold',fontsize=17,color=M_TEXT)
    ax.set_ylim(0,max(vals)*1.22)
fig.text(0.5,-0.02,'数据来源：中国移动/电信/联通低空通感一体技术白皮书（2024）；中兴通讯白皮书（2024）',
         ha='center',fontsize=12,color='#888',style='italic')
fig.suptitle('三大运营商低空通感一体试点部署对比（截至2025年）',
             fontsize=20,fontweight='bold',color=M_TEXT,y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'operator_comparison.png'),dpi=200,bbox_inches='tight',facecolor='white')
plt.close(); print("✓ 图7 运营商")
print(f"\n全部图表已保存至: {OUT}")
