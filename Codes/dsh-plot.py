#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.ticker import FixedLocator

import lib.py.plot.plot as myplot
import lib.py.ns3.flowmonitor as flowmonitor

PFC_RESUME = 0
PFC_PAUSE = 1

hm_name = {
    'Normal': '$\mathrm{SIH}$',
    'DSH': '$\mathrm{DSH}$',
}

KB = 1000
MB = 1000 * KB
GB = 1000 * MB
Kbps = 1000
Mbps = 1000 * Kbps
Gbps = 1000 * Mbps
ns = 1000
us = 1000 * ns
ms = 1000 * us


def plot_fct_pattern(logdir, figdir):
    print('----------fct----------------:')
    total_load = 0.9
    back_loads = np.arange(0.2, 0.81, 0.1)
    incast_num = 64
    incast_size = 64
    n_pg = 8
    for pattern in ('hadoop', 'mining', 'cache'):
        for cc in ('DCQCN', 'PowerTCP'):
            print(cc, ':')
            results = {}
            back_lpp = myplot.LinePointPlot()
            incast_lpp = myplot.LinePointPlot()
            baseline = 'Normal'
            for hmid in ('DSH', 'Normal'):
                back_fcts, incast_fcts, overall_fcts = [], [], []
                for bload in back_loads:
                    expname = '{pattern}-1.0ratio-{hm}-PQ-{cc}-{bload:.1f}back-{iload:.1f}burst-{incast_num}x{incast_size}KB-{n_pg}pg'.format(
                        pattern=pattern,
                        hm=hmid, cc=cc, bload=bload, 
                        iload=total_load-bload, incast_num=incast_num,
                        incast_size=incast_size, n_pg=n_pg,
                    )
                    expdir = os.path.join(
                        'exp_benchmark_fct-{}'.format(expname),
                    )
                    # stat fct
                    logfile = os.path.join(logdir, expdir, 'fct.txt')
                    df = pd.read_csv(
                        logfile, delim_whitespace=True,
                        names=(
                            'srcip', 'dstip', 'srcport', 'dstport',
                            'pg', 'rxBytes', 'flow_start_time', 'fct'
                        ),
                    )
                    back_df = df[df.dstport == 100]
                    incast_df = df[df.dstport == 200]
                    assert len(incast_df) == 0 or incast_df['rxBytes'].max() == incast_size << 10
                    assert len(incast_df) == 0 or incast_df['rxBytes'].min() == incast_size << 10
                    back_fct = flowmonitor.get_fct_breakdown(back_df)
                    incast_fct = flowmonitor.get_fct_breakdown(incast_df)
                    overall_fct = flowmonitor.get_fct_breakdown(df)
                    back_fcts.append(back_fct)
                    incast_fcts.append(incast_fct)
                    overall_fcts.append(overall_fct)
                    # stat pfc
                df_back = pd.DataFrame(back_fcts)
                df_incast = pd.DataFrame(incast_fcts)
                df_overall = pd.DataFrame(overall_fcts)
                results[hmid] = {
                    'back': df_back,
                    'incast': df_incast,
                    'overall': df_overall,
                }
            # Normalize results
            for hmid in results:
                for type in ('back', 'incast', 'overall'):
                    results[hmid][type] = results[hmid][type] / results[baseline][type]
                    results[hmid][type]['back_load'] = back_loads
                    results[hmid][type]['incast_load'] = \
                        total_load - results[hmid][type]['back_load']
                print(hmid, ',', 'background', ':')
                print(results[hmid]['back'])
                print(hmid, ',', 'incast', ':')
                print(results[hmid]['incast'])
                back_lpp.plot(
                    back_loads,
                    results[hmid]['back']['overall_avg'],
                    label=hm_name[hmid],
                    clip_on=False,
                )
                incast_lpp.plot(
                    back_loads,
                    results[hmid]['incast']['overall_avg'],
                    label=hm_name[hmid],
                    clip_on=False,
                )
            # set plotting styles
            for lpp in (back_lpp, incast_lpp):
                lpp.ax.set_xlim(back_loads.min()-0.05, back_loads.max()+0.05)
                lpp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
                lpp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
                lpp.ax.yaxis.set_major_locator(MultipleLocator(0.2))
                lpp.ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
                lpp.ax.set_xlabel('背景流负载')
                lpp.ax.set_ylabel('归一化流完成时间')
                lpp.ax.set_ylim(0.4, 1.2)
            # incast_lpp.ax.set_ylim(0.4, 1.2)
            # save fig
            fig_fname = 'benchmark-pattern-avg-fct-back-{}-{}.pdf'.format(cc.lower(), pattern)
            fig_fname = os.path.join(figdir, fig_fname)
            back_lpp.fig.savefig(fig_fname)
            fig_fname = 'benchmark-pattern-avg-fct-incast-{}-{}.pdf'.format(cc.lower(), pattern)
            fig_fname = os.path.join(figdir, fig_fname)
            incast_lpp.fig.savefig(fig_fname)
            # plt.show()


def plot_pfc(logdir, figdir):
    print('----------pause duration-----------:')
    PFC_RESUME = 0
    PFC_PAUSE = 1
    total_load = 0.9
    back_loads = np.arange(0.2, 0.8, 0.1)
    incast_size = 64
    n_pg = 8
    for cc in ('DCQCN', 'HPCC', 'PowerTCP'):
        print(cc, ':')
        results = {}
        lpp = myplot.LinePointPlot()
        baseline = 'Normal'
        for hmid in ('DSH', 'Normal'):
            t_pauses = []
            for bload in back_loads:
                expname = '{hm}-DWRR-{cc}-{bload:.1f}back-{iload:.1f}burst-{incast_size}KB-{n_pg}pg'.format(
                    hm=hmid, cc=cc,
                    bload=bload, iload=total_load-bload,
                    incast_size=incast_size, n_pg=n_pg,
                )
                expdir = os.path.join(
                    'exp_benchmark_fct-xpod_{}'.format(expname),
                )
                # stat fct
                logfile = os.path.join(logdir, expdir, 'pfc.txt')
                df = pd.read_csv(
                    logfile, delim_whitespace=True,
                    names=('time', 'nodeid', 'nodetype', 'portid', 'pg', 'type'),
                )
                df_pause = df[df['type'] == PFC_PAUSE]
                df_resume = df[df['type'] == PFC_RESUME]
                assert len(df_pause) == len(df_resume)
                t_pause = df_resume['time'].sum() - df_pause['time'].sum()
                t_pauses.append(t_pause)
            results[hmid] = np.array(t_pauses)
        # Normalize results
        for hmid in results:
            results[hmid] = results[hmid] / results[baseline]
            lpp.plot(
                back_loads,
                results[hmid],
                label=hm_name[hmid],
                clip_on=False,
            )
            print(hmid, ':')
            print(results[hmid])
        # set plotting styles
        lpp.ax.set_xlim(back_loads.min()-0.05, back_loads.max()+0.05)
        lpp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        lpp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        lpp.ax.set_xlabel('Load of Background Traffic')
        lpp.ax.set_ylabel('Normalized Pause Duration')
        # back_lpp.ax.set_ylim(0.6, 1.2)
        lpp.ax.set_ylim(0.4, 1.2)
        # save fig
        fig_fname = 'benchmark-pause-duration-{}.pdf'.format(cc.lower())
        fig_fname = os.path.join(figdir, fig_fname)
        lpp.fig.savefig(fig_fname)
        
def plot_buffer_usage(logdir, figdir):
    print('----------buffer usage------------:')
    logfile = os.path.join(logdir, 'buffer-usage-hsih-func-verify.csv')
    data = pd.read_csv(logfile)
    lp = myplot.LinePlot()

    time = data['time']
    sram = data['sram']
    wcache = data['wcache']
    dram = data['dram']

    lp.plot(time, sram, label='$\mathrm{SRAM}$')
    lp.plot(time, wcache, label='$\mathrm{WCache}$')
    lp.plot(time, dram, label='$\mathrm{DRAM}')

    lp.ax.set_xlabel('Time')
    lp.ax.set_ylabel('Buffer Usage')

    # plt.show()
    fig_fname = 'hsih-func-verify.pdf'
    fig_fname = os.path.join(figdir, fig_fname)
    lp.fig.savefig(fig_fname)
    
def plot_buffer_size(logdir, figdir):
    print('----------buffer size------------:')
    chip = ['$\mathrm{Trident+}$', '$\mathrm{Trident2}$', '$\mathrm{Tomahawk2}$', '$\mathrm{Tomahawk3}$', '$\mathrm{Tomahawk4}$']
    buffer_size = np.array([9.6, 12.8, 42.0, 64.0, 113.0]) # MB
    headroom_size = np.array([4.07, 5.78, 21.65, 38.29, 76.58]) # MB
    capacity = np.array([480.0, 1280.0, 6400.0, 12800.0, 25600.0]) # Gbps
    buffer_per_capacity = 8 * buffer_size * MB / (capacity * Gbps) * us
    headroom_per_capacity = 8 * headroom_size * MB / (capacity * Gbps) * us
    bottom_headroom_per_capacity = buffer_per_capacity - headroom_per_capacity
    fraction = headroom_size / buffer_size

    # set style
    style_fname = './lib/py/plot/paper.mplstyle' 
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(
        os.path.join(cur_dir, style_fname)
    )  
    
    axes = [] 
    fig, ax1 = plt.subplots(figsize=(12, 5.2))
    axes.append(ax1)

    # create left Y-axis
    ax1.bar(chip, buffer_per_capacity, color='#a0a0a0', width=0.4,label='缓存大小')
    ax1.set_ylabel('缓存容量/带宽（$\mathrm{us}$）')
    ax1.set_ylim([0, 160])
    ax1.set_xlim([-0.75, 4.75])
    ax1.tick_params(axis='x', colors='black')

    ax1.bar(chip, headroom_per_capacity, bottom=bottom_headroom_per_capacity, 
            color='#a00000', width=0.36, label='净空缓存大小')

    # create right Y-axis
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(chip, fraction, color='green', marker='x', markersize=20, 
             linestyle='-', markeredgewidth=4, linewidth=4, label='净空缓存占比')
    ax2.set_ylabel('净空缓存占比')
    ax2.set_ylim([0.4, 0.8])
    
    # merge the labels
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = ['缓存大小', '净空缓存大小', '净空缓存占比']
    ax1.legend(handles, labels, loc='upper center')
    
    # set ax style
    for ax in axes:
        ax.minorticks_on()
        ax.grid(
            b=True, axis='both', which='major',
            ls='dotted', linewidth=2, alpha=0.9,
        )
        ax.grid(
            b=True, axis='both', which='minor',
            ls='dotted', linewidth=1.5, alpha=0.5,
        )
        ax.yaxis.set_major_locator(MultipleLocator(40))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

    fig_fname = 'buffer_size.png'
    fig_fname = os.path.join(figdir, fig_fname)
    fig.savefig(fig_fname)
    
    # plt.show()
    
    
def plot_buffer_size_effect(logdir, figdir):
    print('----------buffer size effect------------:')    
    cc = 'PowerTCP'
    buffer_size = [i for i in range(14, 31)]
    avg_fcts = []
    for size in buffer_size:
        datadir = 'exp_motivation_fct'
        expdir = 'exp_motivation_fct-xpod_{cc}-Normal-{buffer_size}M-DWRR-0.9back-0.0burst-8pg'.format(
                    cc=cc, buffer_size=size,
                )
        logfile = os.path.join(logdir, datadir, expdir, 'fct.txt')
        df = pd.read_csv(
            logfile, delim_whitespace=True,
            names=('src_ip', 'dst_ip', 'src_port', 'dst_port', 'pg',
                   'flow_size', 'start_time', 'fct'),
        )
        df_fct = df['fct']
        avg_fcts.append(np.mean(df_fct) * ns / ms)   
    
    lp = myplot.LinePointPlot()
    plt.xlabel('缓存大小（$\mathrm{MB}$）')
    plt.ylabel('平均流完成时间（$\mathrm{ms}$）')
    lp.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    lp.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    lp.ax.set_ylim([1.5, 3.0])
    lp.plot(buffer_size, avg_fcts)
    fig_fname = 'buffer_size_effect.pdf'
    fig_fname = os.path.join(figdir, fig_fname)
    lp.fig.savefig(fig_fname)
    # plt.show()
    
def plot_headroom_utilization(logdir, figdir):
    print('----------headroom utilization------------:')
    
    datadir = 'headroom-utilization'
    logfile = os.path.join(logdir, datadir, 'hdrm_local_max_value-web.txt')
    df = pd.read_csv(
        logfile, delim_whitespace=True,
        names=('time', 'sw_id', 'hdrm_bytes', 'hdrm_ratio'),
    )
    df_hdrm_ratio = df['hdrm_ratio'] * 100
    cdf = myplot.CDFPlot()
    plt.xlabel('净空缓存利用率（%）')
    cdf.ax.set_xlim([0, 60])
    cdf.plot(df_hdrm_ratio)
    fig_fname = 'headroom_utilization.pdf'
    fig_fname = os.path.join(figdir, fig_fname)
    cdf.fig.savefig(fig_fname)
    # plt.show()
    
def plot_pfc_avoidance(logdir, figdir):
    print('----------pfc avoidance------------:')
    sih_pause = []
    dsh_pause = []
    burst_perc = [i / 10 for i in range(0,  11)]
    for mmu in ['DSH', 'Normal']:
        for burst in burst_perc:
            datadir = 'exp_pfc_avoidance'
            expdir = 'exp_pfc_avoidance_{mmu}-DWRR-None-16h-{perc}bp-8pg'.format(
                        mmu=mmu, perc=burst,
                    )
            logfile = os.path.join(logdir, datadir, expdir, 'pfc.txt')
            df = pd.read_csv(
                logfile, delim_whitespace=True,
                names=('time', 'nodeid', 'nodetype', 'portid', 'pg', 'type'),
            )

            # delete mismatched data
            for index, row in df.iterrows():
                if row['type'] == 1:
                    resume_row = df.loc[(df['type'] == 0) & (df['nodeid'] == row['nodeid']) &
                                        (df['nodetype'] == row['nodetype']) & 
                                        (df['portid'] == row['portid']) &
                                        (df['pg'] == row['pg']) & 
                                        (df.index > index)].head(1)
                    if resume_row.empty or resume_row.iloc[0]['type'] == 1:
                        df.drop(index, inplace=True)
            
            df_pause = df[df['type'] == PFC_PAUSE]
            df_resume = df[df['type'] == PFC_RESUME]
            assert len(df_pause) == len(df_resume)
            t_pause = (df_resume['time'].sum() - df_pause['time'].sum()) / ms * ns - 100
            
            if mmu == 'DSH':
                dsh_pause.append(t_pause)
            else:
                sih_pause.append(t_pause)
    
    lpp = myplot.LinePointPlot()
    lpp.plot(burst_perc, dsh_pause, label=hm_name['DSH'], clip_on=False)
    lpp.plot(burst_perc, sih_pause, label=hm_name['Normal'], clip_on=False)

    # set plotting styles
    # lpp.ax.set_xlim(back_loads.min()-0.05, back_loads.max()+0.05)
    # lpp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
    # lpp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    lpp.ax.yaxis.set_minor_locator(MultipleLocator(1))
    lpp.ax.yaxis.set_major_locator(MultipleLocator(5))
    lpp.ax.set_xlabel('突发大小占缓存容量比例')
    lpp.ax.set_ylabel('暂停时间（$\mathrm{ms}$)')
    # back_lpp.ax.set_ylim(0.6, 1.2)
    lpp.ax.set_xlim(0, 1)
    lpp.ax.set_ylim(0, 20)
    # save fig
    fig_fname = 'pfc-avoidance.pdf'
    fig_fname = os.path.join(figdir, fig_fname)
    lpp.fig.savefig(fig_fname)
    # plt.show()
    
def plot_deadlock_avoidance(logdir, figdir):
    print('----------pfc avoidance------------:')
    start_time = 2000000000
    ccs = ['PowerTCP', 'DCQCN']
    mmus = ['Normal', 'DSH']
    flow = 'hadoop'
    result = {}
    for cc in ccs:
        result[cc] = {}
        for mmu in mmus:
            datadir = 'exp_deadlock'
            filename = '{cc}-{mmu}-{flow}.txt'.format(
                        cc=cc, mmu=mmu, flow=flow
                    )
            logfile = os.path.join(logdir, datadir, filename)
            df = pd.read_csv(
                logfile, delim_whitespace=True,
                names=('test_id', 'state', 'time'),
            )
            df_deadlock_time = df['time']
            result[cc][mmu] = (df_deadlock_time - start_time) / ms * ns
    cp = myplot.CDFPlot()
    plt.xlabel('死锁时间（$\mathrm{ms}$）')
    cp.plot(result['DCQCN']['Normal'], label='$\mathrm{SIH/DCQCN}$')
    cp.plot(result['PowerTCP']['Normal'], label='$\mathrm{SIH/PowerTCP}$')
    cp.plot(result['DCQCN']['DSH'], label='$\mathrm{DSH/DCQCN}$')
    cp.plot(result['PowerTCP']['DSH'], label='$\mathrm{DSH/PowerTCP}$')   
    cp.ax.set_xlim([0, 100]) 
    # plt.show()
    fig_fname = 'deadlock-avoidance.pdf'
    fig_fname = os.path.join(figdir, fig_fname)
    cp.fig.savefig(fig_fname)
    
def plot_collateral_damage(logdir, figdir):
    burst_time = 2001000000
    start_time = 2003800000
    stop_time = 2004600000 
    interval = 20000
    sw_id = 29
    f0_id = 25 * 2 + 1
    f1_id = 26 * 2 + 1

    result = {}
    time = [(i - start_time) * ns / ms for i in range(start_time, stop_time, interval)]
    for mmu in ['DSH', 'Normal']:
        result[mmu] = {}
        for cc in ['DCQCN', 'None', 'PowerTCP']:
            result[mmu][cc] = {}
            datadir = 'collateral-damage/exp_collateral_damage'
            expdir = 'exp_collateral_damage_{mmu}-DWRR-{cc}-24h-8pg'.format(
                        mmu=mmu, cc=cc,
                    )
            logfile = os.path.join(logdir, datadir, expdir, 'throughput.txt')
            # df = pd.read_csv(logfile, delim_whitespace=True)
            # print(df.shape)
            # df = df[(df.iloc[:, 1] == sw_id) & 
            #         (df.iloc[:, 0] >= start_time) & 
            #         (df.iloc[:, 0] < stop_time)]
            
            # result[mmu][cc]['f0'] = df.iloc[:, f0_id]
            # result[mmu][cc]['f1'] = df.iloc[:, f1_id]
            
            # print(result[mmu][cc]['f0'])
            # print(result[mmu][cc]['f1'])
            
            result[mmu][cc]['f0'] = []
            result[mmu][cc]['f1'] = []
            
            file = open(logfile)
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(' ')
                if int(line[0]) < start_time:
                    continue
                if int(line[0]) >= stop_time:
                    break
                if int(line[1]) == sw_id:
                    result[mmu][cc]['f0'].append(float(line[f0_id]))
                    result[mmu][cc]['f1'].append(float(line[f1_id]))
    
    for cc in ['DCQCN', 'None', 'PowerTCP']:
        ldp = myplot.LineDashPlot()

        plt.xlabel('时间（$\mathrm{ms}$）')
        plt.ylabel('吞吐率（$\mathrm{Gbps}$）')
        ldp.ax.set_xlim(0, 0.8)
        ldp.ax.set_ylim(0, 100)
        ldp.plot(time, result['DSH'][cc]['f1'], label=hm_name['DSH'])
        ldp.plot(time, result['Normal'][cc]['f1'], label=hm_name['Normal'])
        
        fig_fname = f'collateral-damage-{cc}.pdf'
        fig_fname = os.path.join(figdir, fig_fname)
        ldp.fig.savefig(fig_fname)
        # plt.show()  
        
def plot_fct_benchmark(logdir, figdir):
    print('----------fct----------------:')
    total_load = 0.9
    back_loads = np.arange(0.2, 0.81, 0.1)
    incast_size = 64
    n_pg = 8
    for cc in ('DCQCN', 'HPCC', 'PowerTCP'):
        print(cc, ':')
        results = {}
        back_lpp = myplot.LinePointPlot()
        incast_lpp = myplot.LinePointPlot()
        baseline = 'Normal'
        for hmid in ('DSH', 'Normal'):
            back_fcts, incast_fcts, overall_fcts = [], [], []
            for bload in back_loads:
                expname = '{hm}-DWRR-{cc}-{bload:.1f}back-{iload:.1f}burst-{incast_size}KB-{n_pg}pg'.format(
                    hm=hmid, cc=cc,
                    bload=bload, iload=total_load-bload,
                    incast_size=incast_size, n_pg=n_pg,
                )
                expdir = os.path.join(
                    'exp_benchmark_fct-xpod_{}'.format(expname),
                )
                # stat fct
                logfile = os.path.join(logdir, expdir, 'fct.txt')
                df = pd.read_csv(
                    logfile, delim_whitespace=True,
                    names=(
                        'srcip', 'dstip', 'srcport', 'dstport',
                        'pg', 'rxBytes', 'flow_start_time', 'fct'
                    ),
                )
                back_df = df[df.pg != 1]
                incast_df = df[df.pg == 1]
                assert len(incast_df) == 0 or incast_df['rxBytes'].max() == incast_size << 10
                assert len(incast_df) == 0 or incast_df['rxBytes'].min() == incast_size << 10
                back_fct = flowmonitor.get_fct_breakdown(back_df)
                incast_fct = flowmonitor.get_fct_breakdown(incast_df)
                overall_fct = flowmonitor.get_fct_breakdown(df)
                back_fcts.append(back_fct)
                incast_fcts.append(incast_fct)
                overall_fcts.append(overall_fct)
                # stat pfc
            df_back = pd.DataFrame(back_fcts)
            df_incast = pd.DataFrame(incast_fcts)
            df_overall = pd.DataFrame(overall_fcts)
            results[hmid] = {
                'back': df_back,
                'incast': df_incast,
                'overall': df_overall,
            }
        # Normalize results
        for hmid in results:
            for type in ('back', 'incast', 'overall'):
                results[hmid][type] = results[hmid][type] / results[baseline][type]
                results[hmid][type]['back_load'] = back_loads
                results[hmid][type]['incast_load'] = \
                    total_load - results[hmid][type]['back_load']
            print(hmid, ',', 'background', ':')
            print(results[hmid]['back'])
            print(hmid, ',', 'incast', ':')
            print(results[hmid]['incast'])
            back_lpp.plot(
                back_loads,
                results[hmid]['back']['overall_avg'],
                label=hm_name[hmid],
                clip_on=False,
            )
            incast_lpp.plot(
                back_loads,
                results[hmid]['incast']['overall_avg'],
                label=hm_name[hmid],
                clip_on=False,
            )
        # set plotting styles
        for lpp in (back_lpp, incast_lpp):
            lpp.ax.set_xlim(back_loads.min()-0.05, back_loads.max()+0.05)
            lpp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
            lpp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            lpp.ax.yaxis.set_major_locator(MultipleLocator(0.2))
            lpp.ax.yaxis.set_minor_locator(MultipleLocator(0.05))            
            lpp.ax.set_xlabel('背景流负载')
            lpp.ax.set_ylabel('归一化流完成时间')
        back_lpp.ax.set_ylim(0.6, 1.2)
        incast_lpp.ax.set_ylim(0.4, 1.2)
        # save fig
        fig_fname = 'benchmark-avg-fct-back-{}.pdf'.format(cc.lower())
        fig_fname = os.path.join(figdir, fig_fname)
        back_lpp.fig.savefig(fig_fname)
        fig_fname = 'benchmark-avg-fct-incast-{}.pdf'.format(cc.lower())
        fig_fname = os.path.join(figdir, fig_fname)
        incast_lpp.fig.savefig(fig_fname)  
        # plt.show()
    
def plot_fct_fattree(logdir, figdir):
    print('----------fct----------------:')
    total_load = 0.9
    back_loads = np.arange(0.2, 0.81, 0.1)
    incast_num = 128
    incast_size = 64
    n_pg = 8
    pattern = 'cache'
    # for pattern in ('hadoop', 'mining', 'cache'):
    for cc in ('DCQCN', 'PowerTCP'):
        print(cc, ':')
        results = {}
        back_lpp = myplot.LinePointPlot()
        incast_lpp = myplot.LinePointPlot()
        baseline = 'Normal'
        for hmid in ('DSH', 'Normal'):
            back_fcts, incast_fcts, overall_fcts = [], [], []
            for bload in back_loads:
                expname = '{hm}-PQ-{cc}-{bload:.1f}back-{iload:.1f}burst-{incast_num}x{incast_size}KB-{n_pg}pg-0.1'.format(
                    pattern=pattern,
                    hm=hmid, cc=cc, bload=bload, 
                    iload=total_load-bload, incast_num=incast_num,
                    incast_size=incast_size, n_pg=n_pg,
                )
                expdir = os.path.join(
                    'exp_benchmark_fattree_k16-{}'.format(expname),
                )
                # stat fct
                logfile = os.path.join(logdir, expdir, 'fct.txt')
                df = pd.read_csv(
                    logfile, delim_whitespace=True,
                    names=(
                        'srcip', 'dstip', 'srcport', 'dstport',
                        'pg', 'rxBytes', 'flow_start_time', 'fct'
                    ),
                )
                back_df = df[df.dstport == 100]
                incast_df = df[df.dstport == 200]
                assert len(incast_df) == 0 or incast_df['rxBytes'].max() == incast_size << 10
                assert len(incast_df) == 0 or incast_df['rxBytes'].min() == incast_size << 10
                back_fct = flowmonitor.get_fct_breakdown(back_df)
                incast_fct = flowmonitor.get_fct_breakdown(incast_df)
                overall_fct = flowmonitor.get_fct_breakdown(df)
                back_fcts.append(back_fct)
                incast_fcts.append(incast_fct)
                overall_fcts.append(overall_fct)
                # stat pfc
            df_back = pd.DataFrame(back_fcts)
            df_incast = pd.DataFrame(incast_fcts)
            df_overall = pd.DataFrame(overall_fcts)
            results[hmid] = {
                'back': df_back,
                'incast': df_incast,
                'overall': df_overall,
            }
        # Normalize results
        for hmid in results:
            for type in ('back', 'incast', 'overall'):
                results[hmid][type] = results[hmid][type] / results[baseline][type]
                results[hmid][type]['back_load'] = back_loads
                results[hmid][type]['incast_load'] = \
                    total_load - results[hmid][type]['back_load']
            print(hmid, ',', 'background', ':')
            print(results[hmid]['back'])
            print(hmid, ',', 'incast', ':')
            print(results[hmid]['incast'])
            back_lpp.plot(
                back_loads,
                results[hmid]['back']['overall_avg'],
                label=hm_name[hmid],
                clip_on=False,
            )
            incast_lpp.plot(
                back_loads,
                results[hmid]['incast']['overall_avg'],
                label=hm_name[hmid],
                clip_on=False,
            )
        # set plotting styles
        for lpp in (back_lpp, incast_lpp):
            lpp.ax.set_xlim(back_loads.min()-0.05, back_loads.max()+0.05)
            lpp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
            lpp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            lpp.ax.yaxis.set_major_locator(MultipleLocator(0.2))
            lpp.ax.yaxis.set_minor_locator(MultipleLocator(0.05))             
            lpp.ax.set_xlabel('背景流负载')
            lpp.ax.set_ylabel('归一化流完成时间')
        back_lpp.ax.set_ylim(0.6, 1.2)
        incast_lpp.ax.set_ylim(0.4, 1.2)
        # save fig
        fig_fname = 'benchmark-fattree-avg-fct-back-{}-{}.pdf'.format(cc.lower(), pattern)
        fig_fname = os.path.join(figdir, fig_fname)
        back_lpp.fig.savefig(fig_fname)
        fig_fname = 'benchmark-fattree-avg-fct-incast-{}-{}.pdf'.format(cc.lower(), pattern)
        fig_fname = os.path.join(figdir, fig_fname)
        incast_lpp.fig.savefig(fig_fname)
        # plt.show()    


def main():
    if len(sys.argv) > 1:
        basedir = sys.argv[1]
    else:
        basedir = os.path.join('F:\\', 'research', 'DSH')
    basedir = os.path.expanduser(basedir)
    figdir = os.path.join(basedir, 'figs')
    logdir = os.path.join(basedir, 'data')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
        
    # plot_pfc(logdir, figdir)
    
    # # benchmark
    # datadir = os.path.join(logdir, "exp_benchmark")
    # plot_fct_benchmark(datadir, figdir)
    
    # # benchmark (hadoop/mining/cache)
    # datadir = os.path.join(logdir, "exp_benchmark_pattern")
    # plot_fct_pattern(datadir, figdir)
    
    # # benchmark (fattree)
    # datadir = os.path.join(logdir, "fattree")
    # plot_fct_fattree(datadir, figdir)
    
    # # plot_buffer_usage(logdir, figdir)
    
    plot_buffer_size(logdir, figdir)
    
    # plot_buffer_size_effect(logdir, figdir)
    
    # plot_headroom_utilization(logdir, figdir)
    
    # plot_pfc_avoidance(logdir, figdir)
    
    # plot_deadlock_avoidance(logdir, figdir)
    
    # plot_collateral_damage(logdir, figdir)


if __name__ == '__main__':
    main()
