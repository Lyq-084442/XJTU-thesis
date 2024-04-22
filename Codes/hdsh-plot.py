import sys
# sys.path.append("../../../../utils/plot")

import os
import xml.etree.ElementTree as etree
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
import os.path
# import plot as myplot
import lib.py.plot.plot as myplot
from matplotlib.ticker import MultipleLocator

KB = 1024
MB = KB * 1024
GB = MB * 1024
Kbps = 1000
Mbps = Kbps * 1000
Gbps = Mbps * 1000
ms = 1000
us = ms * 1000
ns = us * 1000

hm_name = {
    'H-SIH': '$\mathrm{H}$-$\mathrm{SIH}$',
    'H-DSH': '$\mathrm{H}$-$\mathrm{DSH}$',
}

def get_fct_from_xml(file_name):
    root = etree.parse(file_name).getroot()
    flow_stats = root[0]
    ipv4_flow_classifier = root[1]
    flow_dict = {}
    for child in ipv4_flow_classifier:
        flow_id = child.attrib["flowId"]
        src_ip = child.attrib["sourceAddress"]
        dst_ip = child.attrib["destinationAddress"]
        protocol = child.attrib["protocol"]
        src_port = child.attrib["sourcePort"]
        dst_port = child.attrib["destinationPort"]
        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        
        index = int(flow_id) - 1
        flow_stats_attr = flow_stats[index].attrib
        start_time = float(flow_stats_attr["timeFirstTxPacket"].replace("+","").replace("ns",""))
        stop_time = float(flow_stats_attr["timeLastRxPacket"].replace("+","").replace("ns",""))
        loss_packets = int(flow_stats_attr["lostPackets"])
        receive_packets = int(flow_stats_attr["rxPackets"])
        receive_bytes = int(flow_stats_attr["rxBytes"])
        
        complete_time = stop_time - start_time
        
        flow_dict[flow_key] = {}
        flow_dict[flow_key]["startTime"] = start_time
        flow_dict[flow_key]["stopTime"] = stop_time
        flow_dict[flow_key]["completeTime"] = complete_time
        flow_dict[flow_key]["isPositive"] = "UDP"   #   the first set Flase, we can delete
        flow_dict[flow_key]["lostPackets"] = loss_packets
        flow_dict[flow_key]["rxPackets"] = receive_packets
        flow_dict[flow_key]["rxBytes"] = receive_bytes
        flow_dict[flow_key]["flowid"] = int(flow_id)          
        
        base_rtt = 1.0 * 4 / us
        bandwidth = 100 * Gbps
        
        # flow slow down
        slow_down = complete_time / ns / (base_rtt + receive_bytes * 8.0 / bandwidth)
        flow_dict[flow_key]["fctSlowDown"] = slow_down
        
    flow_fct = []
    flow_fct_slowdown = []
    
    for flow_key, flow_value in flow_dict.items():
        flow_fct.append(flow_value["completeTime"])
        flow_fct_slowdown.append(flow_value["fctSlowDown"])
    
    flow_fct.sort()
    flow_fct_slowdown.sort()
    fct_99th_index = math.ceil(len(flow_fct) / 100 * 99) - 1
    
    result = {}
    result["fct_avg"] = np.mean(flow_fct)
    result["fct_99th"] = flow_fct[fct_99th_index]
    
    result["fct_slowdown_avg"] = np.mean(flow_fct_slowdown)
    result["fct_slowdown_99th"] = flow_fct_slowdown[fct_99th_index]
    
    # print("fct_avg: ", result["fct_avg"])
    # print("fct_99th: ", result["fct_99th"])
    # print("fct_slowdown_avg", result["fct_slowdown_avg"])
    # print("fct_slowdown_99th", result["fct_slowdown_99th"])
    
    return result

def plot_fct_slow_down(file_dir, save_path):
    
    bm_alg = ["hsih", "hdsh"]  
    fct_hsih = []
    fct_hdsh = []
    fct_slowdown_hsih = []
    fct_slowdown_hdsh = []
    for bm in bm_alg:
        for load in range(1, 10):
            file_name = f'flow-monitor-fct-evaluation-{bm}-load{load / 10}.xml'
            file_path = os.path.join(file_dir, file_name)
            result = get_fct_from_xml(file_path)  
            if bm == "hsih":
                fct_hsih.append(result["fct_avg"])
                fct_slowdown_hsih.append(result["fct_slowdown_avg"])
            elif bm == "hdsh":
                fct_hdsh.append(result["fct_avg"])
                fct_slowdown_hdsh.append(result["fct_slowdown_avg"])
    
    loads = [i/10 for i in range(1, 10)]
    
    fct_hsih_norm = [1.0 for fct in fct_hsih]
    fct_hdsh_norm = [fct_hdsh[i] / fct_hsih[i] for i in range(len(fct_hsih))]
    
    lp = myplot.LinePointPlot()
    myplot.plt.xlabel('背景流负载')
    myplot.plt.ylabel('归一化流完成时间')
    # lp.ax.set_ylim([0.6, 1.2])
    # lp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
    # lp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    # lp.ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # lp.ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
    lp.plot(loads, fct_hdsh_norm, label=hm_name['H-DSH'])
    lp.plot(loads, fct_hsih_norm, label=hm_name['H-SIH'])
    myplot.plt.savefig(save_path)
    
def plot_pfc_avoidance_incast(file_dir, save_path):
    
    bm_alg = ["hsih", "hdsh"]  
    burst_size = [8 * i for i in range(1, 9)]
    pause_duration_hsih = []
    pause_duration_hdsh = []
    for bm in bm_alg:
        for burst in burst_size:
            file_name = f'pfc-duration-pfc-avoidance-{bm}-burst{burst}.csv'
            file_path = os.path.join(file_dir, file_name)
            total_pause_duration = 0
            data = pd.read_csv(file_path)
            # burst_data = data[data['port'] > 3]
            total_pause_duration = data['duration'].sum() * ms / ns
            if bm == "hsih":
                pause_duration_hsih.append(total_pause_duration)
            elif bm == "hdsh":
                pause_duration_hdsh.append(total_pause_duration)
    
    pause_duration_hsih_norm = [1.0 for duration in pause_duration_hsih]
    pause_duration_hdsh_norm = [pause_duration_hdsh[i] / pause_duration_hsih[i] for i in range(len(pause_duration_hsih))]
    
    # print(pause_duration_hsih)
    # print(pause_duration_hdsh)
    
    lp = myplot.LinePointPlot()
    lp.ax.set_ylim([0, 120])
    lp.ax.xaxis.set_major_locator(MultipleLocator(8))
    lp.ax.xaxis.set_minor_locator(MultipleLocator(8))
    lp.ax.yaxis.set_major_locator(MultipleLocator(40))
    lp.ax.yaxis.set_minor_locator(MultipleLocator(10)) 
    myplot.plt.xlabel('突发大小（$\mathrm{MB}$）')
    myplot.plt.ylabel('暂停时长（$\mathrm{ms}$）')
    lp.plot(burst_size, pause_duration_hdsh, label=hm_name['H-DSH'])
    lp.plot(burst_size, pause_duration_hsih, label=hm_name['H-SIH'])
    myplot.plt.savefig(save_path)   
    
def plot_pfc_avoidance_web(file_dir, save_path):
    
    bm_alg = ["hsih", "hdsh"]  
    pause_duration_hsih = []
    pause_duration_hdsh = []
    burst_size = [16 * i for i in range(1, 9)]
    loads = [i/10 for i in range(1, 10)]
    for bm in bm_alg:
        for load in loads:
            file_name = f'pfc-duration-pfc-avoidance-{bm}-load{load}.csv'
            file_path = os.path.join(file_dir, file_name)
            total_pause_duration = 0
            data = pd.read_csv(file_path)
            # burst_data = data[data['port'] < 24]
            total_pause_duration = data['duration'].sum() * ms / ns
            if bm == "hsih":
                pause_duration_hsih.append(total_pause_duration)
            elif bm == "hdsh":
                pause_duration_hdsh.append(total_pause_duration)
    
    # print(pause_duration_hsih)
    # print(pause_duration_hdsh)
    
    lp = myplot.LinePointPlot()
    lp.ax.set_ylim([-10, 320])
    lp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
    lp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    lp.ax.yaxis.set_major_locator(MultipleLocator(80))
    lp.ax.yaxis.set_minor_locator(MultipleLocator(20)) 
    myplot.plt.xlabel('背景流负载')
    myplot.plt.ylabel('暂停时长（$\mathrm{ms}$）')
    lp.plot(loads, pause_duration_hdsh, label=hm_name['H-DSH'])
    lp.plot(loads, pause_duration_hsih, label=hm_name['H-SIH'])
    myplot.plt.savefig(save_path)  
    
def plot_collateral_damage_throughput(file_dir, save_path):
    
    bm_alg = ["hsih", "hdsh"]
    cc_alg = ["wo", "TcpLinuxReno", "TcpNewReno", "TcpBic", "TcpCubic", "TcpBbr"]
    # cc_alg = ["TcpCubic"]
    throughput = {}
    start_time = 1009800000
    end_time = start_time + 500000
    time_interval = 10000
    time = [(i * time_interval) / 1000000 for i in range(50)]
    for cc in cc_alg:
        cc_throughput = {}
        for bm in bm_alg:
            file_name = f'port-throughput-collateral-damage-{bm}-{cc}-p0.csv'
            file_path = os.path.join(file_dir, file_name)
            data = pd.read_csv(file_path)
            data = data[(data['start'] >= start_time) & (data['end'] <= end_time)]
            
            # insert the missing 0 sendRate
            current_start = 0
            next_start = 0
            for i in range(len(data) - 1):
                current_start = data['start'].iloc[i]
                next_start = data['start'].iloc[i + 1]
                time_diff = next_start - current_start
                while time_diff > time_interval:
                    current_start = current_start + time_interval
                    new_row = {
                                'start': current_start, 
                                'end': current_start + time_interval, 
                                'sendRate': 0
                               }
                    data.loc[len(data)] = new_row
                    time_diff = next_start - current_start
            while current_start + time_interval < end_time - time_interval:
                current_start = current_start + time_interval
                new_row = {
                            'start': current_start, 
                            'end': current_start + time_interval, 
                            'sendRate': 0
                            }
                data.loc[len(data)] = new_row                       
            
            data = data.sort_values(by='start').reset_index(drop=True)
            # print(data)
            
            rate_data = data['sendRate'] / Gbps
            cc_throughput[bm] = rate_data

        throughput[cc] = cc_throughput
   
    for cc in cc_alg:
        lp = myplot.LineDashPlot()
        # lp.ax.set_ylim([0, 100])
        # lp.ax.xaxis.set_major_locator(MultipleLocator(0.2))
        # lp.ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        # lp.ax.yaxis.set_major_locator(MultipleLocator(10))
        # lp.ax.yaxis.set_minor_locator(MultipleLocator(2))         
        myplot.plt.xlabel('时间（$\mathrm{ms}$）')
        myplot.plt.ylabel('吞吐率（$\mathrm{Gbps}$）')
        lp.plot(time, throughput[cc]['hdsh'], label=hm_name['H-DSH'])
        lp.plot(time, throughput[cc]['hsih'], label=hm_name['H-SIH'])
        save_fig = save_path + '-' + cc + '.pdf'
        myplot.plt.savefig(save_fig)  
    
    # Motivation
    for cc in cc_alg:
        lp = myplot.LineDashPlot()
        # lp.ax.set_ylim([0, 100])
        # lp.ax.xaxis.set_major_locator(MultipleLocator(0.2))
        # lp.ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        # lp.ax.yaxis.set_major_locator(MultipleLocator(10))
        # lp.ax.yaxis.set_minor_locator(MultipleLocator(2))          
        myplot.plt.xlabel('时间（$\mathrm{ms}$）')
        myplot.plt.ylabel('吞吐率（$\mathrm{Gbps}$）')
        lp.plot(time, throughput[cc]['hsih'])
        save_fig = save_path + '-' + cc + '-motivation.pdf'
        myplot.plt.savefig(save_fig) 
        
def plot_flow_pattern_cdf(file_dir, save_path):
    patterns = ['websearch', 'mining', 'hadoop', 'cache']
    patterns_name = {
        'websearch': '$\mathrm{Web \; Search}$',
        'mining': '$\mathrm{Data \; Mining}$',
        'hadoop': '$\mathrm{Hadoop}$',
        'cache': '$\mathrm{Cache}$'
    }
    lp = myplot.LineDashPlot()
    myplot.plt.xlabel('流大小（$\mathrm{KB}$)')
    myplot.plt.ylabel('$\mathrm{CDF}$')  
    ax = myplot.plt.gca()
    ax.set_xscale('log') 
    ax.xaxis.grid(False, which='minor') 
    lp.plot([], [])
    for pattern in patterns:
        file_path = file_dir + pattern + '.txt'
        flow_size = []
        flow_cdf = []
        with open(file_path, 'r') as file:
            for line in file:
                values = line.split()
                if len(values) >= 2:
                    flow_size.append(float(values[0]) / KB)
                    flow_cdf.append(float(values[1]))
                    
        lp.plot(flow_size, flow_cdf, label=patterns_name[pattern])

    myplot.plt.savefig(save_path) 
    
def plot_fct_evaluation_pattern(file_dir, save_path):
    bm_alg = ["hsih", "hdsh"] 
    patterns = ['websearch', 'mining', 'hadoop', 'cache'] 
    
    for pattern in patterns:
        fct_hsih = []
        fct_hdsh = []
        fct_slowdown_hsih = []
        fct_slowdown_hdsh = []
        for bm in bm_alg:
            for load in range(1, 10):
                file_name = f'flow-monitor-fct-evaluation-pattern-{bm}-load{load / 10}-{pattern}.xml'
                file_path = os.path.join(file_dir, file_name)
                result = get_fct_from_xml(file_path)  
                if bm == "hsih":
                    fct_hsih.append(result["fct_avg"])
                    fct_slowdown_hsih.append(result["fct_slowdown_avg"])
                elif bm == "hdsh":
                    fct_hdsh.append(result["fct_avg"])
                    fct_slowdown_hdsh.append(result["fct_slowdown_avg"])
        
        loads = [i/10 for i in range(1, 10)]
        
        fct_hsih_norm = [1.0 for fct in fct_hsih]
        fct_hdsh_norm = [fct_hdsh[i] / fct_hsih[i] for i in range(len(fct_hsih))]
        
        lp = myplot.LinePointPlot()
        # lp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        # lp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        # lp.ax.yaxis.set_major_locator(MultipleLocator(0.2))
        # lp.ax.yaxis.set_minor_locator(MultipleLocator(0.05))     
        myplot.plt.xlabel('背景流负载')
        myplot.plt.ylabel('归一化流完成时间')
        lp.plot(loads, fct_hdsh_norm, label=hm_name['H-DSH'])
        lp.plot(loads, fct_hsih_norm, label=hm_name['H-SIH'])
        save_fig = save_path + '-' + pattern + '.pdf'
        myplot.plt.savefig(save_fig)                   
 
def plot_fct_evaluation_cc(file_dir, save_path):
    bm_alg = ["hsih", "hdsh"] 
    cc_alg = ["TcpLinuxReno", "TcpNewReno", "TcpBic", "TcpCubic", "TcpBbr"]
    
    for cc in cc_alg:
        fct_hsih = []
        fct_hdsh = []
        fct_slowdown_hsih = []
        fct_slowdown_hdsh = []
        for bm in bm_alg:
            for load in range(1, 10):
                file_name = f'flow-monitor-fct-evaluation-cc-{bm}-load{load / 10}-{cc}.xml'
                file_path = os.path.join(file_dir, file_name)
                result = get_fct_from_xml(file_path)  
                if bm == "hsih":
                    fct_hsih.append(result["fct_avg"])
                    fct_slowdown_hsih.append(result["fct_slowdown_avg"])
                elif bm == "hdsh":
                    fct_hdsh.append(result["fct_avg"])
                    fct_slowdown_hdsh.append(result["fct_slowdown_avg"])
        
        loads = [i/10 for i in range(1, 10)]
        
        fct_hsih_norm = [1.0 for fct in fct_hsih]
        fct_hdsh_norm = [fct_hdsh[i] / fct_hsih[i] for i in range(len(fct_hsih))]
        
        lp = myplot.LinePointPlot()
        # lp.ax.set_ylim([0.6, 1.2])
        # lp.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        # lp.ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        # lp.ax.yaxis.set_major_locator(MultipleLocator(0.2))
        # lp.ax.yaxis.set_minor_locator(MultipleLocator(0.05))         
        myplot.plt.xlabel('背景流负载')
        myplot.plt.ylabel('归一化流完成时间')
        lp.plot(loads, fct_hdsh_norm, label=hm_name['H-DSH'])
        lp.plot(loads, fct_hsih_norm, label=hm_name['H-SIH'])
        save_fig = save_path + '-' + cc + '.pdf'
        myplot.plt.savefig(save_fig)   

def plot_burst_absorption(file_dir, save_path):
    bm_alg = ["hsih", "hdsh"]
    burst_size = [4 * i for i in range(1, 11)]
    
    dram_qlen_hsih = []
    dram_qlen_hdsh = []
    for bm in bm_alg:
        for burst in burst_size:
            file_name = f'buffer-usage-burst-absorption-{bm}-burst{burst}MB.csv'
            file_path = os.path.join(file_dir, file_name)
            data = pd.read_csv(file_path)
            dram = data['dram'] / MB
            dram = np.array(dram)
            dram = dram[dram > 0]
            avg_qlen = np.mean(dram)
            if bm == 'hsih':
                dram_qlen_hsih.append(avg_qlen)
            if bm == 'hdsh':
                dram_qlen_hdsh.append(avg_qlen)
            
    lp = myplot.LinePointPlot()
    myplot.plt.xlabel('突发大小（$\mathrm{MB}$）')
    myplot.plt.ylabel('片外缓存队列长度（$\mathrm{MB}$）')
    lp.plot(burst_size, dram_qlen_hdsh, label=hm_name['H-DSH'])
    lp.plot(burst_size, dram_qlen_hsih, label=hm_name['H-SIH'])
    save_fig = save_path + '.pdf'
    myplot.plt.savefig(save_fig) 
    
    # Motivation
    lp1 = myplot.LinePointPlot()
    lp1.ax.set_ylim([4, 40])
    lp1.ax.xaxis.set_major_locator(MultipleLocator(4))
    lp1.ax.xaxis.set_minor_locator(MultipleLocator(1))
    lp1.ax.yaxis.set_major_locator(MultipleLocator(5))
    lp1.ax.yaxis.set_minor_locator(MultipleLocator(1))      
    myplot.plt.xlabel('突发大小（$\mathrm{MB}$）')
    myplot.plt.ylabel('片外缓存队列长度（$\mathrm{MB}$）')
    lp1.plot(burst_size, dram_qlen_hsih) 
    save_fig = save_path + '-motivation.pdf'
    myplot.plt.savefig(save_fig)      
    
def plot_loss_rate(file_dir, save_path):
    bm_alg = ["hsih", "hdsh"]
    link_delay = [100 * i for i in range(1, 21)]
    
    hsih_loss = []
    hdsh_loss = []
    for bm in bm_alg:
        for delay in link_delay:
            file_name = f'bm-result-long-haul-transmission-{bm}-delay{delay}us.csv'
            file_path = os.path.join(file_dir, file_name)
            data = pd.read_csv(file_path)
            result = data['result']
            result = np.array(result)
            loss_count = np.sum(result == 2)
            loss_ratio = loss_count / len(result)
            if bm == 'hsih':
                hsih_loss.append(loss_ratio)
            if bm == 'hdsh':
                hdsh_loss.append(loss_ratio)
        
    lp = myplot.LinePointPlot()
    myplot.plt.xlabel('链路时延（$\mathrm{us}$）')
    myplot.plt.ylabel('丢包率')
    lp.plot(link_delay, hdsh_loss, label=hm_name['H-DSH'])
    lp.plot(link_delay, hsih_loss, label=hm_name['H-SIH'])
    save_fig = save_path + '.pdf'
    myplot.plt.savefig(save_fig)  
    
    # Motivation
    lp1 = myplot.LinePointPlot()
    myplot.plt.xlabel('链路时延（$\mathrm{us}$）')
    myplot.plt.ylabel('丢包率')
    lp1.plot(link_delay, hsih_loss)
    save_fig = save_path + '-motivation.pdf'
    myplot.plt.savefig(save_fig)            
                   
def plot_motivation_burst_absorption(file_dir, save_path):    
    dram_qlen_hsih = []

    file_name = f'buffer-usage-burst-absorption-hsih.csv'
    file_path = os.path.join(file_dir, file_name)
    data = pd.read_csv(file_path)
    dram = data['dram'] / MB
    dram = np.array(dram)
    
    cp = myplot.CDFPlot()
    cp.plot(dram)
    
    myplot.plt.xlabel('片外缓存队列长度（$\mathrm{MB}$）')
    myplot.plt.savefig(save_path)   
    plt.show()   
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        basedir = sys.argv[1]
    else:
        basedir = os.path.join('F:/', 'research', 'HDSH')
    basedir = os.path.expanduser(basedir)
    save_dir = os.path.join(basedir, 'figs/')
    data_dir = os.path.join(basedir, 'data/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # file_dir = "/home/yqliu/hb-logs"
    # data_dir = "/home/yqliu/git-workspace/hybrid-buffer-pfc/ns-3-dev/" + \
    #            "examples/hybrid-buffer/tests/data/"
    # save_dir = "/home/yqliu/git-workspace/hybrid-buffer-pfc/ns-3-dev/" + \
    #            "examples/hybrid-buffer/tests/figs/"

    ## --------------------FCT Evaluation---------------------
    fct_fig = "fct-evaluation.pdf"
    save_path = save_dir + fct_fig
    data_path = data_dir + "fct-evaluation"
    plot_fct_slow_down(data_path, save_path)
    
    # ## --------------------PFC Avoidance----------------------
    # incast_fig = "pfc-avoidance-incast.pdf"
    # save_path = save_dir + incast_fig
    # data_path = data_dir + "pfc-avoidance-incast"
    # plot_pfc_avoidance_incast(data_path, save_path)
    
    # web_fig = "pfc-avoidance-web.pdf"
    # save_path = save_dir + web_fig
    # data_path = data_dir + "pfc-avoidance-back"
    # plot_pfc_avoidance_web(data_path, save_path)
    
    # ## --------------------Collateral Damage-------------------
    # tput_fig = "collateral-damage"
    # save_path = save_dir + tput_fig
    # data_path = data_dir + "collateral-damage"
    # plot_collateral_damage_throughput(data_path, save_path)
    
    # ## --------------------Traffic Pattern---------------------
    # cdf_dir = basedir + '/flow-pattern/'
    # cdf_fig = "traffic-pattern-cdf.pdf"
    # save_path = save_dir + cdf_fig
    # plot_flow_pattern_cdf(cdf_dir, save_path) 
    
    # pattern_fig = "fct-evaluation-pattern"
    # save_path = save_dir + pattern_fig
    # data_path = data_dir + "fct-evaluation-pattern"
    # plot_fct_evaluation_pattern(data_path, save_path)
    
    # ## --------------------Congestion Control-------------------
    # cc_fig = "fct-evaluation-cc"
    # save_path = save_dir + cc_fig
    # data_path = data_dir + "fct-evaluation-cc"
    # plot_fct_evaluation_cc(data_path, save_path) 
    
    ## --------------------Burst Absorption---------------------
    burst_fig = "burst-absorption"
    save_path = save_dir + burst_fig
    data_path = data_dir + "burst-absorption"
    plot_burst_absorption(data_path, save_path)
    
    # ## ------------------Long-haul Transmission-----------------
    # long_haul_fig = "long-haul-transmission"
    # save_path = save_dir + long_haul_fig
    # data_path = data_dir + "long-haul-transmission"
    # plot_loss_rate(data_path, save_path)
    
    # ## ---------------Motivation Burst Absorption---------------
    # motivation_burst_fig = "motivation-burst-absorption.pdf"
    # save_path = save_dir + motivation_burst_fig
    # data_path = data_dir + "motivation-burst-absorption"
    # plot_motivation_burst_absorption(data_path, save_path)
        