
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
class plotter:
    def __init__(self, n_historical = 20):
        self.input_data = []
        self.n_hist = n_historical

        self.control_group_data = []
        for prop in range(1,4):
            for run in range(1,11):
                self.control_group_data.append(self.process_file("validation/output_{0}c{1}.log".format(prop, run)))

        t0 = self.control_group_data[0][0][0]
        for i in range(1,len(self.control_group_data)):
            offset_t0 = self.control_group_data[i][0][0] - t0
            for j in range(len(self.control_group_data[i][0])):
                self.control_group_data[i][0][j] -= offset_t0

        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4)
        self.fig.set_dpi(200)
        self.process_input_file_rnn(n_historical = n_historical)

    def process_input_file_hnn(self, filename="validation/input.in", n_historical = 20):
        for line in open(filename, 'r').readlines():
            input_counter = 0
            for data in line.split(','):
                x = float(data) # x[0] = u
                for _ in range(self.n_hist-input_counter):
                    x += [0,0,0,0,0]
                for i in range(min(input_counter, n_historical)):
                    m_n_minus_i = []
                    m_n_minus_i.append(self.control_group_data[0][1][(input_counter * 5) - i - 1])#DT
                    m_n_minus_i.append(self.control_group_data[0][2][(input_counter * 5) - i - 1])#RPM
                    m_n_minus_i.append(self.control_group_data[0][3][(input_counter * 5) - i - 1])#V
                    m_n_minus_i.append(self.control_group_data[0][4][(input_counter * 5) - i - 1])#C
                    m_n_minus_i.append(self.control_group_data[0][6][(input_counter * 5) - i - 1])#TEMP
                    x += m_n_minus_i

                input_counter = input_counter +1
                self.input_data.append(x)

    def process_input_file_rnn(self, filename="validation/input.in", n_historical = 20):
        for line in open(filename, 'r').readlines():
            input_counter = 0
            prev_data = 0
            for data in line.split(','):
                x = []
                data_f = float(data)
                for _ in range(self.n_hist-input_counter):
                    x.append([0,0,0,0,0,0])
                for i in range(min(input_counter, n_historical)):
                    m_n_minus_i = []
                    m_n_minus_i.append(float(prev_data + (data_f - prev_data) * float(i) / float(n_historical)  ))#Throttel
                    m_n_minus_i.append(self.control_group_data[0][1][((input_counter * n_historical) - n_historical) + i])#DT
                    m_n_minus_i.append(self.control_group_data[0][2][((input_counter * n_historical) - n_historical) + i])#RPM
                    m_n_minus_i.append(self.control_group_data[0][3][((input_counter * n_historical) - n_historical) + i])#V
                    m_n_minus_i.append(self.control_group_data[0][4][((input_counter * n_historical) - n_historical) + i])#C
                    m_n_minus_i.append(self.control_group_data[0][6][((input_counter * n_historical) - n_historical) + i])#TEMP
                    x.append(m_n_minus_i)

                input_counter = input_counter +1
                self.input_data.append(x)
                prev_data = data_f

        

    def process_file(self, filename):
        T = []
        DT = []
        RPM = []
        C = []
        V = []
        P = []
        TEMP = []
        THROTTLE = []
        THROTTLE_T = []

        new_time = False

        read_messages = 100000
        count = 0
        for line in open(filename, 'r').readlines():
            l = line.strip()
            if "ts_real=" in l:
                if count == read_messages:
                    break
                else:
                    count += 1
                T.append(float(l.split("ts_mono=")[1].split("  ts_real")[0]))
                if len(T) > 1:
                    DT.append(T[-1] - T[-2])
                new_time = True
            if "rpm:" in l:
                RPM.append(float(l.split("rpm:")[1].strip()))
            if "voltage:" in l:
                V.append(float(l.split("voltage:")[1].strip()))
            if "current:" in l:
                C.append(float(l.split("current:")[1].strip()))
                P.append(V[-1]*C[-1])
            if "temperature:" in l:
                TEMP.append(float(l.split("temperature:")[1].strip()))
            if "commands:[" in l and new_time:
                THROTTLE.append(float(l.split("commands:[")[1].split(',')[0].strip()))
                THROTTLE_T.append(T[-1])
                new_time = False

        return (T, DT, RPM, V, C, P, TEMP, THROTTLE, THROTTLE_T)

    def plot_from_file(self):        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax1.set_ylim(0,11000)
        self.ax2.set_ylim(14.5,17)
        self.ax3.set_ylim(0,12)
        self.ax4.set_ylim(0,200)
        self.fig.suptitle('Axes values are scaled individually by default')
        for dataset in self.control_group_data:
            self.ax1.plot(dataset[0], dataset[2], color='b', marker='.', linestyle=' ', alpha=0.5)
            self.ax2.plot(dataset[0], dataset[3], color='b', marker='.', linestyle=' ', alpha=0.5)
            self.ax3.plot(dataset[0], dataset[4], color='b', marker='.', linestyle=' ', alpha=0.5)
            self.ax4.plot(dataset[0], dataset[5], color='b', marker='.', linestyle=' ', alpha=0.5)
        self.ax1.set_title('RPM')
        self.ax2.set_title('V')
        self.ax3.set_title('C')
        self.ax4.set_title('P')
        # plt.legend()
        # plt.draw()
        # plt.pause(0.001)

    def plot_dataset(self, prediction_output, skip_first = 30):
        T = []
        for i in range(0, len(self.control_group_data[0][0]), 5):
            T.append(self.control_group_data[0][0][i])
        RPM = []
        V = []
        C = []
        P = []
        for data in prediction_output:
            RPM.append(data[0])
            V.append(data[1])
            C.append(data[2])
            P.append(data[1] * data[2])
        self.ax1.plot(T[skip_first-1:-1], RPM[skip_first:], color='r')
        self.ax2.plot(T[skip_first-1:-1], V[skip_first:], color='r')
        self.ax3.plot(T[skip_first-1:-1], C[skip_first:], color='r')
        self.ax4.plot(T[skip_first-1:-1], P[skip_first:], color='r')

        plt.savefig('figures/{0}.png'.format(datetime.now().strftime('%Y%m%d%H%M%S')))