#!/usr/bin/env python

import os, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from convert_time import convert_time
from datetime import datetime, timedelta
import sys
        
def funct(x, a, b):
    # f = a*np.exp(b*x)
    f = a*2**(b*x)
    return f        

class update_covid:
    def __init__(self, remove):
        self.filename = datetime.now().strftime('covid_daily_%Y%m%d.csv')
        self.data = {}
        self.state = 'MD'

        self.get_data()
        self.read_data()
        self.fit_data(remove=remove)
        self.calc_ratio()
        self.plot_data()

    def get_data(self):
        if not os.path.isfile(self.filename):
            com = f'curl -o {self.filename} https://covidtracking.com/api/states/daily.csv'
            subprocess.call(com, shell=True)
            print(f' + downloaded {self.filename}')

    def read_data(self):
        df = pd.read_csv(self.filename)
        md = df[df['state'] == self.state]
        print(md)
        dates = np.flip(np.array(md.date)) # read dates in YYYYMMDD format
        dates = convert_time(dates) # convert to datetime
        nday = np.arange(len(md)) # dates as integer array
        self.data['nday'] = nday
        self.data['dates'] = dates
        self.data['positive'] = np.flip(np.array(md.positive))
        self.data['deaths'] = np.flip(np.array(md.death))

    def fit_data(self, remove=0):
        x = self.data['nday']
        y = self.data['deaths']
        good = np.isfinite(y)
        x = x[good]
        y = y[good]
        x = x[:len(x)-remove]
        y = y[:len(y)-remove]
        popt, pcov = curve_fit(funct, x, y, p0=[1,1])
        self.popt = popt

    def calc_ratio(self):
        pos = self.data['positive']
        deaths = self.data['deaths']
        good = np.where(np.isfinite(deaths) == True)
        ratio = 1/(deaths[good]/pos[good])
        return np.median(ratio)

    def extrapolate_fit(self):
        days = np.arange(120)
        pfit = funct(days, self.popt[0], self.popt[1])
        match = np.where(pfit <= 1.0E+4)
        days = days[match]
        pfit = pfit[match]

        deaths = self.data['deaths']

        ratio = self.calc_ratio()

        date0 = self.data['dates'][0]
        for day in days:
            n = int(day)
            d = (date0 + timedelta(days=n)).strftime('%d-%b-%Y')
            est_deaths = pfit[n]            
            est_pos = pfit[n]*ratio

            if n < len(deaths):
                num = deaths[n]
                if np.isnan(num): num = 0
                act_deaths = f'{num:,.0f}'
                act_pos = self.data['positive'][n]
                act_pos = f'{act_pos:10,.0f}'
            else:
                act_deaths = '---'
                act_pos = '---'
            print(f'{n:3d}{d:>15}{act_pos:>10}{est_pos:10,.0f}'\
                  f'{est_deaths:10,.0f}{act_deaths:>10}')
        
        return days, pfit

    def plot_data(self):
        t, pfit = self.extrapolate_fit()
        ratio = self.calc_ratio()

        period = 1/self.popt[1]
        title = f'COVID Data for {self.state} | Doubling Period = {period:0.1f} days'\
            f' | Death Ratio = {ratio:0.3f}'

        plt.semilogy(self.data['nday'], self.data['positive'], 'o', color='C0', \
                     label='Actual Positive')
        plt.semilogy(t, pfit*ratio, ':', label='Modeled Positive', color='C0')
        plt.semilogy(self.data['nday'], self.data['deaths'], 'or', label='Actual Deaths')
        plt.semilogy(t, pfit, ':r', label='Modeled Deaths')
        plt.xlabel('Days Since March 4, 2020')
        plt.ylabel('Cummulative Positive Cases')
        plt.title(title, fontsize=10)
        plt.ylim((1,1.0E+6))
        plt.legend()
        plt.savefig('plot_covid.png')
        plt.show()


if __name__ == '__main__':

    if len(sys.argv) > 1:
        remove = int(sys.argv[1])
    else:
        remove = 0

    o = update_covid(remove=remove)
