import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
import json
from matplotlib.axis import Axis
import matplotlib.patches as mpatches
from tkinter import messagebox
from tkinter.filedialog import askopenfilename  # Dateidialog
from tkinter import filedialog,simpledialog,messagebox,colorchooser
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import warnings

plt.style.use('seaborn')
# plt.style.use('seaborn-colorblind') #alternative
# plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 100
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from arch import arch_model

import yfinance as yf


"""##################### Using the yfinance Library to Extract Stock Data######################################

Using the `Ticker` module we can create an object that will allow us to access functions to extract data. 
To do this we need to provide the ticker symbol for the stock, here the company is IBM and the ticker 
symbol is `IBM`.

It’s completely free and super easy to setup- a single line to install the library:

    pip install yfinance --upgrade --no-cache-dir

"""

# import matplotlib.cm as cm # matplotlib colormaps

def dialog():
    global mystr
    # show Input Dialog
    title = "Ticket"
    prompt = "Enter String"
    mystr = simpledialog.askstring(title, prompt)


def myplot(y):
    print(y)
#    plt.ion() # Fenster ist nicht modal / plt.ioff() ist modal
    # Explore the structure of the data.
    # name = r'eq_data_1_day_m1.json'
    with open(name) as f:
        all_eq_data = json.load(f)

    all_eq_dicts = all_eq_data['features']  # enthält den key "Features", der die interesssierenden Daten enthält
    #    print(type(all_eq_dicts[0]))  # Alle Daten aus "Features" sind jetzt in all_eq_dicts.
    #    print(all_eq_dicts[0].keys())
    #    print(all_eq_dicts[0]['geometry']['coordinates'][0])
    #
    mags, plas, lons, lats = [], [], [], []
    for eq_dict in all_eq_dicts:  # all_eq_dicts enthält Liste. Diese kann durchlaufen werden.
        mag = eq_dict['properties']['mag']  #
        pla = eq_dict['properties']['place']  #
        lon = eq_dict['geometry']['coordinates'][0]  # Ausgabe durch Angabe der passenden Keys und "subkeys".
        lat = eq_dict['geometry']['coordinates'][1]
        mags.append(mag)
        plas.append(pla)
        lons.append(lon)
        lats.append(lat)
    #
    smags = [entry * 10 for entry in mags]  # Multiply a List with a constant factor
    # smagm = ["x" if 0<entry<4 else "o" for entry in mags]  # Multiply a List with a constant factor
    # print(smagm)
    fig, ax = plt.subplots()
    # for lo, la, si, sm in zip(lons,lats,smags, smagm):
    #    scatter = ax.scatter(lo,la,s = si, marker = sm)

    sc = scatter = ax.scatter(lons, lats, marker="o", c=mags, s=smags, cmap='plasma_r',
                              label="eqs mag")  # c= CN Farbzyklus ('C0-C9'), colormap 'viridis' or 'viridis_r' for reversed colormap
    # colormap tutorial: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    for i, mag in enumerate(mags):
        if mag >= 4:  # Nur Beschriftung fuer Magnitude ueber 4
            plt.annotate(mag, xy=(lons[i], lats[i]), ha="left", va="bottom", textcoords="offset pixels", xytext=(4, 4),
                         zorder=i, fontsize=8)
            # textcoords = "offset points": Koordinatensystem für xytext, xytext= (4,4) ist der offset in pixel
            # Bedeutung der Parameter: ha = "horizontal Alignment", va = "vertical Alignment",
    cb = fig.colorbar(scatter,
                      ax=ax)  # list of colormaps: https://matplotlib.org/examples/color/colormaps_reference.html
    # cb.set_edgecolor("face") # funktioniert manchmal
    # cb.solids.set_rasterized(True) # nicht immer # legend_elements

    # red_dot = mpatches.Patch(color = "red", label = "eqs magnitude")
    # ax.legend(handles = [red_dot])
    # ax.legend()

    ax.grid()
    plt.minorticks_on()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # xticks = np.arange(min(lons), max(lons), 45)  # xticks erzeugen mit numpy "arange" - Funktion, Schrittweite 45
    # ax.set_xticks(xticks)

    plt.show() # block = False/True schaltet modales Verhalten aus /ein

    res = messagebox.askyesno(title= "New Ticker download ?") # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

def callback():

    global name
    name = askopenfilename(filetypes=[("json", "*.json"), ("PNG", "*.png"), ("All Files", "*.*")])  # filetypes enthält Filter für Dateitypen
    b1.config(state='normal')

############################## Log returns vs Monthly realized volatility #########################################

def volat_retn():

    df = yf.download({mystr},
                     start='2000-01-01',
                     end='2021-12-31',
                     auto_adjust=False,
                     progress=False)

    # keep only the adjusted close price

    df = df.loc[:, ['Adj Close']]
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

    # calculate simple returns
    df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

    # remove redundant data
    df.drop('adj_close', axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)

    print(df.head)

    def realized_volatility(x):

        return np.sqrt(np.sum(x ** 2))

    df_rv = df.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
    df_rv.rename(columns={'log_rtn': 'rv'}, inplace=True)

    df_rv.rv = df_rv.rv * np.sqrt(12)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(df)
    ax[0].set(title='Log returns', ylabel='Log returns (%)')

    ax[1].plot(df_rv)
    ax[1].set(title='Monthly realized volatility', ylabel='Monthly volatility')

    plt.show()

    res = messagebox.askyesno(title="New Ticker download ?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

############################## Log returns vs Monthly realized volatility #########################################


#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Autoregressive Conditional Heteroskedasticity (ARCH) and other tools for financial econometrics,
# written in Python (with Cython and/or Numba used to improve performance)
#
# ARCH-Modelle (ARCH, Akronym für: AutoRegressive Conditional Heteroscedasticity, deutsch autoregressive
# bedingte Heteroskedastizität) bzw. autoregressive bedingt heteroskedastische Zeitreihenmodelle sind
# stochastische Modelle zur Zeitreihenanalyse, mit deren Hilfe insbesondere finanzmathematische Zeitreihen
# mit nicht konstanter Volatilität beschrieben werden können. Sie gehen von der Annahme aus, dass die bedingte
# Varianz der zufälligen Modellfehler abhängig ist vom realisierten Zufallsfehler der Vorperiode, so dass
# große und kleine Fehler dazu tendieren, in Gruppen aufzutreten. ARCH-Modelle wurden von Robert F. Engle
# in den 1980er Jahren entwickelt. Im Jahr 2003 wurde ihm dafür der Nobelpreis für Wirtschaftswissenschaften verliehen.

# Die Idee des ARCH-Modells wurde in verschiedener Weise weiterentwickelt und gehört heute ganz selbstverständlich
# zu den fortgeschrittenen Methoden der Ökonometrie.

# Eine Verallgemeinerung sind die GARCH-Modelle (generalized autoregressive conditional heteroscedasticity),
# die 1986 von Tim Bollerslev entwickelt wurden. Hierbei hängt die bedingte Varianz nicht nur von der Historie der
# Zeitreihe ab, sondern auch von ihrer eigenen Vergangenheit. Zeitstetige Analoga, sogenannte
# COGARCH-Modelle (continuous-time GARCH), wurden von Feike C. Drost und Bas J. C. Werker sowie Claudia Klüppelberg,
# Alexander Lindner und Ross Maller vorgestellt.

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def risky_assets(): #################### Risky Assets #############################################################
    global mystr
    RISKY_ASSETS = ['GOOG', 'MSFT', 'AAPL', 'IBM']
    N = len(RISKY_ASSETS)
    START_DATE = '2000-01-01'
    END_DATE = '2021-12-30'

    df4 = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE, adjusted=True)
    returns = 500 * df4['Adj Close'].pct_change().dropna()
    returns.plot(subplots=True, title=f'CCC-GARCH model for multivariate volatility forecasting: {START_DATE} - {END_DATE} ');

    coeffs = []
    cond_vol = []
    std_resids = []
    models = []

    for asset in returns.columns:
        model = arch_model(returns[asset], mean='Constant',vol='GARCH', p=1, o=0,q=1).fit(update_freq=0, disp='off')
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)

    coeffs_df = pd.DataFrame(coeffs, index=returns.columns)
    cond_vol_df = pd.DataFrame(cond_vol).transpose().set_axis(returns.columns, axis='columns',inplace=False)
    std_resids_df = pd.DataFrame(std_resids).transpose().set_axis(returns.columns,axis='columns',inplace=False)

    coeffs_df

    R = std_resids_df.transpose().dot(std_resids_df).div(len(std_resids_df))

    # define objects
    diag = []
    D = np.zeros((N, N))

    # populate the list with conditional variances
    for model in models:
        diag.append(model.forecast(horizon=1).variance.values[-1][0])
    # take the square root to obtain volatility from variance
    diag = np.sqrt(np.array(diag))
    # fill the diagonal of D with values from diag
    np.fill_diagonal(D, diag)

    # calculate the conditional covariance matrix
    H = np.matmul(np.matmul(D, R.values), D)

    print(H)

    plt.show()

    res = messagebox.askyesno(title="New Ticker download ?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

################################# End : Risky Assets  ##########################

def dividens(): #################### Dividens #############################################################

    import yfinance as yf
    global mystr
    a = yf.Ticker("IBM")

    # a = yf.download({mystr},
    #                  start='2000-01-01',
    #                  end='2021-12-31',
    #                  auto_adjust=False,
    #                  progress=False)

    a.dividends.plot(title=f' {mystr} Dividens 1960 - 2021 ')  # Dividense

    plt.show()

    res = messagebox.askyesno(title="New Ticker download ?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()
#################### End Dividens #############################################################


def max_exchange(): ###### Max Exchange rate from 1960 - 2021 #################################

    import yfinance as yf
    global mystr

    ibm = yf.Ticker("IBM")
    df1 = ibm.history( period="max")
    print(df1)
    ax = df1.plot(title=f' {mystr} Max Exchange rate from 1960 - 2021 ')
    plt.show()

    res = messagebox.askyesno(title="New Ticker download ?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()
#####################   Max Exchange rate from 1960 - 2021 #################################

def cash_flow():################ Cashflow ###################################################

    #import yfinance as yf
    #global mystr

    ibm = yf.Ticker("IBM")
    df2 = ibm.cashflow.plot()
    print(df2)
    ax2 = df2.plot()
    #plt.show(title=f' {mystr} Cash Flow ')
    plt.show()

    res = messagebox.askyesno(title="New Ticker download ?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

def last_week(): ############ Open last week #####################################################

    #import yfinance as yf
    #global mystr

    ibm = yf.Ticker("IBM")

    df3= ibm.history(titel="IBM", interval="1wk")
    print(df3)
    ax3 = df3.plot()
    #plt.show(title=f' {mystr} Last Week Exchange Rate ')
    plt.show()

    res = messagebox.askyesno(title="New Ticker download ?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

####################################################################################################

# download data as pandas DataFrame
# IBM time series 1960 - 2021 :  Stock price - Simple returns - Log returns
def stock_simple_log():
    df = yf.download('IBM', auto_adjust = False, progress=False)
    df = df.loc[:, ['Adj Close']]
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

# create simple and log returns
    df['simple_rtn'] = df.adj_close.pct_change()
    df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

# dropping NA's in the first row
    df.dropna(how = 'any', inplace = True)

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# add prices
    df.adj_close.plot(ax=ax[0])
    ax[0].set(title='IBM time series 1960 - 2021   Stock price - Simple returns - Log returns',
          ylabel='Stock price ($)')

# add simple returns
    df.simple_rtn.plot(ax=ax[1])
    ax[1].set(ylabel='Simple returns (%)')

# add log returns
    df.log_rtn.plot(ax=ax[2])
    ax[2].set(xlabel='Date',
          ylabel='Log returns (%)')

    ax[2].tick_params(axis='x',
                  which='major',
                  labelsize=12)

    plt.tight_layout()
    plt.savefig('ibm_Stock_Returns.png')
    plt.show()

def vix_index(): #####################################################################################
## Download and preprocess the prices of IBM and VIX
# CBOE Volatility Index (^VIX) vs IBM
"""
Wir haben zusätzlich den Korrelationskoeffizienten zwischen den beiden Reihen berechnet und in den Titel aufgenommen.
Es zeigt sich, dass sowohl die negative Steigung der Regressionsgeraden als auch eine starke negative Korrelation
zwischen den beiden Reihen die Existenz des Leverage-Effekts in den Renditereihen bestätigen.

"""

    df = yf.download(['IBM', '^VIX'],
                 start='1980-01-01',
                 end='2021-12-31',
                 progress=False)
    df = df[['Adj Close']]
    df.columns = df.columns.droplevel(0)
    df = df.rename(columns={'IBM': 'ibm', '^VIX': 'vix'})

    df['log_rtn'] = np.log(df.ibm / df.ibm.shift(1))
    df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
    df.dropna(how='any', axis=0, inplace=True)

    corr_coeff = df.log_rtn.corr(df.vol_rtn)

    ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, line_kws={'color': 'red'})
    ax.set(title=f'IBM Volatility vs. CBOE Volatility Index (VIX)   1980 - 2021  ($\\rho$ = {corr_coeff:.2f})',
        ylabel='VIX log returns',
        xlabel='IBM log returns')

    plt.tight_layout()
    plt.savefig('VIX_vs_IBM.png')
    plt.show()
#####################################################################################################

root = tk.Tk()
mystr = tk.StringVar()

#global mystr
global name
name = " open file "

x = 12
y = 20

l1 = tk.Label(root, text="Stock Exchange Alalysis")
l1.grid(row=0, column=0, sticky=tk.E + tk.W, ipadx=40)  # Ohne ipadx = 40 wird ein 2-spaltiges Gitter erzeugt.

b1 = tk.Button(root, text="Plot", state='disabled', command=lambda: myplot(x) if x < 11 else myplot(y))
b1.grid(row=1, column=0, sticky=tk.E + tk.W)

b3 = tk.Button(root, text='Open File', command=callback)  # command = enthält den auszuführenden Befehl
b3.grid(row=2, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b4 = tk.Button(root, text='Ticker', command=dialog)  # command = enthält den auszuführenden Befehl
b4.grid(row=3, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5 = tk.Button(root, text='History Max', command=max_exchange)  # command = enthält den auszuführenden Befehl
b5.grid(row=4, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5e = tk.Button(root, text='Open Last Week', command=last_week)  # command = enthält den auszuführenden Befehl
b5e.grid(row=5, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5d = tk.Button(root, text='Cash Flow', command=cash_flow)  # command = enthält den auszuführenden Befehl
b5d.grid(row=6, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5c = tk.Button(root, text='Volatility', command=volat_retn)  # command = enthält den auszuführenden Befehl
b5c.grid(row=7, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5b = tk.Button(root, text='Dividens', command=dividens)  # command = enthält den auszuführenden Befehl
b5b.grid(row=8, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5a = tk.Button(root, text='Risky Assets', command=risky_assets)  # command = enthält den auszuführenden Befehl
b5a.grid(row=9, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b6 = tk.Button(root, text="Quit", command=root.quit)
b6.grid(row=10, column=0, sticky=tk.E + tk.W)

root.mainloop()

