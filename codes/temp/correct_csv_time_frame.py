import pandas as pd

filenames = ['GBPUSD1.csv', 'EURGBP1.csv', 'EURUSD1.csv', 'GBPJPY1.csv', 'USDJPY1.csv']

pd_data = []
for filename in filenames:
    pd_data.append(pd.read_csv('../../data/' + filename, parse_dates=['Time', 'Date']))

res = pd_data[0]
for _pd_data in pd_data[1:]:
    res = pd.merge(res, _pd_data, on=['Date', 'Time'], how='inner')

res.to_csv('all_currency.csv')
res.to_excel('all_currency.xlsx')
# res[20000:40000].to_csv('all_currency_2.csv')
# res[20000:40000].to_excel('all_currency_2.xlsx')
# res[40000:60000].to_csv('all_currency_3.csv')
# res[40000:60000].to_excel('all_currency_3.xlsx')
# res[60000:].to_csv('all_currency_4.csv')
# res[60000:].to_excel('all_currency_4.xlsx')
print(res)