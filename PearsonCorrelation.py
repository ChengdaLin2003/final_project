
import pandas as pd
from scipy.stats import pearsonr

# Load the data
layoffs = pd.read_excel('cleaned_layoffs.xlsx')
GDPdata = pd.read_excel('cleaned_GDP.xlsx')
stockData = pd.read_excel('cleaned_stock_market.xlsx')

# Prepare layoffs data
layoffs['Date_layoffs'] = pd.to_datetime(layoffs['Date_layoffs'])
layoffs.set_index('Date_layoffs', inplace=True)
df_layoffs_quarterly = layoffs['Laid_Off'].resample('Q').sum()

# Prepare GDP data
df_gdp_us = GDPdata[GDPdata['Country Code'] == 'USA']
gdp_time_columns = df_gdp_us.columns[df_gdp_us.columns.str.contains(r'\[\d{4}Q\d\]')]
df_gdp_us_quarterly = df_gdp_us[gdp_time_columns].transpose()
df_gdp_us_quarterly.index = pd.to_datetime(df_gdp_us_quarterly.index.str.extract(r'(\d{4}Q\d)')[0])
df_gdp_us_quarterly.index = df_gdp_us_quarterly.index.to_period('Q').to_timestamp('Q')
df_gdp_us_quarterly.columns = ['GDP']
df_gdp_us_quarterly = df_gdp_us_quarterly.sort_index()

# Prepare stock market data
df_stock_market_cleaned = stockData.dropna(subset=['Country'])
df_stock_market_us = df_stock_market_cleaned[df_stock_market_cleaned['Country'].str.contains('United States')]
stock_time_columns = df_stock_market_us.columns[df_stock_market_us.columns.str.match(r'\d{4}Q\d')]
df_stock_market_us_quarterly = df_stock_market_us[stock_time_columns].transpose()
df_stock_market_us_quarterly.index = pd.to_datetime(df_stock_market_us_quarterly.index).to_period('Q').to_timestamp('Q')
df_stock_market_us_quarterly.columns = ['Stock Market']

# Merge data
merged_data = pd.concat([df_layoffs_quarterly, df_gdp_us_quarterly, df_stock_market_us_quarterly], axis=1).dropna()

# Calculate Pearson correlation coefficients
pearson_layoffs_gdp = pearsonr(merged_data['Laid_Off'], merged_data['GDP'])
pearson_layoffs_stock_market = pearsonr(merged_data['Laid_Off'], merged_data['Stock Market'])
pearson_gdp_stock_market = pearsonr(merged_data['GDP'], merged_data['Stock Market'])

print('Pearson correlation (Layoffs and GDP):', pearson_layoffs_gdp)
print('Pearson correlation (Layoffs and Stock Market):', pearson_layoffs_stock_market)
print('Pearson correlation (GDP and Stock Market):', pearson_gdp_stock_market)
