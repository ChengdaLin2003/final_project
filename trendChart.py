import pandas as pd
import matplotlib.pyplot as plt

# Read the final merged dataset
layoffs = pd.read_excel('tech_layoffs.xlsx')
GDPdata = pd.read_excel('GDP.xlsx')
stockData = pd.read_excel('Stock market.xlsx')

### Quarterly Layoffs Trend ###
# Ensure that 'Date_layoffs' is a datetime type
layoffs['Date_layoffs'] = pd.to_datetime(layoffs['Date_layoffs'])
# Group by Month or Quarter to count the number of layoffs
layoffs_by_quarter = layoffs.groupby(layoffs['Date_layoffs'].dt.to_period('Q'))['Laid_Off'].sum()
# Plotting the trend by Quarter
layoffs_by_quarter.plot(kind='line', figsize=(12, 6))
plt.title('Quarterly Layoffs Trend')
plt.xlabel('Quarter')
plt.ylabel('Number of Layoffs')
plt.grid(True)
plt.show()


### Top 10 Countries by Number of Layoffs ###
# Group by country and sum the layoffs
layoffs_by_country = layoffs.groupby('Country')['Laid_Off'].sum().sort_values(ascending=False)
# Plot the top countries with the highest number of layoffs
layoffs_by_country.head(10).plot(kind='bar', figsize=(12, 6))
plt.title('Top 10 Countries by Number of Layoffs')
plt.xlabel('Country')
plt.ylabel('Number of Layoffs')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


### GDP Quarterly Trends ###
# Filter for the relevant countries
countries = ['United States', 'India', 'Germany']
gdp_data = GDPdata[GDPdata['Country'].isin(countries)]
# Specify the order of columns (quarters)
columns_order = [
    '2020Q1 [2020Q1]', '2020Q2 [2020Q2]', '2020Q3 [2020Q3]', '2020Q4 [2020Q4]',
    '2021Q1 [2021Q1]', '2021Q2 [2021Q2]', '2021Q3 [2021Q3]', '2021Q4 [2021Q4]',
    '2022Q1 [2022Q1]', '2022Q2 [2022Q2]', '2022Q3 [2022Q3]', '2022Q4 [2022Q4]',
    '2023Q1 [2023Q1]', '2023Q2 [2023Q2]', '2023Q3 [2023Q3]', '2023Q4 [2023Q4]'
]
gdp_data = gdp_data[['Country'] + columns_order]
# Convert data to numeric and replace missing values manually
for index, row in gdp_data.iterrows():
    for i in range(1, len(columns_order)):
        if pd.isna(row[columns_order[i]]):
            gdp_data.at[index, columns_order[i]] = gdp_data.at[index, columns_order[i-1]]
# Convert the data for plotting to floats, ensuring all values are numeric
plot_data_numeric = gdp_data.set_index('Country').T.apply(pd.to_numeric, errors='coerce').ffill()
# Plotting
plt.figure(figsize=(14, 8))
for country in plot_data_numeric.columns:
    plt.plot(plot_data_numeric.index, plot_data_numeric[country], marker='o', label=country)
plt.title('GDP Quarterly Trends (2020-2023)')
plt.xlabel('Quarters')
plt.ylabel('GDP (current US$, millions)')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.grid(True)
plt.tight_layout()
plt.show()


### Stock Market Quarterly Trends ###
# Filter for the relevant countries
stock_data_filtered = stockData[stockData['Country'].isin(countries)]
# Define the mapping from months to quarters
quarters_mapping = {
    'Q1': ['01', '02', '03'],
    'Q2': ['04', '05', '06'],
    'Q3': ['07', '08', '09'],
    'Q4': ['10', '11', '12']
}
# Initialize a DataFrame to hold the quarterly data
stock_data_quarterly_corrected = pd.DataFrame()
# Get the list of all months available in the dataset
available_months = [col for col in stock_data_filtered if col.startswith('202') and 'M' in col]
# For each country, calculate the quarterly average from monthly data
for country in countries:
    country_data = stock_data_filtered[stock_data_filtered['Country'] == country].iloc[0, 4:]  # Skip initial columns
    country_data = country_data.replace('..', pd.NA).astype(float)  # Replace '..' with NaN and convert to float
    # Quarterly data will be stored in a dictionary
    country_quarterly_data = {}
    # For each year-quarter, calculate the mean of available months
    for year in range(2020, 2024):
        for quarter, months in quarters_mapping.items():
            # Filter the available months for the current quarter and year
            month_keys = [f"{year}M{month} [{year}M{month}]" for month in months if f"{year}M{month} [{year}M{month}]" in available_months]
            # Calculate the mean, skipping NaN values
            country_quarterly_data[f'{year}Q{quarter}'] = pd.Series(country_data[month_keys]).mean(skipna=True)
    # Convert the quarterly data to a DataFrame
    country_quarterly_df = pd.DataFrame(country_quarterly_data, index=[country]).T
    # Combine the data for all countries
    stock_data_quarterly_corrected = pd.concat([stock_data_quarterly_corrected, country_quarterly_df], axis=1)
# Fill any missing quarterly values with the previous quarter's data
stock_data_quarterly_corrected.ffill(inplace=True)
# Plotting
plt.figure(figsize=(14, 8))
for country in stock_data_quarterly_corrected.columns:
    plt.plot(stock_data_quarterly_corrected.index, stock_data_quarterly_corrected[country], marker='o', label=country)
plt.title('Stock Market Quarterly Trends (2020-2023)')
plt.xlabel('Quarters')
plt.ylabel('Stock Market Index')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.grid(True)
plt.tight_layout()
plt.show()


### USA GDP and Layoffs ###
us_gdp_data = GDPdata[GDPdata['Country'] == 'United States'].iloc[:, 4:]
us_gdp_data = us_gdp_data.replace('..', pd.NA).fillna(method='ffill', axis=1)
gdp_long = pd.melt(us_gdp_data, var_name='Quarter', value_name='GDP')
gdp_long['Quarter'] = gdp_long['Quarter'].str.extract(r'(\d{4}Q\d)')[0]
gdp_long['Date'] = pd.PeriodIndex(gdp_long['Quarter'], freq='Q').to_timestamp()
# Stock Market and Layoffs
us_stock_market_data = stockData[stockData['Country'] == 'United States'].iloc[:, 4:]
us_stock_market_data = us_stock_market_data.replace('..', pd.NA).fillna(method='ffill', axis=1)
stock_market_long = pd.melt(us_stock_market_data, var_name='Month', value_name='Stock_Index')
stock_market_long['Month'] = stock_market_long['Month'].str.extract(r'(\d{4}M\d{2})')[0]
stock_market_long['Date'] = pd.to_datetime(stock_market_long['Month'], format='%YM%m').dt.to_period('M').dt.to_timestamp()
stock_market_long['Quarter'] = stock_market_long['Date'].dt.to_period('Q')
quarterly_stock_market = stock_market_long.groupby('Quarter').agg({'Stock_Index':'mean'}).reset_index()
quarterly_stock_market['Date'] = quarterly_stock_market['Quarter'].dt.to_timestamp()
# Load layoffs data
us_layoffs_data = layoffs[layoffs['Country'] == 'USA']
us_layoffs_data['Date'] = pd.to_datetime(us_layoffs_data['Date_layoffs'])
us_layoffs_data['Quarter'] = us_layoffs_data['Date'].dt.to_period('Q')
quarterly_layoffs = us_layoffs_data.groupby('Quarter').agg({'Laid_Off':'sum'}).reset_index()
quarterly_layoffs['Date'] = quarterly_layoffs['Quarter'].dt.to_timestamp()
# Merge all data
merged_data = pd.merge(gdp_long, quarterly_layoffs, on='Date', how='outer')
merged_data = pd.merge(merged_data, quarterly_stock_market, on='Date', how='outer')
merged_data['GDP'] = merged_data['GDP'].astype(float) / 1000
merged_data['Stock_Index'] = merged_data['Stock_Index'].astype(float)
merged_data['Laid_Off'] = merged_data['Laid_Off'].fillna(0)
filtered_data = merged_data[(merged_data['Date'] >= '2020-01-01') & (merged_data['Date'] <= '2023-12-31')]
# Create quarterly dates for x-ticks
quarter_dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='QS')
quarter_labels = [f'{date.year}-Q{((date.month-1)//3)+1}' for date in quarter_dates]
# Plotting
# Chart 1: GDP and Layoffs
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(filtered_data['Date'], filtered_data['GDP'], color='blue', label='GDP (Billions USD)')
ax1.set_xlabel('Date')
ax1.set_ylabel('GDP (Billions USD)', color='blue')
ax1.set_title('U.S. GDP and Tech Industry Layoffs (2020-2023)')
ax1.set_xticks(quarter_dates)
ax1.set_xticklabels(quarter_labels, rotation=45, ha='right')
ax2 = ax1.twinx()
ax2.bar(filtered_data['Date'], filtered_data['Laid_Off'], color='red', label='Layoffs', width=20)
ax2.set_ylabel('Number of Layoffs', color='red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
# Chart 2: Stock Market and Layoffs
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(filtered_data['Date'], filtered_data['Stock_Index'], color='green', label='Stock Index')
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Index', color='green')
ax1.set_title('U.S. Stock Market Indexes and Technology Industry Layoffs (2020-2023)')
ax1.set_xticks(quarter_dates)
ax1.set_xticklabels(quarter_labels, rotation=45, ha='right')
ax2 = ax1.twinx()
ax2.bar(filtered_data['Date'], filtered_data['Laid_Off'], color='red', label='Layoffs', width=20)
ax2.set_ylabel('Number of Layoffs', color='red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()


### Germany GDP and Layoffs ###
gdp_data = pd.read_excel("GDP.xlsx")
germany_gdp_data = gdp_data[gdp_data['Country'] == 'Germany'].iloc[:, 4:]
germany_gdp_data = germany_gdp_data.replace('..', pd.NA).fillna(method='ffill', axis=1)
gdp_long = pd.melt(germany_gdp_data, var_name='Quarter', value_name='GDP')
gdp_long['Quarter'] = gdp_long['Quarter'].str.extract(r'(\d{4}Q\d)')[0]
gdp_long['Date'] = pd.PeriodIndex(gdp_long['Quarter'], freq='Q').to_timestamp()
germany_layoffs_data = layoffs[layoffs['Country'] == 'Germany']
germany_layoffs_data['Date'] = pd.to_datetime(germany_layoffs_data['Date_layoffs'])
germany_layoffs_data['Quarter'] = germany_layoffs_data['Date'].dt.to_period('Q')
quarterly_layoffs = germany_layoffs_data.groupby('Quarter').agg({'Laid_Off':'sum'}).reset_index()
quarterly_layoffs['Date'] = quarterly_layoffs['Quarter'].dt.to_timestamp()
merged_data = pd.merge(gdp_long, quarterly_layoffs, on='Date', how='outer')
merged_data['GDP'] = merged_data['GDP'].astype(float) / 1000
merged_data['Laid_Off'] = merged_data['Laid_Off'].fillna(0)
filtered_data = merged_data[(merged_data['Date'] >= '2020-01-01') & (merged_data['Date'] <= '2023-12-31')]
# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(filtered_data['Date'], filtered_data['GDP'], color='blue', label='GDP (Billions EUR)')
ax.set_xlabel('Date')
ax.set_ylabel('GDP (Billions EUR)', color='blue')
ax.set_title('Germany GDP and Tech Industry Layoffs (2020-2023)')
ax.set_xticks(filtered_data['Date'])
ax.set_xticklabels(filtered_data['Date'].dt.strftime('%Y-%m'), rotation=45, ha='right')
ax2 = ax.twinx()
ax2.bar(filtered_data['Date'], filtered_data['Laid_Off'], color='red', label='Layoffs', width=20)
ax2.set_ylabel('Number of Layoffs', color='red')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()


### Germany stock market and Layoffs ###
germany_stock_market_data = stockData[stockData['Country'] == 'Germany'].iloc[:, 4:]
germany_stock_market_data = germany_stock_market_data.replace('..', pd.NA).fillna(method='ffill', axis=1)
stock_market_long = pd.melt(germany_stock_market_data, var_name='Month', value_name='Stock_Index')
stock_market_long['Month'] = stock_market_long['Month'].str.extract(r'(\d{4}M\d{2})')[0]
stock_market_long['Month'] = stock_market_long['Month'].str.replace('M', '')
stock_market_long['Date'] = pd.to_datetime(stock_market_long['Month'], format='%Y%m').dt.to_period('M').dt.to_timestamp()
stock_market_long['Quarter'] = stock_market_long['Date'].dt.to_period('Q')
quarterly_stock_market = stock_market_long.groupby('Quarter').agg({'Stock_Index':'mean'}).reset_index()
quarterly_stock_market['Date'] = quarterly_stock_market['Quarter'].dt.to_timestamp()
# Load and prepare layoffs data
germany_layoffs_data = layoffs[layoffs['Country'] == 'Germany']
germany_layoffs_data['Date'] = pd.to_datetime(germany_layoffs_data['Date_layoffs']).dt.to_period('M').dt.to_timestamp()
germany_layoffs_data['Quarter'] = germany_layoffs_data['Date'].dt.to_period('Q')
quarterly_layoffs = germany_layoffs_data.groupby('Quarter').agg({'Laid_Off':'sum'}).reset_index()
quarterly_layoffs['Date'] = quarterly_layoffs['Quarter'].dt.to_timestamp()
# Merge the datasets and prepare for plotting
stock_market_merged_data = pd.merge(quarterly_stock_market, quarterly_layoffs, on='Date', how='outer')
stock_market_merged_data['Stock_Index'] = stock_market_merged_data['Stock_Index'].astype(float)
stock_market_merged_data['Laid_Off'] = stock_market_merged_data['Laid_Off'].fillna(0)
stock_market_filtered_data = stock_market_merged_data[(stock_market_merged_data['Date'] >= '2020-01-01') & (stock_market_merged_data['Date'] <= '2023-12-31')]
# Plotting the data
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_market_filtered_data['Date'], stock_market_filtered_data['Stock_Index'], color='green', label='Stock Index')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Index', color='green')
ax.set_title('Germany Stock Market Indexes and Technology Industry Layoffs (2020-2023)')
quarter_dates = [d for d in stock_market_filtered_data['Date']]
quarter_labels = [d.strftime('%Y Q%q') for d in stock_market_filtered_data['Date'].dt.to_period('Q')]
ax.set_xticks(quarter_dates)
ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
ax2 = ax.twinx()
ax2.bar(stock_market_filtered_data['Date'], stock_market_filtered_data['Laid_Off'], color='red', label='Layoffs', width=20)
ax2.set_ylabel('Number of Layoffs', color='red')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()


### India GDP and Layoffs ###
india_gdp_data = gdp_data[gdp_data['Country'] == 'India'].iloc[:, 4:]
india_gdp_data = india_gdp_data.replace('..', pd.NA).fillna(method='ffill', axis=1)
gdp_long_india = pd.melt(india_gdp_data, var_name='Quarter', value_name='GDP')
gdp_long_india['Quarter'] = gdp_long_india['Quarter'].str.extract(r'(\d{4}Q\d)')[0]
gdp_long_india['Date'] = pd.PeriodIndex(gdp_long_india['Quarter'], freq='Q').to_timestamp()

india_layoffs_data = layoffs[layoffs['Country'] == 'India']
india_layoffs_data['Date'] = pd.to_datetime(india_layoffs_data['Date_layoffs'])
india_layoffs_data['Quarter'] = india_layoffs_data['Date'].dt.to_period('Q')
quarterly_layoffs_india = india_layoffs_data.groupby('Quarter').agg({'Laid_Off':'sum'}).reset_index()
quarterly_layoffs_india['Date'] = quarterly_layoffs_india['Quarter'].dt.to_timestamp()

merged_data_india = pd.merge(gdp_long_india, quarterly_layoffs_india, on='Date', how='outer')
merged_data_india['GDP'] = merged_data_india['GDP'].astype(float) / 1000
merged_data_india['Laid_Off'] = merged_data_india['Laid_Off'].fillna(0)
filtered_data_india = merged_data_india[(merged_data_india['Date'] >= '2020-01-01') & (merged_data_india['Date'] <= '2023-12-31')]

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(filtered_data_india['Date'], filtered_data_india['GDP'], color='blue', label='GDP (Billions INR)')
ax.set_xlabel('Date')
ax.set_ylabel('GDP (Billions INR)', color='blue')
ax.set_title('India GDP and Tech Industry Layoffs (2020-2023)')
ax.set_xticks(filtered_data_india['Date'])
ax.set_xticklabels(filtered_data_india['Date'].dt.strftime('%Y-%m'), rotation=45, ha='right')
ax2 = ax.twinx()
ax2.bar(filtered_data_india['Date'], filtered_data_india['Laid_Off'], color='red', label='Layoffs', width=20)
ax2.set_ylabel('Number of Layoffs', color='red')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()


### India stock market and Layoffs ###
india_stock_market_data = stockData[stockData['Country'] == 'India'].iloc[:, 4:]
india_stock_market_data = india_stock_market_data.replace('..', pd.NA).fillna(method='ffill', axis=1)
stock_market_long = pd.melt(india_stock_market_data, var_name='Month', value_name='Stock_Index')
stock_market_long['Month'] = stock_market_long['Month'].str.extract(r'(\d{4}M\d{2})')[0]
stock_market_long['Month'] = stock_market_long['Month'].str.replace('M', '')
stock_market_long['Date'] = pd.to_datetime(stock_market_long['Month'], format='%Y%m').dt.to_period('M').dt.to_timestamp()
stock_market_long['Quarter'] = stock_market_long['Date'].dt.to_period('Q')
quarterly_stock_market = stock_market_long.groupby('Quarter').agg({'Stock_Index':'mean'}).reset_index()
quarterly_stock_market['Date'] = quarterly_stock_market['Quarter'].dt.to_timestamp()

# Load and prepare layoffs data
india_layoffs_data = layoffs[layoffs['Country'] == 'India']
india_layoffs_data['Date'] = pd.to_datetime(india_layoffs_data['Date_layoffs']).dt.to_period('M').dt.to_timestamp()
india_layoffs_data['Quarter'] = india_layoffs_data['Date'].dt.to_period('Q')
quarterly_layoffs = india_layoffs_data.groupby('Quarter').agg({'Laid_Off':'sum'}).reset_index()
quarterly_layoffs['Date'] = quarterly_layoffs['Quarter'].dt.to_timestamp()

# Merge the datasets and prepare for plotting
stock_market_merged_data = pd.merge(quarterly_stock_market, quarterly_layoffs, on='Date', how='outer')
stock_market_merged_data['Stock_Index'] = stock_market_merged_data['Stock_Index'].astype(float)
stock_market_merged_data['Laid_Off'] = stock_market_merged_data['Laid_Off'].fillna(0)
stock_market_filtered_data = stock_market_merged_data[(stock_market_merged_data['Date'] >= '2020-01-01') & (stock_market_merged_data['Date'] <= '2023-12-31')]

# Plotting the data
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_market_filtered_data['Date'], stock_market_filtered_data['Stock_Index'], color='green', label='Stock Index')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Index', color='green')
ax.set_title('India Stock Market Indexes and Technology Industry Layoffs (2020-2023)')
quarter_dates = [d for d in stock_market_filtered_data['Date']]
quarter_labels = [d.strftime('%Y Q%q') for d in stock_market_filtered_data['Date'].dt.to_period('Q')]
ax.set_xticks(quarter_dates)
ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
ax2 = ax.twinx()
ax2.bar(stock_market_filtered_data['Date'], stock_market_filtered_data['Laid_Off'], color='red', label='Layoffs', width=20)
ax2.set_ylabel('Number of Layoffs', color='red')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
