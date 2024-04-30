import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
layoffs = pd.read_excel('cleaned_layoffs.xlsx')
GDPdata = pd.read_excel('cleaned_GDP.xlsx')
stockData = pd.read_excel('cleaned_stock_market.xlsx')

# Prepare layoffs data
layoffs['Date_layoffs'] = pd.to_datetime(layoffs['Date_layoffs'])
layoffs.set_index('Date_layoffs', inplace=True)

# Filter and prepare GDP and stock market data for USA, India, and Germany
countries = ['United States', 'India', 'Germany']
gdp_data = []
stock_market_data = []
for country in countries:
    df_gdp_country = GDPdata[GDPdata['Country'] == country]
    gdp_time_columns = df_gdp_country.columns[df_gdp_country.columns.str.contains(r'\d{4}Q\d')]
    df_gdp_country_quarterly = df_gdp_country[gdp_time_columns].transpose()
    df_gdp_country_quarterly.index = pd.to_datetime(df_gdp_country_quarterly.index.str.extract(r'(\d{4}Q\d)')[0])
    df_gdp_country_quarterly.index = df_gdp_country_quarterly.index.to_period('Q').to_timestamp('Q')
    df_gdp_country_quarterly.columns = [f'GDP_{country}']
    gdp_data.append(df_gdp_country_quarterly)

    df_stock_market_country = stockData[stockData['Country'].str.contains(country, na=False)]
    stock_time_columns = df_stock_market_country.columns[df_stock_market_country.columns.str.match(r'\d{4}Q\d')]
    df_stock_market_country_quarterly = df_stock_market_country[stock_time_columns].transpose()
    df_stock_market_country_quarterly.index = pd.to_datetime(df_stock_market_country_quarterly.index).to_period('Q').to_timestamp('Q')
    df_stock_market_country_quarterly.columns = [f'Stock Market_{country}']
    stock_market_data.append(df_stock_market_country_quarterly)

# Combine all GDP and Stock Market data
gdp_combined = pd.concat(gdp_data, axis=1)
stock_market_combined = pd.concat(stock_market_data, axis=1)

# Resample layoffs data quarterly and sum the layoffs, filter for selected countries
df_layoffs_selected = layoffs[layoffs['Country'].isin(countries)]
df_layoffs_quarterly = df_layoffs_selected['Laid_Off'].resample('Q').sum()

# Merge all data on the date index
merged_data = pd.concat([df_layoffs_quarterly, gdp_combined, stock_market_combined], axis=1).dropna()

# Prepare data for modeling
X = merged_data.drop('Laid_Off', axis=1)
y = merged_data['Laid_Off']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models with parameters to prevent overfitting
dt_regressor = DecisionTreeRegressor(random_state=42, max_depth=5)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
dt_regressor.fit(X_train, y_train)
rf_regressor.fit(X_train, y_train)

# Predict on the entire dataset for visualization
merged_data['DT_Predictions'] = dt_regressor.predict(X)
merged_data['RF_Predictions'] = rf_regressor.predict(X)

# Evaluate the model on the test set and print MSE
dt_mse = mean_squared_error(y_test, dt_regressor.predict(X_test))
rf_mse = mean_squared_error(y_test, rf_regressor.predict(X_test))
print(f'Decision Tree MSE: {dt_mse}')
print(f'Random Forest MSE: {rf_mse}')

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(merged_data.index, merged_data['Laid_Off'], label='Actual Layoffs', marker='o')
plt.plot(merged_data.index, merged_data['DT_Predictions'], label='Decision Tree Predictions', marker='x')
plt.plot(merged_data.index, merged_data['RF_Predictions'], label='RandomForest Predictions', linestyle='--')
plt.title('Comparison of Layoffs Predictions and Actual Data by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Number of Layoffs')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
