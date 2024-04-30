import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the data
layoffs = pd.read_excel('cleaned_layoffs.xlsx')
GDPdata = pd.read_excel('cleaned_GDP.xlsx')
stockData = pd.read_excel('cleaned_stock_market.xlsx')

# Prepare layoffs data
layoffs['Date_layoffs'] = pd.to_datetime(layoffs['Date_layoffs'])
layoffs.set_index('Date_layoffs', inplace=True)

# Filter and prepare GDP data for USA, India, and Germany
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

    # Prepare stock market data similarly
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

# Normalize the features
scaler = StandardScaler()
feature_columns = [col for col in merged_data.columns if col not in ['Laid_Off']]
scaled_features = scaler.fit_transform(merged_data[feature_columns])
scaled_df = pd.DataFrame(scaled_features, index=merged_data.index, columns=feature_columns)

# Prepare final DataFrame for modeling
final_data = pd.concat([scaled_df, merged_data['Laid_Off']], axis=1).dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    final_data[feature_columns],
    final_data['Laid_Off'],
    test_size=0.2,
    random_state=42
)

# Constructing the neural network model
model = Sequential([
    Dense(32, input_shape=(len(feature_columns),), activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Predict on the entire dataset for visualization
final_data['Predictions'] = model.predict(final_data[feature_columns])

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(final_data.index, final_data['Laid_Off'], label='Actual Layoffs', marker='o')
plt.plot(final_data.index, final_data['Predictions'], label='Neural Network Predictions', linestyle='--')
plt.title('Comparison of Neural Network Predictions and Actual Data')
plt.xlabel('Quarter')
plt.ylabel('Number of Layoffs')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()