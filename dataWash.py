import pandas as pd

# File paths
layoffs = 'tech_layoffs.xlsx'
output_file_path = 'cleaned_layoffs.xlsx'
GDPdata = 'GDP.xlsx'
cleaned_gdp_file_path = 'cleaned_GDP.xlsx'
stockData = 'Stock market.xlsx'
cleaned_stock_market_file = 'cleaned_stock_market.xlsx'

# Load Excel data from a specified file path.
def load_data(file_path):
    return pd.read_excel(file_path)

# Save DataFrame to an Excel file without the index.
def save_data(data, output_path):
    data.to_excel(output_path, index=False)

# Filter data for specific countries and modify country names.
def filter_and_modify_data(data):
    filtered_data = data[data['Country'].isin(['USA', 'Germany', 'India'])]
    filtered_data['Country'] = filtered_data['Country'].replace('USA', 'United States')
    return filtered_data

# Load data, replace placeholders, fill forward, and drop rows with all missing quarters.
def clean_and_fill_data(file_path, output_file_path):
    data = pd.read_excel(file_path)
    data.replace('..', pd.NA, inplace=True)
    data.fillna(method='ffill', axis=1, inplace=True)
    data.dropna(subset=[col for col in data.columns if 'Q' in col], how='all', inplace=True)
    data.to_excel(output_file_path, index=False)
    return data

# Convert monthly data to quarterly averages and save to a new Excel file.
def convert_monthly_to_quarterly(file_path, output_file_path):
    data = pd.read_excel(file_path)
    for col in data.columns[4:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    quarters = {'Q1': ['01', '02', '03'], 'Q2': ['04', '05', '06'], 'Q3': ['07', '08', '09'], 'Q4': ['10', '11', '12']}
    quarterly_data = pd.DataFrame(data[['Country', 'Series']])  # Include Country and Series first
    for year in range(2020, 2024):
        for quarter, months in quarters.items():
            month_columns = [f"{year}M{month} [{year}M{month}]" for month in months if f"{year}M{month} [{year}M{month}]" in data.columns]
            quarterly_data[f'{year}{quarter}'] = data[month_columns].mean(axis=1, skipna=True)
    quarterly_data.to_excel(output_file_path, index=False)
    return output_file_path


data = load_data(layoffs)
filtered_data = filter_and_modify_data(data)
save_data(filtered_data, output_file_path)
cleaned_gdp_data = clean_and_fill_data(GDPdata, cleaned_gdp_file_path)
output_path = convert_monthly_to_quarterly(stockData, cleaned_stock_market_file)
