from market_data.market_data import MarketData
import pandas as pd
import sqlite3

# Database constant

C_DB = sqlite3.connect("market_data.sqlite")



def get_historical_data(fund: str) -> pd.DataFrame:
    """ Get historical data """
    
    #if lookback == 'today':
    #    time_period = TimePeriod.today
    #else:
    #    time_period = TimePeriod.five_years
    
    market_data = MarketData()
    fund_name, fund_ticker, fund_id, fee = market_data.get_by_name(fund)
    price_data = market_data.get_historical_data(fund)
    price_data = price_data.rename(columns={'value':'price', 'time':'date'})# .set_index('date')
    price_data['fee'] = fee
    price_data['ticker'] = fund_ticker
    return price_data


def create_data_range(fund: str, con):
    """ Save historical data in a database defined by con """
    
    data = get_historical_data(fund)
    data.to_sql(
        "fund_data", 
        con, 
        if_exists="append", 
        index=False,
        index_label="date2"
    )


def get_data_range(fund: str, con) -> pd.DataFrame:
    """ Get data range from database defined by con """
    
    data = pd.read_sql_query("SELECT * FROM fund_data WHERE ticker='{ffund}'".format(ffund = fund), con)
    
    # Convert to pandas Timestamp
    data['date'] = data['date'].apply(lambda row: pd.Timestamp(row))
    return data


def save_fund_data(fund: str, con) -> pd.DataFrame:
    """ Save fund data in a database defined by con """
    
    data = get_historical_data(fund)
    
    try:
        current_data = get_data_range(fund, con)
    except:
        print("No such table: ", con)
        
    if current_data.empty:
            data.to_sql(
            "fund_data", 
            con, 
            if_exists="append", 
            index=False
            )
    else:
        most_recent_date = pd.Timestamp(current_data.date.iloc[-1]) #if not current_data.empty else pd.Timestamp(0)
        if most_recent_date == data.date.iloc[-1]:
            return print("No new data available for fund: ", fund)
        else:
            new_data = data.where(data.date > most_recent_date).dropna()
            new_data.to_sql(
            "fund_data", 
            con, 
            if_exists="append", 
            index=False
            )
    return print("Data updated for fund: ", fund)


def save_opt_data(data: pd.DataFrame, con):
    """ Save optimization data to table. """
    data.to_sql(
        "opt_data", 
        con, 
        if_exists="append", 
        index=False
    )


def read_table(table: str, con):
    """ Read database table and return everything. """
    return pd.read_sql_query("SELECT * FROM '{ttable}'".format(ttable = table), con)


def delete_from_table(fund: str, con):
    """ Delete fund ticker and data from table. """
    try:
        pd.read_sql_query("DELETE FROM fund_data where ticker='{ffund}'".format(ffund = fund), con)
    except TypeError:
        print("Deleted: ", fund)


def append_data(data: pd.DataFrame, fund: str, con):
    """ Append new data to table for a specific ticker. """
    curr_data = pd.read_sql_query("SELECT * FROM fund_data WHERE ticker='{ffund}'".format(ffund = fund), con)
    data.merge(curr_data, how="inner", on="date")
    return data
    

def drop_table(table: str, con):
    """ Drop table. """
    pd.read_sql_query("DROP TABLE '{ttable}'".format(ttable = table), con)
    

    