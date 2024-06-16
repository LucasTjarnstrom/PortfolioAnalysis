from market_data.constants import InstrumentType,\
                      ChartType,\
                      ChartResoluton, \
                      TimePeriod
from datetime import datetime as dt
import requests
import json
import pandas





class MarketData:
    """ Class that can be used to download historical data from Avanza. """
    
    URL_SEARCH = 'https://www.avanza.se/_mobile/market/search/{instrument}?query={query}&limit={limit}'
    URL_INSTRUMENTS = 'https://avanza.se/_api/market-guide/stock/{id}/'
    URL_INSPIRATION = 'https://avanza.se/_mobile/marketing/inspirationlist'
    URL_HISTORICAL = 'https://www.avanza.se/ab/component/highstockchart/getchart/orderbook'
    URL_FUNDS = 'https://www.avanza.se/fonder/lista.html?sortField=developmentOneYear&sortDirection=DESCENDING&selectedTab=overview'


    def __init__(self):
        self.__session = requests.Session()


    def search(self,
               query: 'str',
               instrument: 'InstrumentType'=InstrumentType.NOT_SELECTED,
               limit: 'int'=5,
               local: 'bool'=False)-> '[]': 
        
        """ Search using the website with the input query. """
        
        if not local:
            url = MarketData.URL_SEARCH.replace('{query}', query)\
                                   .replace('{limit}', str(limit))
            if instrument == InstrumentType.NOT_SELECTED:
                url = url.replace('/{instrument}', '')
            else:
                url = url.replace('{instrument}', instrument.name)
            
            response = self.__request_get_no_login(url)
            hits = []
            
            if response.get('totalNumberOfHits', 0) == 0:
                return []
            else:
                for instrument_hit in response['hits']:
                    for hit in instrument_hit.get('topHits', ''):
                        hits.append(hit)
        else:
            raise Exception(("{} not found".format(query)))

        return hits


    def get_by_name(self, instrument_name):
        """ Find the best match for a given instrument name """

        matches = self.search(instrument_name, limit=1)
        if matches[0] == None:
            raise Exception("No results found.")
        _name = matches[0].get('name')
        _fee = matches[0].get('managementFee')
        _id = matches[0].get('id')
        _ticker = matches[0].get('tickerSymbol')
        #self.disconnect()
        return _name, _ticker, _id, _fee
    
    
    def get_tickers(self, instrument_type: str):
        """ Find tickers for an instrument type """
        
        #matches = self.search(instrument_type, limit=1)
        url = MarketData.URL_SEARCH.replace('{instrument}', instrument_type)
        response = self.__request_get_no_login(url)
        #self.disconnect()
        return response

        # TODO
    def get_fund_tickers(self):
        """ Get all available fund tickers. """
        url = MarketData.URL_FUNDS
        response = self.__request_get_no_login(url)
        #self.disconnect()
        return response
 
    
    def get_id(self, instrument_name):
        """ Get instrument id. """

        matches = self.search(instrument_name, limit=1)
        if matches[0] == None:
            raise Exception("No results found.")
        _id = matches[0].get('id')
        return _id
    
    
    def get_historical_data(self, instrument_name: str):
        """ Get all available historical data for an instrument by instrument name. """

        instrument_id = self.get_id(instrument_name)
        
        #url = self.URL_INSTRUMENTS.replace('{id}', instrument_id)
        #response = self.__request_get_no_login(url)

        hist_data = self.get_historical(instrument_id,
                                chart_type=ChartType.AREA,
                                chart_resolution=ChartResoluton.DAY,
                                time_period=TimePeriod.five_years)

        return hist_data


    def get_historical(self,
                       instrument_id,
                       chart_type=ChartType.AREA,
                       chart_resolution=ChartResoluton.DAY,
                       time_period=TimePeriod.five_years):
        """ Get historical data for a given instrument. """
        
        p = {
            "orderbookId": instrument_id,
            "chartType": chart_type.name,
            "chartResolution": chart_resolution.name,
            "timePeriod": time_period.name
        }

        r = self.__request_post_no_login(self.URL_HISTORICAL, params=p)

        data_series = r['dataPoints']
        for x in data_series:
            x[0] = dt.fromtimestamp(x[0] / 1000)
        if chart_type == ChartType.AREA:
            df = pandas.DataFrame(data_series, columns=['time', 'value'])
        else:
            df = pandas.DataFrame(data_series, columns=['time', 'opens', 'highs', 'lows', 'closes'])

        df = df.dropna()

        return df


    def __request_post_no_login(self, url, params):
        h = {"Content-Type": "application/json"}
        r = self.__session.post(url, data=json.dumps(params), headers=h).json()

        return r


    def __request_get_no_login(self, url):
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            raise Exception("Error in request, status code not 200")