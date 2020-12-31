# Pennsylvania Mail Ballot Open Dataset
# Author Pranav Banthia

# Install key libraries
# pip install sodapy
# pip install addfips
# pip install geopandas
# pip install plotly

# Importing Libraries
import pandas as pd
import requests
from sodapy import Socrata
import addfips
from datetime import datetime
from sklearn.linear_model import LinearRegression
from functools import reduce
import plotly.figure_factory as ff
import io
from sklearn.preprocessing import LabelEncoder


def convert_snake_case(x: str):
    x = reduce(lambda a, b: a + ('_' if a.isnumeric() and b.isalpha() else '') + b, x)
    return x.replace(' ', '_').lower()


def extract_year_from_date(dob: str) -> int:
    date_of_birth = datetime.fromisoformat(dob)
    return date_of_birth.year


def get_ballot_return_time(ballotsentdate: str, ballotreturneddate: str):
    ballot_sent_date = datetime.fromisoformat(ballotsentdate)
    ballot_returned_date = datetime.fromisoformat(ballotreturneddate)
    turnaround_time = ballot_returned_date - ballot_sent_date
    return turnaround_time.days

def get_voter_age(yr_born):
    today = datetime.now()
    return today.year - yr_born


def get_fips(county_name: str):
    return af.get_county_fips(county_name, state='Pennsylvania')

def get_county_color(democratic, republic):
    total = democratic + republic
    blue = int((democratic / total) * 250)
    red = int((republic / total) * 250)
    color = f'rgb({red}, 0, {blue})'
    return color


if __name__ == '__main__':
    # Data source - https://data.pa.gov/Government-Efficiency-Citizen-Engagement/2020-General-Election-Mail-Ballot-Requests-Departm/mcba-yywm

    # Step - 1 Read data from the API endpoint provided by Pennsylvania open data
    # There are two ways to read data. One is by creating a Socrata client(faster) and the other is by hitting the API endpoint
    # Since the assessment asks for pulling data through the API endpoint we will be using the latter technique

    # Uncomment the below lines to retrieve data through the Socrata client
    # client = Socrata("data.pa.gov", apptoken, username, password)
    # results = client.get("mcba-yywm", limit=1000000)
    # applications_in = pd.DataFrame.from_records(results)

    URL = 'https://data.pa.gov/resource/mcba-yywm.csv'

    header = {'X-App-Token': 'PpRBirTt02UZNeONd9omnkhaq'}

    offset = 0
    limit = 1000000
    param = {'$limit': limit, '$offset': offset}

    print('Pulling data from API endpoint. Retrieving 3.08M rows....')

    r = requests.get(url=URL, headers=header, params=param)
    applications_in = pd.DataFrame()

    while offset <= 3080000:  # Total number of rows in dataset provided by PA open data website
        applications_in = applications_in.append(pd.read_csv(io.StringIO(r.text)))
        offset += limit
        param = {'$limit': limit, '$offset': offset}
        r = requests.get(url=URL, headers=header, params=param)

    print('Records retrieved\n')
    print('Size of the dataset (no of rows):', applications_in.size)
    print('---------------------------------------------------------------------------------------------------\n\n')

    print('\n Filtering records with null values')
    invalid_data = applications_in[applications_in.isnull().any(axis=1)].reset_index()
    print(invalid_data.head())
    # Removing any rows with null values
    applications_in = applications_in.dropna(how='any', axis=0).reset_index()

    print('---------------------------------------------------------------------------------------------------\n\n')

    print('Converting the state senate district to snake case\n')
    applications_in['senate'] = applications_in['senate'].apply(lambda x: convert_snake_case(x))
    print(applications_in[['countyname', 'party', 'senate']][0:10])

    print('---------------------------------------------------------------------------------------------------\n\n')

    print('Computing DOB for voters:\n')
    # Since we need the new column immediately to the right of the date of birth
    idx = applications_in.columns.get_loc("dateofbirth") + 1

    yr_born = applications_in['dateofbirth'].apply(lambda x: extract_year_from_date(x))
    applications_in.insert(loc=idx, column='yr_born', value=yr_born)
    print(applications_in[['countyname', 'party', 'dateofbirth', 'yr_born']][0:10])

    print('---------------------------------------------------------------------------------------------------\n\n')
    # For our convenience we will add another column called as age
    applications_in['voter_age'] = applications_in['yr_born'].apply(lambda x: get_voter_age(x))


    # To find a relation between two variables, the simplest way is to use linear regression
    # since party is a categorical column, we will first encode it
    le = LabelEncoder()
    applications_in['numeric_party'] = le.fit_transform(applications_in['party'])

    lr = LinearRegression()
    X = applications_in['voter_age'].values.reshape(-1,1)
    Y = applications_in['numeric_party'].values.reshape(-1,1)
    model = lr.fit(X, Y)
    print('Linear regression model coefficient: ', model.coef_)
    print('Linear regression model intercept: ',model.intercept_)
    print('We see from the model coefficient that there is almost no relation between voters age and the party they vote for. We see that for major parties like democrats or republicans voters consists of all age groups')

    print('---------------------------------------------------------------------------------------------------\n\n')

    print('Computing median latency for each legislative district\n')
    applications_in['turnaround_time'] = applications_in.apply(
                                    lambda x: get_ballot_return_time(x.ballotsentdate, x.ballotreturneddate), axis=1)

    latency_per_legislative_district = applications_in.groupby(applications_in.legislative)[
        ['turnaround_time']].median()
    print(latency_per_legislative_district)

    print('---------------------------------------------------------------------------------------------------\n\n')

    highest_ballot_req = applications_in['congressional'].value_counts().idxmax()
    print('The congressional district with the highest frequency of ballot requests is : ', highest_ballot_req)

    print('---------------------------------------------------------------------------------------------------\n\n')

    print('Creating a visualization of republic and democratic application counts in each county\n')

    af = addfips.AddFIPS()
    # For geographical data FIPS codes helps us to visualise it efficiently. Its a unique 5 digit code for each county
    applications_in['FIPS'] = applications_in['countyname'].apply(lambda x: get_fips(x))

    # Filtering the democratic and republic party records
    app_per_county = applications_in.loc[applications_in.party.isin(['D', 'R'])]
    # Grouping based on FIPS (county) and then the party to get the no of ballot request of each party in a county
    app_per_county = app_per_county.groupby(['FIPS', 'party'])[['countyname']].count().reset_index()
    app_per_county = app_per_county.pivot_table(index=['FIPS'], columns=['party'], values='countyname')
    app_per_county = app_per_county.fillna(0).reset_index()

    # Making our own unique color combination for democratic and republic counties based the number of ballots
    app_per_county['map_color'] = app_per_county.apply(lambda x: get_county_color(x.D, x.R), axis=1)


    # Plotly choropleth maps needs FIPS and values to plot. We plot the winner in each county against the FIPS code of
    # that county. Winner is the one with higher number of ballots. The respective shade of red and blue combination
    # determines to what extent the state is either democratic or republic
    app_per_county['values'] = app_per_county.apply(lambda x: x.D if x.D > x.R else x.R, axis=1)

    fips = app_per_county['FIPS'].to_list()
    values = app_per_county['values'].to_list()
    colorscale = app_per_county['map_color'].to_list()

    # The below visualization requires geopandas, pyshp and shapely libraries
    # Reference : https://plotly.com/python/county-choropleth/
    # !pip install geopandas==0.3.0
    # !pip install pyshp==1.2.10
    # !pip install shapely==1.6.3

    # fig = ff.create_choropleth(
    #    fips=fips,
    #    values=values,
    #    scope=['Pennsylvania'],
    #    show_state_data=True,
    #    colorscale=colorscale,
    #    show_hover=True,
    #    plot_bgcolor='rgb(229,229,229)',
    #    paper_bgcolor='rgb(229,229,229)',
    #    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}
    # )
    # fig.show()
