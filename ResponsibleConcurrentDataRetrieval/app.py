# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='output/PubChem_integration.log')

import pandas as pd
import synchronous
import asynchronous_asyncio_semaphore
from utilities import parse_TOCHeadings
import time
import asyncio


# read the substance list
substances = pd.read_excel(r'input/2022_07_15_chemical_universe_list_en.xlsx', index_col=None)
msk = substances['CAS'].str.match(r'\d{2,7}-\d{2}-\d', na=False)
substances = substances.loc[msk].reset_index(drop=False)


# test the synchronous approach
start = time.perf_counter()
CIDs = synchronous.retrieve_CID(identifier='50-00-0', identifier_type='CAS number')
print(f'execution time {time.perf_counter()-start} sec')
res = synchronous.retrieve_pugview(CIDs[0], type='index')
print(f'execution time {time.perf_counter()-start} sec')
TOCHeading_paths = parse_TOCHeadings(res['Record']['Section'])


# test the asynchronous approach (semaphore)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
nsubs = len(substances)
identifiers = substances.iloc[:nsubs]['CAS'].drop_duplicates().to_list()
identifier_types = ['CAS number']*len(identifiers)
start_overall = time.perf_counter()
results = asyncio.run(asynchronous_asyncio_semaphore.batch_conversion(identifiers = identifiers,
                                                                      identifier_types = identifier_types,
                                                                      limit = 5,
                                                                      timeout_PubChem = 10.,
                                                                      number_of_simultaneous_connections = None))
total_time = time.perf_counter() - start_overall
print(f'total execution time is {total_time} sec')
results[0].to_parquet(r'output/CID_results.parquet')
results[1].to_parquet(r'output/pugview_results.parquet')


# examine the available data for each retrieved CID
results[1]['available data'] = results[1]['pugview data'].apply(lambda x: parse_TOCHeadings(x['Record']['Section']))
data_availability = results[1].explode('available data').groupby('available data')['CID'].count().rename('number of CIDs').sort_index(ascending=False)
data_availability.to_excel('output/data_availability.xlsx')
