# setup logging
import logger

log = logger.get_logger(__name__)

import asyncio
import aiohttp
from functools import reduce
from typing import Dict
from utilities import encode_identifier
from typing import List
import pandas as pd
import json


async def retrieve_CID(identifier: str,
                       identifier_type: str,
                       session: aiohttp.ClientSession,
                       semaphore: asyncio.Semaphore,
                       timeout: float = 30.,
                       retries: int = 5) -> Dict:
    '''
    Retrieves the PubChem CID(s) from an identifier.
    For more information please see https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest-tutorial.
    :param identifier: the input identifier
    :param identifier_type: the input identifier type (so far we allow 'CAS number', 'EC number', or 'name')
    :param session: aiohttp client session
    :param semaphore: asyncio semaphore to control concurrency
    :param timeout: time out for the PubChem POST request (in sec)
    :param retries: number of retries of the REST request

    :return: dictionary with the CID(s)
    '''
    pubchem_base_url = r'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
    url = pubchem_base_url + 'compound/name/cids/JSON'

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if identifier_type == 'name':
        data = 'name=' + encode_identifier(identifier)
    elif identifier_type == 'CAS number':
        data = 'name=' + identifier
    elif identifier_type == 'EC number':
        data = 'name=EINECS ' + identifier
    else:
        raise ValueError(f"identifier type can only take the values 'name', 'CAS number' or 'EC number', "
                         f"{identifier_type} given")
    data = data.encode(encoding='utf-8')

    for i_retry in range(retries):
        try:

            # note the two context managers
            # https://stackoverflow.com/questions/66724841/using-a-semaphore-with-asyncio-in-python
            # we pass a dictionary that is updated by the client tracing
            trace_request_ctx = {'request duration': None}
            async with semaphore, session.post(url=url, data=data, timeout=timeout, headers=headers,
                                               trace_request_ctx=trace_request_ctx) as resp:

                original_response = await resp.json()

                # log the request
                log_request = f"url:status -> {url}:{resp.status}, " \
                             +f"data {data}" \
                             +f"response time {trace_request_ctx['request duration']}, " \
                             +f"dynamic throttle {resp.headers['X-Throttling-Control']}"
                log.info(log_request)

                # critical sleep to ensure that load does not exceed PubChem's thresholds
                min_time_per_request = 1.1
                if trace_request_ctx['request duration'] < min_time_per_request:
                    idle_time = min_time_per_request - trace_request_ctx['request duration']
                    await asyncio.sleep(idle_time)

                # successful response
                if resp.status == 200:
                    # retrieve the PubChem Compound IDs (CID) from the response json
                    CID_path = 'IdentifierList/CID'
                    CIDs = reduce(lambda x, p: x[p], CID_path.split('/'), original_response)
                    result = {'identifier': identifier,
                              'identifier type': identifier_type,
                              'CID': CIDs,
                              'status': 'succeeded',
                              'response_code': resp.status,
                              'response_text': original_response,
                              'response_time': trace_request_ctx['request duration'],
                              'exception_text': None,
                              'number of retries': i_retry + 1}
                    return result

                # unsuccessful response (503: PubChem server busy, we will retry)
                elif resp.status == 503:
                    log.error(log_request + ". response text: " + json.dumps(original_response)
                              + '. response code: ' + str(resp.status))
                    if i_retry == retries - 1:
                        result = {'identifier': identifier,
                                  'identifier type': identifier_type,
                                  'status': 'failed',
                                  'response_code': resp.status,
                                  'response_text': original_response,
                                  'response_time': trace_request_ctx['request duration'],
                                  'exception_text': None,
                                  'number of retries': i_retry + 1}
                        return result

                # unsuccessful response (we will not retry)
                else:
                    log.error(
                        log_request + ". response text: " + json.dumps(original_response) + '. response code: ' + str(
                            resp.status))
                    result = {'identifier': identifier,
                              'identifier type': identifier_type,
                              'status': 'failed',
                              'response_code': resp.status,
                              'response_text': original_response,
                              'response_time': trace_request_ctx['request duration'],
                              'exception_text': None,
                              'number of retries': i_retry + 1}
                    return result

        except asyncio.exceptions.TimeoutError:
            log.error(f'timeout error for: {url}, data: {data}')
            if i_retry == retries - 1:
                # compile the result
                result = {'identifier': identifier,
                          'identifier type': identifier_type,
                          'status': 'failed',
                          'exception_text': 'timeout',
                          'number of retries': i_retry + 1}
                return result

        except aiohttp.ClientError as e:
            log.error(f'client error for: {url}, data: {data}, exception: {str(e)}')
            if i_retry == retries - 1:
                # compile the result
                result = {'identifier': identifier,
                          'identifier type': identifier_type,
                          'status': 'failed',
                          'exception_text': str(e),
                          'number of retries': i_retry + 1}
                return result


        except Exception as e:
            log.error(f'unexpected error for: {url}, data: {data}, exception: {str(e)}')
            # compile the result
            result = {'identifier': identifier,
                      'identifier type': identifier_type,
                      'status': 'failed',
                      'exception_text': str(e),
                      'number of retries': i_retry + 1}
            if i_retry == retries - 1:
                return result


async def retrieve_pugview(CID: str,
                           type: str,
                           session: aiohttp.ClientSession,
                           semaphore: asyncio.Semaphore,
                           timeout: float = 30.,
                           retries: int = 5) -> Dict:
    '''
    Retrieves the index or the complete dataset of a compound using PubChem PUG View.
    For more information please see https://pubchemdocs.ncbi.nlm.nih.gov/pug-view.
    :param CID: The PubChem compound ID (CID)
    :param type: Specifies whether the index (type='index') or complete dataset (type='data') is returned
    :param session: aiohttp client session
    :param semaphore: asyncio semaphore to control concurrency
    :param timeout: time out for the PubChem POST request (in sec)
    :param retries: number of retries of the REST request
    :return: Dictionary with the index or complete dataset
    '''
    if type == 'index':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/index/compound/{CID}/JSON'
    elif type == 'data':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{CID}/JSON'
    else:
        raise ValueError("type must be 'index' or 'data'")

    for i_retry in range(retries):
        try:
            # note the two context managers
            # https://stackoverflow.com/questions/66724841/using-a-semaphore-with-asyncio-in-python
            # we pass a dictionary that is updated by the client tracing
            trace_request_ctx = {'request duration': None}
            async with semaphore, session.get(url=url, timeout=timeout, trace_request_ctx=trace_request_ctx) as resp:

                original_response = await resp.json()

                # log the request
                log_request = f"url:status -> {url}:{resp.status}, " \
                             +f"response time {trace_request_ctx['request duration']}, " \
                             +f"dynamic throttle {resp.headers['X-Throttling-Control']}"
                log.info(log_request)

                # critical sleep to ensure that load does not exceed PubChem's thresholds
                min_time_per_request = 1.1
                if trace_request_ctx['request duration'] < min_time_per_request:
                    idle_time = min_time_per_request - trace_request_ctx['request duration']
                    await asyncio.sleep(idle_time)

                # successful response
                if resp.status == 200:
                    result = {'CID': CID,
                              'pugview data': original_response,
                              'status': 'succeeded',
                              'response_code': resp.status,
                              'response_text': original_response,
                              'response_time': trace_request_ctx['request duration'],
                              'exception_text': None,
                              'number of retries': i_retry + 1}
                    return result

                # unsuccessful response (503: PubChem server busy, we will retry)
                elif resp.status == 503:
                    log.error(log_request + ". response text: " + json.dumps(original_response)
                              + '. response code: ' + str(resp.status))
                    if i_retry == retries - 1:
                        result = {'CID': CID,
                                  'pugview data': {},
                                  'status': 'failed',
                                  'response_code': resp.status,
                                  'response_text': original_response,
                                  'response_time': trace_request_ctx['request duration'],
                                  'exception_text': None,
                                  'number of retries': i_retry + 1}
                        return result

                # unsuccessful response (we will not retry)
                else:
                    log.error(log_request + ". response text: " + json.dumps(original_response)
                              + '. response code: ' + str(resp.status))
                    result = {'CID': CID,
                              'pugview data': {},
                              'status': 'failed',
                              'response_code': resp.status,
                              'response_text': original_response,
                              'response_time': trace_request_ctx['request duration'],
                              'exception_text': None,
                              'number of retries': i_retry + 1}
                    return result
        except asyncio.exceptions.TimeoutError as e:
            log.error(f'timeout error for: {url}')
            if i_retry == retries - 1:
                # compile the result
                result = {'CID': CID,
                          'status': 'failed',
                          'exception_text': 'timeout',
                          'number of retries': i_retry + 1}
                return result

        except aiohttp.ClientError as e:
            log.error(f'client error for: {url}, exception: {str(e)}')
            if i_retry == retries - 1:
                # compile the result
                result = {'CID': CID,
                          'status': 'failed',
                          'exception_text': str(e),
                          'number of retries': i_retry + 1}
                return result

        except Exception as e:
            log.error(f'unexpected error for: {url}, exception: {str(e)}')
            if i_retry == retries - 1:
                # compile the result
                result = {'CID': CID,
                          'status': 'failed',
                          'exception_text': str(e),
                          'number of retries': i_retry + 1}
                return result


async def batch_conversion(identifiers: List[str],
                           identifier_types: List[str],
                           limit: int = 10000,
                           timeout_PubChem: float = 10.,
                           number_of_simultaneous_connections=None) -> pd.DataFrame:
    # set the semaphore
    sem = asyncio.Semaphore(limit)

    # set the session timeout (this affects all requests)
    # https://stackoverflow.com/questions/64534844/python-asyncio-aiohttp-timeout
    # https://github.com/aio-libs/aiohttp/issues/3203
    session_timeout = aiohttp.ClientTimeout(total=None)

    # the below is needed to measure the request time (needed because PubChem imposes constraints on the computing time)
    # https://docs.aiohttp.org/en/stable/client_advanced.html#client-tracing
    # https://stackoverflow.com/questions/56990958/how-to-get-response-time-and-response-size-while-using-aiohttp
    async def on_request_start(session, trace_config_ctx, params):
        trace_config_ctx.start = asyncio.get_event_loop().time()

    async def on_request_end(session, trace_config_ctx, params):
        elapsed_time = asyncio.get_event_loop().time() - trace_config_ctx.start
        if trace_config_ctx.trace_request_ctx['request duration'] is not None:
            raise Exception('should not happen')
        trace_config_ctx.trace_request_ctx['request duration'] = elapsed_time

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)

    # by creating a connector we set up the maximum number of simultaneous connections to one host
    if number_of_simultaneous_connections is not None:
        connector = aiohttp.TCPConnector(limit_per_host=number_of_simultaneous_connections)
    else:
        connector = None
    async with aiohttp.ClientSession(connector=connector, timeout=session_timeout,
                                     trace_configs=[trace_config]) as session:
        # obtain the CID data
        tasks = []
        for identifier, identifier_type in zip(identifiers, identifier_types):
            tasks.append(
                retrieve_CID(identifier, identifier_type, session=session, semaphore=sem, timeout=timeout_PubChem,
                             retries=5))
        CID_results = await asyncio.gather(*tasks)
        CID_results = pd.DataFrame(CID_results)
        await asyncio.sleep(1.)
        # obtain unique CIDs
        CIDs = CID_results['CID'].explode().dropna().drop_duplicates().to_list()
        tasks = []
        for CID in CIDs:
            tasks.append(
                retrieve_pugview(str(CID), type='index', session=session, semaphore=sem, timeout=timeout_PubChem,
                                 retries=5))
        pugview_results = await asyncio.gather(*tasks)
        pugview_results = pd.DataFrame(pugview_results)

    return CID_results, pugview_results
