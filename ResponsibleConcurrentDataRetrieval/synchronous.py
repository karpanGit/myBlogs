# setup logging
import logger
log = logger.get_logger(__name__)

import requests
from functools import reduce
from typing import Dict
from utilities import encode_identifier


def retrieve_CID(identifier: str, identifier_type:str, timeout: float=30.) -> Dict:
    '''
    Retrieves the PubChem CID(s) from an identifier.
    For more information please see https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest-tutorial.
    :param identifier: the input identifier
    :param identifier_type: the input identifier type
    :param timeout: time out for the PubChem POST request (in sec)
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
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=timeout)

        # log the request
        log_request = f'url:status -> {url}:{resp.status_code}, data: {data}'
        log.info(log_request)

        log.info("dynamic throttle: {throttle}".format(throttle=resp.headers['X-Throttling-Control']))

        # successful response
        if resp.status_code == 200:
            # retrieve the PubChem Compound IDs (CID) from the response json
            CID_path = 'IdentifierList/CID'
            CIDs = reduce(lambda x, p: x[p], CID_path.split('/'), resp.json())
            return CIDs

        # unsuccessful response
        else:
            log.error(log_request + ". response text: " + resp.text + '. response code: ' + str(resp.status_code))
            return []

    except requests.exceptions.RequestException as e:
        log_msg = f'requests exception for: {url}, data: {data}, exception: {str(e)}'
        log.error(log_msg)
        return []

    except Exception as e:
        log_msg = f'requests exception for: {url}, data: {data}, exception: {str(e)}'
        log.error(log_msg)
        return []

def retrieve_pugview(CID: str, type: str ='index', timeout: float=30.) -> Dict:
    '''
    Retrieves the index or the complete dataset of a compound using PubChem PUG View.
    For more information please see https://pubchemdocs.ncbi.nlm.nih.gov/pug-view.
    :param CID: The PubChem compound ID (CID)
    :param type: Specifies whether the index (type='index') or complete dataset (type='data') is returned
    :param timeout: time out for the PubChem POST request (in sec)
    :return: Dictionary with the index or complete dataset
    '''
    if type == 'index':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/index/compound/{CID}/JSON'
    elif type == 'data':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{CID}/JSON'
    else:
        raise ValueError("type must be 'index' or 'data'")

    try:
        resp = requests.get(url, timeout=timeout)

        # log the request
        log_request = f'url:status -> {url}:{resp.status_code}'
        log.info(log_request)

        log.info("dynamic throttle: {throttle}".format(throttle=resp.headers['X-Throttling-Control']))

        # successful response
        if resp.status_code == 200:
            return resp.json()

        # unsuccessful response
        else:
            log.error(log_request + ". response text: " + resp.text + '. response code: ' + str(resp.status_code))
            return {}

    except requests.exceptions.RequestException as e:
        log_msg = f'requests exception for: {url}, data: {data}, exception: {str(e)}'
        log.error(log_msg)
        return {}

    except Exception as e:
        log_msg = f'requests exception for: {url}, data: {data}, exception: {str(e)}'
        log.error(log_msg)
        return {}

