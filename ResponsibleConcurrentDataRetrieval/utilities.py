# setup logging
import pydoc_data.topics

import pandas as pd

import logger
log = logger.get_logger(__name__)

import urllib
import unicodedata
from typing import List

greek_alphabet = 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'

def encode_identifier(identifier: str) -> str:
    '''
    It prepares an identifier so that it can be passed to PUG REST.
    The following operations are performed:
    - greek letters are converted to their full english names (in lower case)
    - URL encoding
    :param identifier: input identifier
    :return: converted identifier that can be passed to PUG REST
    '''
    converted_identifier = ''
    for s in identifier:
        if s in greek_alphabet:
            converted_identifier += unicodedata.name(s).split()[-1].lower()
        else:
            converted_identifier += s
    converted_identifier = urllib.parse.quote(converted_identifier)
    return converted_identifier



def parse_TOCHeadings(sections: List, TOCHeading_path='', TOCHeading_paths = None):
    '''
    Returns the TOCHeadings in the index or full dataset returned by PUG view.
    THe function works recursively.

    :param sections: Pug view returns a dictionary in which the TOCHeadings are in the value ['Record']['Section']
    :param TOCHeading_path: not to be used (needed for recursion)
    :param TOCHeading_paths: not to be used (needed for recursion)
    :return:array of TOCHeadings in the index or full dataaset returned by PUG view
    '''
    if TOCHeading_paths is None:
        TOCHeading_paths = []
    for section in sections:
        tmp = TOCHeading_path + ('->' if TOCHeading_path else '') + section['TOCHeading']
        if 'Section' in section:
            parse_TOCHeadings(section['Section'], tmp, TOCHeading_paths)
        else:
            TOCHeading_paths.append(tmp)
    return TOCHeading_paths


def visualise_task_orchestration() -> None:
    '''Produces figures 2-5'''
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    rng = np.random.default_rng(15)
    random.seed(15)
    n_tasks = 5
    n_requests = 10
    for i_ax, mean_request_time in enumerate([0.2, 0.5, 1., 1.5]):
        fig = plt.figure(figsize=(10, 3), dpi=300)
        ax = fig.subplots(1, 1)
        history = []
        for i_task in range(n_tasks):
            start = 0.
            for i_request in range(n_requests):
                duration = max(rng.normal(loc=mean_request_time, scale=mean_request_time/4), 0.05)
                sleep_time = 1-duration if duration<1. else 0.
                end = start + duration
                history.append({'task': i_task, 'request': i_request, 'start': start, 'end':end})
                start = end + sleep_time
        history = pd.DataFrame(history)

        average_compute_time = []
        average_requests_per_sec = []
        for i_task in range(n_tasks):
            msk = (history['task'] == i_task)
            total_computing_time = (history.loc[msk,'end']-history.loc[msk,'start']).sum()
            total_wall_time = history.loc[msk,'end'].iloc[-1]-history.loc[msk,'start'].iloc[0]
            # account for the sleep time of the last request
            if history.loc[msk,'end'].iloc[-1]-history.loc[msk,'start'].iloc[-1]<1.:
                total_wall_time += 1. - (history.loc[msk,'end'].iloc[-1]-history.loc[msk,'start'].iloc[-1])
            summary = f"response time {100*total_computing_time/total_wall_time:.1f} % of wall time\n{n_requests/total_wall_time:.2f} requests per sec"
            average_compute_time.append(100*total_computing_time/total_wall_time)
            average_requests_per_sec.append(n_requests/total_wall_time)
            ax.text(history.loc[msk,'end'].iloc[-1]+0.4, i_task + 1, summary,
                    horizontalalignment='left',
                    verticalalignment='center', fontsize='small')
            ax.scatter(x=history.loc[msk,'start'], y=[i_task+1]*msk.sum(), s=20, facecolors='none', edgecolors='k')
            # ax.scatter(x=history.loc[msk,'end'], y=[i_task+1]*msk.sum(), s=20, facecolors='k', edgecolors='k')
            for _, row in history.loc[msk].iterrows():
                ax.plot([row['start'], row['end']], [i_task+1, i_task+1], linestyle='-', color='k')
                ax.text((row['start']+row['end'])/2., i_task+1+0.2, f"{row['end']-row['start']:.2f}", horizontalalignment='center',
                        verticalalignment='center', fontsize='small')
        ax.set_title(f'response time: mean {mean_request_time:0.2f}, std: {mean_request_time/4:0.2f}')
        # set ylim
        fig.canvas.draw()
        ax.set_yticks(range(1, n_tasks+1),[f'stream {i_task}' for i_task in range(1, n_tasks+1)])
        ax.set_xticks(range(0, math.ceil(history['end'].max())+2))
        ax.set_ylim([0.5, n_tasks+0.5])
        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # labels = [f'stream {label}' if int(label)>=1 and int(label)<=n_tasks else '' for label in labels]
        # ax.set_yticklabels(labels)
        # Hide the right, top, and left spines
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('time (sec)')
        fig.tight_layout()
        fig.savefig(fr'output/task_orchestration_{mean_request_time}.png')
        plt.close(fig)


def visualise_reponse_times():
    '''Produces figure 6'''
    # box plot of response times
    import re
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    pat_cid = re.compile(r'cids.*response time ([^,]+),')
    pat_pugview = re.compile(r'pug_view.*response time ([^,]+),')
    response_times = []
    with open(r'output/PubChem_integration.log', mode='rt') as flog:
        for line in flog:
            response_time = pat_cid.search(line)
            if response_time:
                response_times.append({'request type': 'retrieve_CID', 'response time': float(response_time.group(1))})
            elif response_time := pat_pugview.search(line):
                response_times.append({'request type': 'retrieve_pugview', 'response time': float(response_time.group(1))})
    response_times = pd.DataFrame(response_times)
    fig = plt.figure(figsize=(8, 4), dpi=300)
    ax = fig.subplots(1,1)
    sns.boxplot(ax=ax, data=response_times, y='request type', x='response time', fliersize=2)
    ax.set_xlabel('response time (sec)')
    fig.tight_layout()
    fig.savefig(r'output/response_times.png')
    plt.close(fig)

