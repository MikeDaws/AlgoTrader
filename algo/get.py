#!/usr/bin/env python

import common.config
import common.args
from .view import price_to_string
import time
import argparse


def get(a):
    
    """
    Get the prices for a list of Instruments for the active Account.
    Repeatedly poll for newer prices if requested.
    """
    a=common.args.instrument(a)
    parser = argparse.ArgumentParser()

    common.config.add_argument(parser)
    args = parser.parse_args()
    account_id = args.config.active_account
    
    api = args.config.create_context()
    args.poll = 0
    latest_price_time = None 
#    print(a)
    



    response = api.pricing.get(
            account_id,
            instruments=a,
            since=latest_price_time,
            includeUnitsAvailable=False
        )
    callback=response.get("prices", 200)
    
#    a1=common.args.datetime(callback[0].time)
    
    a=callback[0].time
    b=callback[0].bids[0].price
       
    return a, b

