#!/usr/bin/env python

import argparse
import common.config
import common.args
from datetime import datetime#
import numpy as np
import array

def getpast(a,b):
    """
    Create an API context, and use it to fetch candles for an instrument.

    The configuration for the context is parsed from the config file provided
    as an argumentV
    """

    parser = argparse.ArgumentParser()

    #
    # The config object is initialized by the argument parser, and contains
    # the REST APID host, port, accountID, etc.
    #
    common.config.add_argument(parser)






    """
    Get the prices for a list of Instruments for the active Account.
    Repeatedly poll for newer prices if requested.
    """

    parser = argparse.ArgumentParser()

    common.config.add_argument(parser)
    args = parser.parse_args()
    account_id = args.config.active_account
    
    api = args.config.create_context()










#
#    parser.add_argument(
#        "instrument",
#        type=common.args.instrument,
#        help="The instrument to get candles for"
#    )
#
#    parser.add_argument(
#        "--mid", 
#        action='store_true',
#        help="Get midpoint-based candles"
#    )
#
#    parser.add_argument(
#        "--bid", 
#        action='store_true',
#        help="Get bid-based candles"
#    )
#
#    parser.add_argument(
#        "--ask", 
#        action='store_true',
#        help="Get ask-based candles"
#    )
#
#    parser.add_argument(
#        "--smooth", 
#        action='store_true',
#        help="'Smooth' the candles"
#    )
#
#    parser.set_defaults(mid=False, bid=False, ask=False)
#
#    parser.add_argument(
#        "--granularity",
#        default=None,
#        help="The candles granularity to fetch"
#    )
#
#    parser.add_argument(
#        "--count",
#        default=None,
#        help="The number of candles to fetch"
#    )
#
#    date_format = "%Y-%m-%d %H:%M:%S"
#
#    parser.add_argument(
#        "--from-time",
#        default=None,
#        type=common.args.date_time(),
#        help="The start date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
#    )
#
#    parser.add_argument(
#        "--to-time",
#        default=None,
#        type=common.args.date_time(),
#        help="The end date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
#    )
#
#    parser.add_argument(
#        "--alignment-timezone",
#        default=None,
#        help="The timezone to used for aligning daily candles"
#    )

    args = parser.parse_args()

    account_id = args.config.active_account

    #
    # The v20 config object creates the v20.Context for us based on the
    # contents of the config file.
    #
    api = args.config.create_context()

    kwargs = {}

    kwargs["granularity"] = b

    
    kwargs["count"] = 5000
#
#    
#    price = "mid"
#
#    if args.mid:
#        kwargs["price"] = "M" + kwargs.get("price", "")
#        price = "mid"
#
#    if args.bid:
#        kwargs["price"] = "B" + kwargs.get("price", "")
#        price = "bid"
#
#    if args.ask:
#        kwargs["price"] = "A" + kwargs.get("price", "")
#        price = "ask"

    #
    # Fetch the candles
    #
    response = api.instrument.candles(a, **kwargs)

   

    print("Instrument: {}".format(response.get("instrument", 200)))
    print("Granularity: {}".format(response.get("granularity", 200)))
#
#    printer = CandlePrinter()

#
#    printer.print_header()

    candles = response.get("candles", 200)
#    A = np.ndarray(shape=(1,4), dtype=float)
#    A = np.zeros((1,4)
    A = np.zeros( (5000, 6) )
    B = []
    for ii in range(0,5000):
        if candles[ii].complete==True:
            A[ii,0] = candles[ii].mid.o
            A[ii,1] = candles[ii].mid.h
            A[ii,2] = candles[ii].mid.l
            A[ii,3] = candles[ii].mid.c
            A[ii,4] = candles[ii].volume
            
            tempstore=datetime.strptime(candles[ii].time,"%Y-%m-%dT%H:%M:%S.000000000Z")
            A[ii,5] = tempstore.hour
#        printer.print_candle(candle)
    return A
#
#if __name__ == "__main__":
#    main()
    
