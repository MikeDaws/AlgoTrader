#!/usr/bin/env python

import argparse
import common.args
from order.view import print_order_create_response_transactions
import v20


def main():
    """
    Create a Market Order in an Account based on the provided command-line
    arguments.
    """

    parser = argparse.ArgumentParser()

    #
    # Add arguments for API connection
    #
    parser.add_argument(
        "--hostname",
        default="api-fxpractice.oanda.com",
        help="v20 REST Server hostname"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=443,
        help="v20 REST Server port"
    )

    #
    # Add Account arguments
    #
    parser.add_argument(
        "accountid",
        default = "101-004-7103805-001",
        help="v20 Account ID"
    )

    parser.add_argument(
        "token",
        default = "f070579396b52b79ced6f5b1bbf69892-5255a1d785280f58bf607158a9b6fb9e",
        help="v20 Auth Token"
    )

    #
    # Add arguments for minimal Market Order
    #
    parser.add_argument(
        "instrument",
        type=common.args.instrument,
        help="The instrument to place the Market Order for"
    )

    parser.add_argument(
        "units",
        help="The number of units for the Market Order"
    )

    args = parser.parse_args()

    #
    # Create the API context based on the provided arguments
    #
    api = v20.Context(
        args.hostname,
        args.port,
        token=args.token
    )

    #
    # Submit the request to create the Market Order
    #
    response = api.order.market(
        args.accountid,
        instrument=args.instrument,
        units=args.units
    )

    #
    # Process the response
    #
    print("Response: {} ({})".format(response.status, response.reason))

    print("")

    print_order_create_response_transactions(response)


if __name__ == "__main__":
    main()
