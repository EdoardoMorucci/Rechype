import argparse

from Code.Modules.Report.report_creation import create_report

parser = argparse.ArgumentParser(description="Parse endpoint and client commands")

parser.add_argument("endpoint", help="the endpoint to connect to")
parser.add_argument("client", help="the name of the client")

args, kwargs = parser.parse_known_args()

# Convert list of arguments to dictionary
kwargs_dict = {}
for i in range(0, len(kwargs), 2):
    key = kwargs[i].lstrip("--")
    value = kwargs[i + 1]
    kwargs_dict[key] = value


if __name__ == "__main__":
    create_report(args.endpoint, args.client, **kwargs_dict)
