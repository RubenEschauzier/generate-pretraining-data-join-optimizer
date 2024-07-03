import argparse

from main import main

if __name__ == "__main__":
    default_location_blazegraph_template = "/home/rubscrub/blazegraph/endpoint_{}/"
    parser = argparse.ArgumentParser(
        prog='Random query generation',
        description='Randomly generates query over graph in .nt format',
        epilog='')
    parser.add_argument("-g", "--graph_location", help="Path to graph",required=True)
    parser.add_argument("-t", "--task_type", help="What task to execute", choices=["generation"], required=True)
    parser.add_argument("-s", "--save_location", help="Where to save results to", required=True)

    args = vars(parser.parse_args())
    location = args["graph_location"]
    task = args["task_type"]
    save_location = args["save_location"]
    main(location, task, save_location)
